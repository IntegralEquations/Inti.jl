module IntiKernelMatrixExt

import Inti
import KernelAbstractions as KA

using StaticArrays
using LinearAlgebra
using KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize, @private, @uniform

function __init__()
    return @debug "Loading Inti.jl KernelAbstractions (matrix-free) extension"
end

function _node_normal(q::AbstractVector)
    return zero(q)
end
function _node_normal(q::Inti.QuadratureNode{N, T}) where {N, T}
    n = Inti.normal(q)
    if isnothing(n)
        return zero(Inti.coords(q))
    else
        return n
    end
end

"""
    KernelMatrix{T} <: AbstractMatrix{T}

Matrix-free representation of an [`Inti.IntegralOperator`](@ref): it applies the
operator in `O(N²)` work and `O(N)` memory on a `KernelAbstractions` backend,
without ever assembling the dense matrix. Built via
[`Inti.assemble_kernelmatrix`](@ref). Use `mul!`/`*` for fast batched evaluation;
scalar `getindex` is supported but slow (it copies a node back from the device).
"""
struct KernelMatrix{T, Op, B, VC, VW} <: AbstractMatrix{T}
    iop::Op       # the IntegralOperator's kernel, called as K(target, source)
    backend::B
    tcoords::VC     # device Vector{SVector{N,Tc}}  (targets)
    tnormals::VC    # device Vector{SVector{N,Tc}}  (zeros where absent)
    scoords::VC     # device Vector{SVector{N,Tc}}  (sources)
    snormals::VC    # device Vector{SVector{N,Tc}}
    weights::VW     # device Vector{Tc}             (quadrature weights)
end

Inti.kernel(A::KernelMatrix) = Inti.kernel(A.iop)
Base.size(A::KernelMatrix) = size(A.iop)
Base.getindex(A::KernelMatrix, args...) = getindex(A.iop, args...)

function KernelMatrix(iop::Inti.IntegralOperator; backend = KA.CPU())
    X = Inti.target(iop)
    Y = Inti.source(iop)
    T = eltype(iop)

    tc = KA.adapt(backend, map(Inti.coords, X))
    tn = KA.adapt(backend, map(_node_normal, X))
    sc = KA.adapt(backend, map(Inti.coords, Y))
    sn = KA.adapt(backend, map(_node_normal, Y))
    w = KA.adapt(backend, map(Inti.weight, Y))

    return KernelMatrix{T, typeof(iop), typeof(backend), typeof(tc), typeof(w)}(
        iop, backend, tc, tn, sc, sn, w,
    )
end

# Public entry point (method on the core stub).
function Inti.assemble_kernelmatrix(iop::Inti.IntegralOperator; backend = KA.CPU())
    return KernelMatrix(iop; backend)
end

# Threadgroup size: one work-item per target, KMM_TG of them cooperate to stage a tile
# of KMM_TG sources into on-chip shared memory before reading them back from there.
const KMM_TG = 256

# Single matrix-free matvec kernel, portable across the KA CPU and GPU backends.
#
# Portability note: on the CPU backend KA emulates each `@synchronize` by splitting the
# kernel into regions and looping over work-items per region; only kernel arguments,
# `@index`, `@uniform` and `@localmem`/`@private` storage survive across a barrier
# (plain locals do not). Hence `m`/`n` are `@uniform`, the accumulator is `@private`,
# and `ti` is recomputed inside the tile loop rather than hoisted above it. On the GPU
# none of this matters (registers persist across barriers); the annotations are free.
@kernel function _kmm_mul!(
        y, @Const(wx), K,
        @Const(tc), @Const(tn), @Const(sc), @Const(sn),
    )
    i = @index(Global)        # my target (global index; may exceed m on padding lanes)
    il = @index(Local)         # my lane within the threadgroup, 1 … KMM_TG
    @uniform m = length(y)
    @uniform n = length(wx)

    # Shared-memory tile: filled cooperatively (one slot per lane), read by everyone.
    lsc = @localmem eltype(sc) (KMM_TG,)   # source coords
    lsn = @localmem eltype(sn) (KMM_TG,)   # source normals
    lwx = @localmem eltype(wx) (KMM_TG,)   # weighted density (weights already folded in)

    acc = @private eltype(y) (1,)          # accumulator (must persist across barriers)
    @inbounds acc[1] = zero(eltype(y))
    @inbounds for tile in 0:KMM_TG:(n - 1)
        j = tile + il                      # this lane stages source j
        ok = j ≤ n
        # Out-of-range slots get a zero source with zero weight -> contribute 0
        # (K is finite, so 0 * K = 0). Every lane participates, so the barrier holds.
        lsc[il] = ok ? sc[j] : zero(eltype(sc))
        lsn[il] = ok ? sn[j] : zero(eltype(sn))
        lwx[il] = ok ? wx[j] : zero(eltype(wx))
        @synchronize                       # tile fully staged before anyone reads
        # `min(i, m)` clamps padding lanes (i > m) to a valid target; their result is
        # computed but never written back (guard below).
        ti = (coords = tc[min(i, m)], normal = tn[min(i, m)])
        for k in 1:KMM_TG
            sj = (coords = lsc[k], normal = lsn[k])
            # `apply_kernel(K, ti, sj, v)` == `K(ti, sj) * v`, but for matrix-valued
            # kernels with low-rank structure (Stokes) it computes the action
            # directly, never assembling the per-pair matrix. Laplace flows through
            # the generic fallback unchanged.
            acc[1] += Inti.apply_kernel(K, ti, sj, lwx[k])
        end
        @synchronize                       # all done reading before next overwrite
    end
    @inbounds i ≤ m && (y[i] = acc[1])     # masked write-back
end

function LinearAlgebra.mul!(y::AbstractVector, A::KernelMatrix, x::AbstractVector)
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), expected $(n)"))
    length(y) == m || throw(DimensionMismatch("y has length $(length(y)), expected $(m)"))
    backend = A.backend
    wx = A.weights .* KA.adapt(backend, x)            # device density, weights folded in
    R = Base.promote_op(*, eltype(A), eltype(wx))     # SVector for Stokes, scalar for Laplace
    ydev = KA.zeros(backend, R, m)
    K = Inti.kernel(A)
    # Fixed workgroupsize KMM_TG (matches the @localmem tile); ndrange padded to a whole
    # number of groups so KA launches no partial group whose extra lanes would skip the
    # @synchronize barrier (-> deadlock).
    _kmm_mul!(backend, KMM_TG)(
        ydev, wx, K, A.tcoords, A.tnormals, A.scoords, A.snormals;
        ndrange = cld(m, KMM_TG) * KMM_TG,
    )
    KA.synchronize(backend)
    copyto!(y, ydev)
    return y
end

# 5-arg mul! (y = α·A·x + β·y) — used by Krylov solvers.
function LinearAlgebra.mul!(
        y::AbstractVector, A::KernelMatrix, x::AbstractVector, α::Number, β::Number,
    )
    tmp = A * x
    @. y = α * tmp + β * y
    return y
end

end # module
