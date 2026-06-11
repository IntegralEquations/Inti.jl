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

# Threadgroup size: KMM_TG work-items cooperate to stage a tile of KMM_TG sources into
# on-chip shared memory before reading them back from there. Each work-item owns KMM_TB
# targets ("register blocking"): every staged source is reused KMM_TB times from
# registers, which both amortizes the shared-memory reads and gives the scheduler
# independent arithmetic to hide the rsqrt latency. Empirically (M3 Pro, Float32,
# N≈92k): Laplace DL 121→159 Gpairs/s, Stokes DL 69→75; performance is flat for
# KMM_TG in 32…128 and degrades ≥256, and flat for KMM_TB in 2…8 — so the values
# below are not worth exposing as user-facing knobs.
const KMM_TG = 64
const KMM_TB = 4

# Single matrix-free matvec kernel, portable across the KA CPU and GPU backends.
#
# Portability note: on the CPU backend KA emulates each `@synchronize` by splitting the
# kernel into regions and looping over work-items per region; only kernel arguments,
# `@index`, `@uniform` and `@localmem`/`@private` storage survive across a barrier
# (plain locals do not). Hence `m`/`n` are `@uniform` and the accumulators and
# per-thread target data live in `@private` storage. On the GPU none of this matters
# (registers persist across barriers); the annotations are free.
@kernel function _kmm_mul!(
        y, @Const(wx), K,
        @Const(tc), @Const(tn), @Const(sc), @Const(sn),
    )
    gi = @index(Group)         # my threadgroup
    il = @index(Local)         # my lane within the threadgroup, 1 … KMM_TG
    @uniform m = length(y)
    @uniform n = length(wx)

    # Shared-memory tile: filled cooperatively (one slot per lane), read by everyone.
    lsc = @localmem eltype(sc) (KMM_TG,)   # source coords
    lsn = @localmem eltype(sn) (KMM_TG,)   # source normals
    lwx = @localmem eltype(wx) (KMM_TG,)   # weighted density (weights already folded in)

    # This thread's KMM_TB targets: group gi covers targets (gi-1)*KMM_TB*KMM_TG+1 …
    # gi*KMM_TB*KMM_TG, in KMM_TB lane-contiguous (coalesced) sub-blocks of KMM_TG.
    # `min(i, m)` clamps out-of-range slots to a valid target; their result is
    # computed but never written back (guard at the end).
    acc = @private eltype(y) (KMM_TB,)     # accumulators (persist across barriers)
    tci = @private eltype(tc) (KMM_TB,)    # target coords
    tni = @private eltype(tn) (KMM_TB,)    # target normals
    @inbounds for b in 1:KMM_TB
        acc[b] = zero(eltype(y))
        i = (gi - 1) * KMM_TB * KMM_TG + (b - 1) * KMM_TG + il
        tci[b] = tc[min(i, m)]
        tni[b] = tn[min(i, m)]
    end
    @inbounds for tile in 0:KMM_TG:(n - 1)
        j = tile + il                      # this lane stages source j
        ok = j ≤ n
        # Out-of-range slots get a zero source with zero weight -> contribute 0
        # (K is finite, so 0 * K = 0). Every lane participates, so the barrier holds.
        lsc[il] = ok ? sc[j] : zero(eltype(sc))
        lsn[il] = ok ? sn[j] : zero(eltype(sn))
        lwx[il] = ok ? wx[j] : zero(eltype(wx))
        @synchronize                       # tile fully staged before anyone reads
        for k in 1:KMM_TG
            sj = (coords = lsc[k], normal = lsn[k])
            for b in 1:KMM_TB
                ti = (coords = tci[b], normal = tni[b])
                # `apply_kernel_unscaled(K, ti, sj, v)` == `K(ti, sj) * v` up to the
                # constant `kernel_prefactor(K)`, which `mul!` folds into `wx` on the
                # host. For matrix-valued kernels with low-rank structure (Stokes,
                # Elastostatic) it computes the action directly, never assembling the
                # per-pair matrix; kernels without a specialization fall back to
                # forming the kernel value and multiplying.
                acc[b] += Inti.apply_kernel_unscaled(K, ti, sj, lwx[k])
            end
        end
        @synchronize                       # all done reading before next overwrite
    end
    @inbounds for b in 1:KMM_TB
        i = (gi - 1) * KMM_TB * KMM_TG + (b - 1) * KMM_TG + il
        i ≤ m && (y[i] = acc[b])           # masked write-back
    end
end

function LinearAlgebra.mul!(y::AbstractVector, A::KernelMatrix, x::AbstractVector)
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), expected $(n)"))
    length(y) == m || throw(DimensionMismatch("y has length $(length(y)), expected $(m)"))
    backend = A.backend
    K = Inti.kernel(A)
    # Device density: quadrature weights and the constant kernel prefactor are folded
    # in once here, so the O(N²) loop runs the unscaled action only. The prefactor is
    # converted to the weight precision (it may be an exact Float64 like 1/4π).
    c = convert(eltype(A.weights), Inti.kernel_prefactor(K))
    # Materialize views/reinterpreted vectors (e.g. the column slices `bdim_correction`
    # passes for vector-valued problems) before the device transfer: adapting a wrapper
    # would upload its whole parent, or fail outright on some backends.
    xh = x isa Array ? x : Array(x)
    wx = c .* A.weights .* KA.adapt(backend, xh)
    R = Base.promote_op(*, eltype(A), eltype(wx))     # SVector for Stokes, scalar for Laplace
    ydev = KA.zeros(backend, R, m)
    # Fixed workgroupsize KMM_TG (matches the @localmem tile); each group handles
    # KMM_TB*KMM_TG targets. ndrange is padded to a whole number of groups so KA
    # launches no partial group whose extra lanes would skip the @synchronize barrier
    # (-> deadlock).
    _kmm_mul!(backend, KMM_TG)(
        ydev, wx, K, A.tcoords, A.tnormals, A.scoords, A.snormals;
        ndrange = cld(m, KMM_TB * KMM_TG) * KMM_TG,
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

# Matrix right-hand sides (e.g. the monopole traces in `bdim_correction`): apply the
# device matvec column by column. Without this method, `mul!` with a matrix falls back
# to LinearAlgebra's generic matmul, which evaluates the operator entry by entry on the
# host through `getindex` — O(m·n) kernel evaluations per column, never touching the
# device.
function LinearAlgebra.mul!(
        Y::AbstractMatrix, A::KernelMatrix, X::AbstractMatrix, α::Number, β::Number,
    )
    size(Y, 2) == size(X, 2) ||
        throw(DimensionMismatch("Y has $(size(Y, 2)) columns, X has $(size(X, 2))"))
    for k in axes(X, 2)
        mul!(view(Y, :, k), A, view(X, :, k), α, β)
    end
    return Y
end

end # module
