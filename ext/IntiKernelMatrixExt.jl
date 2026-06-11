module IntiKernelMatrixExt

import Inti
import KernelAbstractions as KA

using StaticArrays
using LinearAlgebra
using KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize, @private, @uniform

function __init__()
    return @debug "Loading Inti.jl KernelAbstractions (matrix-free) extension"
end

_node_normal(q::AbstractVector) = zero(q)
function _node_normal(q::Inti.QuadratureNode)
    n = Inti.normal(q)
    return isnothing(n) ? zero(Inti.coords(q)) : n
end

# Default tile shape. On an M3 Pro (Float32, N≈92k) performance is flat for
# workgroupsize in 32…128 (degrades ≥256) and for targets_per_lane in 2…8; other
# architectures may prefer different values, hence the keywords below.
const KMM_TG = 64   # lanes per workgroup == sources staged per shared-memory tile
const KMM_TB = 4    # targets owned by each lane (register blocking)

"""
    KernelMatrix{T} <: AbstractMatrix{T}

Matrix-free representation of an [`Inti.IntegralOperator`](@ref): applies the
operator in `O(N²)` work and `O(N)` memory on a `KernelAbstractions` backend,
without assembling the dense matrix. Built via
[`Inti.assemble_kernelmatrix`](@ref). Use `mul!`/`*` for fast batched evaluation;
scalar `getindex` is supported but slow.
"""
struct KernelMatrix{T, Op, B, VC, VW} <: AbstractMatrix{T}
    iop::Op        # the IntegralOperator's kernel, called as K(target, source)
    backend::B
    tcoords::VC    # device vectors of coords/normals (zero normal where absent)
    tnormals::VC
    scoords::VC
    snormals::VC
    weights::VW    # device vector of quadrature weights
    workgroupsize::Int
    targets_per_lane::Int
end

Inti.kernel(A::KernelMatrix) = Inti.kernel(A.iop)
Base.size(A::KernelMatrix) = size(A.iop)
Base.getindex(A::KernelMatrix, args...) = getindex(A.iop, args...)

function KernelMatrix(
        iop::Inti.IntegralOperator;
        backend = KA.CPU(),
        workgroupsize::Integer = KMM_TG,
        targets_per_lane::Integer = KMM_TB,
    )
    X = Inti.target(iop)
    Y = Inti.source(iop)
    T = eltype(iop)

    tc = KA.adapt(backend, map(Inti.coords, X))
    tn = KA.adapt(backend, map(_node_normal, X))
    sc = KA.adapt(backend, map(Inti.coords, Y))
    sn = KA.adapt(backend, map(_node_normal, Y))
    w = KA.adapt(backend, map(Inti.weight, Y))

    return KernelMatrix{T, typeof(iop), typeof(backend), typeof(tc), typeof(w)}(
        iop, backend, tc, tn, sc, sn, w, workgroupsize, targets_per_lane,
    )
end

# Public entry point (method on the core stub).
function Inti.assemble_kernelmatrix(iop::Inti.IntegralOperator; kwargs...)
    return KernelMatrix(iop; kwargs...)
end

# Tiled matvec, portable across the KA CPU and GPU backends. Each workgroup of TG
# lanes cooperatively stages TG sources into shared memory; each lane owns TB targets
# and reuses every staged source TB times from registers, which amortizes the
# shared-memory reads and hides the rsqrt latency. On the CPU backend only
# `@uniform`/`@private`/`@localmem` storage survives a `@synchronize` barrier, hence
# the annotations (free on the GPU).
@kernel function _kmm_mul!(
        y, @Const(wx), β, K,
        @Const(tc), @Const(tn), @Const(sc), @Const(sn),
        ::Val{TG}, ::Val{TB},
    ) where {TG, TB}
    gi = @index(Group)
    il = @index(Local)
    @uniform m = length(y)
    @uniform n = length(wx)

    lsc = @localmem eltype(sc) (TG,)
    lsn = @localmem eltype(sn) (TG,)
    lwx = @localmem eltype(wx) (TG,)

    # Lane il of group gi owns TB lane-contiguous (coalesced) sub-blocks of targets;
    # out-of-range slots are clamped to a valid target and masked on write-back.
    acc = @private eltype(y) (TB,)
    tci = @private eltype(tc) (TB,)
    tni = @private eltype(tn) (TB,)
    @inbounds for b in 1:TB
        acc[b] = zero(eltype(y))
        i = (gi - 1) * TB * TG + (b - 1) * TG + il
        tci[b] = tc[min(i, m)]
        tni[b] = tn[min(i, m)]
    end
    @inbounds for tile in 0:TG:(n - 1)
        j = tile + il
        ok = j ≤ n
        # Out-of-range slots stage a zero source with zero weight (contributes 0),
        # so every lane participates and the barrier holds.
        lsc[il] = ok ? sc[j] : zero(eltype(sc))
        lsn[il] = ok ? sn[j] : zero(eltype(sn))
        lwx[il] = ok ? wx[j] : zero(eltype(wx))
        @synchronize
        for k in 1:TG
            sj = (coords = lsc[k], normal = lsn[k])
            for b in 1:TB
                ti = (coords = tci[b], normal = tni[b])
                # == K(ti, sj) * lwx[k] up to the constant kernel_prefactor(K), which
                # mul! folds into wx; low-rank matrix-valued kernels (Stokes, ...)
                # compute the action without forming the per-pair matrix.
                acc[b] += Inti.apply_kernel_unscaled(K, ti, sj, lwx[k])
            end
        end
        @synchronize
    end
    # y = acc + β·y; when β == 0, y must not be read (it may be uninitialized).
    @inbounds for b in 1:TB
        i = (gi - 1) * TB * TG + (b - 1) * TG + il
        i ≤ m && (y[i] = iszero(β) ? acc[b] : acc[b] + β * y[i])
    end
end

# Move a host vector to the device; device-resident input is used as-is. Host
# wrappers (views, reinterpreted vectors) are materialized first: adapting one would
# upload its whole parent, or fail outright on some backends.
function _on_device(backend, v)
    KA.get_backend(v) == backend && return v
    return KA.adapt(backend, v isa Array ? v : Array(v))
end

# Convert a scalar to the device's real precision, so that e.g. a Float64 α does not
# promote a Float32 device computation (which some backends reject outright).
_to_precision(::Type{T}, a::Complex) where {T} = convert(Complex{T}, a)
_to_precision(::Type{T}, a::Number) where {T} = convert(T, a)

# Primary implementation: y = α·A·x + β·y, with x and y each living on the host or
# on A's backend; device-resident vectors are used in place, so passing both runs
# entirely on the device. The 3-arg mul! comes from LinearAlgebra's generic fallback.
function LinearAlgebra.mul!(
        y::AbstractVector, A::KernelMatrix, x::AbstractVector, α::Number, β::Number,
    )
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), expected $(n)"))
    length(y) == m || throw(DimensionMismatch("y has length $(length(y)), expected $(m)"))
    backend = A.backend
    K = Inti.kernel(A)
    # Fold α, the quadrature weights, and the constant kernel prefactor into the
    # density once, so the O(N²) loop runs the unscaled action only.
    Tw = eltype(A.weights)
    c = _to_precision(Tw, Inti.kernel_prefactor(K)) * _to_precision(Tw, α)
    wx = c .* A.weights .* _on_device(backend, x)
    R = Base.promote_op(*, eltype(A), eltype(wx))     # SVector for Stokes, scalar for Laplace
    ondevice = KA.get_backend(y) == backend
    ydev = ondevice ? y : KA.allocate(backend, R, m)  # scratch is fully overwritten
    β′ = ondevice ? _to_precision(Tw, β) : zero(Tw)
    tg, tb = A.workgroupsize, A.targets_per_lane
    # ndrange is padded to whole workgroups: a partial group would skip the
    # @synchronize barrier on some lanes (-> deadlock).
    _kmm_mul!(backend, tg)(
        ydev, wx, β′, K, A.tcoords, A.tnormals, A.scoords, A.snormals, Val(tg), Val(tb);
        ndrange = cld(m, tb * tg) * tg,
    )
    KA.synchronize(backend)
    if !ondevice
        Axh = copyto!(Vector{R}(undef, m), ydev)
        iszero(β) ? (y .= Axh) : (@. y = Axh + β * y)
    end
    return y
end

# Matrix right-hand sides: apply the device matvec column by column. Without this,
# LinearAlgebra's generic fallback would evaluate the operator entry by entry on the
# host through getindex.
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
