module IntiQPGreenExt

import Inti
import QPGreen
import LinearAlgebra
import ForwardDiff
import StaticArrays: SMatrix

function __init__()
    return @info "Loading Inti.jl QPGreen extension"
end

struct HelmholtzPeriodic1D{
        N,
        NT <: NamedTuple,
        F1,
        F2 <: NamedTuple,
        F3 <: NamedTuple,
        C1 <: QPGreen.AbstractIntegrationCache,
    } <: Inti.AbstractDifferentialOperator{N}
    params::NT
    val_interp::F1
    grad_interp::F2
    hess_interp::F3
    Yε_cache::C1
end

"""
    HelmholtzPeriodic1D(; alpha, k, dim)

Helmholtz's differential operator `-Δu - k²u` in `dim` dimension with periodic boundary 
conditions along the first dimension. 

The real parameters `alpha` and `k` represent the quasiperiod and the wave number. 
The period is fixed to `2π` and the periodic cell is defined as `[-π, π]`.

Note that at the moment, there is no control over parameters c, c_tilde, epsilon, order
and grid_size since it might add some complexity to the interface (but we can discuss 
whether we want to give users control over these parameters in the future.)
"""
function Inti.HelmholtzPeriodic1D(; alpha, k, dim)
    params = (alpha = alpha, k = k, c = 0.6, c_tilde = 1.0, epsilon = 0.45, order = 8)
    grid_size = 1024
    val_interp, grad_interp, hess_interp, Yε_cache =
        QPGreen.init_qp_green_fft(params, grid_size; grad = true, hess = true)
    return HelmholtzPeriodic1D{
        dim,
        typeof(params),
        typeof(val_interp),
        typeof(grad_interp),
        typeof(hess_interp),
        typeof(Yε_cache),
    }(
        params,
        val_interp,
        grad_interp,
        hess_interp,
        Yε_cache,
    )
end

function Base.show(io::IO, op::HelmholtzPeriodic1D{N}) where {N}
    return print(
        io,
        "Periodic Helmholtz operator -Δu-k²u in $N dimensions with periodic conditions along the first dimension",
    )
end

Inti.default_kernel_eltype(::HelmholtzPeriodic1D) = ComplexF64
Inti.default_density_eltype(::HelmholtzPeriodic1D) = ComplexF64

function (SL::Inti.SingleLayerKernel{T, <:HelmholtzPeriodic1D{N}})(
        target,
        source,
        r = Inti.coords(target) - Inti.coords(source),
    ) where {N, T}
    d = LinearAlgebra.norm(r)
    (d ≤ Inti.SAME_POINT_TOLERANCE) && return zero(T)
    if N == 2
        return QPGreen.eval_qp_green(r, SL.op.params, SL.op.val_interp, SL.op.Yε_cache)
    else
        error(
            "Single layer kernel for HelmholtzPeriodic1D not implemented in $N dimensions",
        )
    end
end

function (DL::Inti.DoubleLayerKernel{T, <:HelmholtzPeriodic1D{N}})(
        target,
        source,
        r = Inti.coords(target) - Inti.coords(source),
    ) where {N, T}
    d = LinearAlgebra.norm(r)
    (d ≤ Inti.SAME_POINT_TOLERANCE) && return zero(T)
    ny = Inti.normal(source)
    if N == 2
        grad = -QPGreen.grad_qp_green(r, DL.op.params, DL.op.grad_interp, DL.op.Yε_cache)
        out = grad[1] * ny[1] + grad[2] * ny[2]
        return out
    else
        error(
            "Double layer kernel for HelmholtzPeriodic1D not implemented in $N dimensions",
        )
    end
end

function (ADL::Inti.AdjointDoubleLayerKernel{T, <:HelmholtzPeriodic1D{N}})(
        target,
        source,
        r = Inti.coords(target) - Inti.coords(source),
    ) where {N, T}
    d = LinearAlgebra.norm(r)
    (d ≤ Inti.SAME_POINT_TOLERANCE) && return zero(T)
    nx = Inti.normal(target)
    if N == 2
        grad = QPGreen.grad_qp_green(r, ADL.op.params, ADL.op.grad_interp, ADL.op.Yε_cache)
        out = grad[1] * nx[1] + grad[2] * nx[2]
        return out
    else
        error(
            "Adjoint double layer kernel for HelmholtzPeriodic1D not implemented in $N dimensions",
        )
    end
end

function (HS::Inti.HyperSingularKernel{T, <:HelmholtzPeriodic1D{N}})(
        target,
        source,
        r = Inti.coords(target) - Inti.coords(source),
    ) where {N, T}
    d = LinearAlgebra.norm(r)
    (d ≤ Inti.SAME_POINT_TOLERANCE) && return zero(T)
    nx = Inti.normal(target)
    ny = Inti.normal(source)
    if N == 2
        ### Alternative via AutoDiff (only working if evaluate Hankel function with SpecialFunctions.jl)
        # x = Inti.coords(target)
        # dGdny = Inti.DoubleLayerKernel(HS.op)
        # ForwardDiff.derivative(t -> dGdny(x + t * nx, source), 0)
        hess = QPGreen.hess_qp_green(r, HS.op.params, HS.op.hess_interp, HS.op.Yε_cache)
        hess_matrix = SMatrix{2, 2}(hess[1], hess[2], hess[2], hess[3])
        out = -transpose(nx) * hess_matrix * ny
        return out
    else
        return error(
            "Hypersingular kernel for HelmholtzPeriodic1D not implemented in $N dimensions",
        )
    end
end

end # module
