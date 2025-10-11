"""
    struct IntegralPotential

Represent a potential given by a `kernel` and a `quadrature` over which
integration is performed.

`IntegralPotential`s are created using `IntegralPotential(kernel, quadrature)`.

Evaluating an integral potential requires a density `σ` (defined over the
quadrature nodes of the source mesh) and a point `x` at which to evaluate the
integral
```math
\\int_{\\Gamma} K(\boldsymbol{x},\boldsymbol{y})\\sigma(y) ds_y, x \\not \\in \\Gamma
```

Assuming `𝒮` is an integral potential and `σ` is a vector of values defined on
`quadrature`, calling `𝒮[σ]` creates an anonymous function that can be
evaluated at any point `x`.
"""
struct IntegralPotential{K, Q <: Quadrature}
    kernel::K
    quadrature::Q
end

function Base.getindex(pot::IntegralPotential, σ::AbstractVector)
    K = pot.kernel
    Q = pot.quadrature.qnodes
    return (x) -> _evaluate_potential(K, σ, x, Q)
end

@noinline function _evaluate_potential(K, σ, x, Q)
    iter = zip(Q, σ)
    out = sum(iter) do (qi, σi)
        wi = weight(qi)
        return K(x, qi) * σi * wi
    end
    return out
end

"""
    SingleLayerPotential(op::AbstractDifferentialOperator, source::Quadrature)

An [`IntegralPotential`](@ref) over `source` with kernel given by `SingleLayerKernel(op)`.
"""
function SingleLayerPotential(op::AbstractDifferentialOperator, source::Quadrature)
    return IntegralPotential(SingleLayerKernel(op), source)
end

"""
    DoubleLayerPotential(op::AbstractDifferentialOperator, source::Quadrature)

An [`IntegralPotential`](@ref) over `source` with kernel given by `DoubleLayerKernel(op)`.
"""
function DoubleLayerPotential(op::AbstractDifferentialOperator, source::Quadrature)
    return IntegralPotential(DoubleLayerKernel(op), source)
end

"""
    struct IntegralOperator{T} <: AbstractMatrix{T}

A discrete linear integral operator given by
```math
I[u](x) = \\int_{\\Gamma\\_s} K(x,y)u(y) ds_y, x \\in \\Gamma_{t}
```
where ``\\Gamma_s`` and ``\\Gamma_t`` are the source and target domains, respectively.
"""
struct IntegralOperator{V, K, T, S <: Quadrature} <: AbstractMatrix{V}
    kernel::K
    # since the target can be as simple as a vector of points, leave it untyped
    target::T
    # the source, on the other hand, has to be a quadrature for our Nystrom method
    source::S
end

kernel(iop::IntegralOperator) = iop.kernel
target(iop::IntegralOperator) = iop.target
source(iop::IntegralOperator) = iop.source

function IntegralOperator(k, X, Y::Quadrature = X)
    # check that all entities in the quadrature are of the same dimension
    if !allequal(geometric_dimension(ent) for ent in entities(Y))
        msg = "entities in the target quadrature have different geometric dimensions"
        throw(ArgumentError(msg))
    end
    T = return_type(k, eltype(X), eltype(Y))
    msg = """IntegralOperator of nonbits being created: $T"""
    isbitstype(T) || (@warn msg)
    return IntegralOperator{T, typeof(k), typeof(X), typeof(Y)}(k, X, Y)
end

Base.size(iop::IntegralOperator) = (length(iop.target), length(iop.source))

function Base.getindex(iop::IntegralOperator, i::Integer, j::Integer)
    k = kernel(iop)
    return k(iop.target[i], iop.source[j]) * weight(iop.source[j])
end

"""
    assemble_matrix(iop::IntegralOperator; threads = true)

Assemble a dense matrix representation of an `IntegralOperator`.
"""
function assemble_matrix(iop::IntegralOperator; threads = true)
    T = eltype(iop)
    m, n = size(iop)
    out = if T <: SMatrix
        BlockArray{T}(undef, m, n)
    else
        Array{T}(undef, m, n)
    end
    K = kernel(iop)
    # function barrier
    _assemble_matrix!(out, K, iop.target, iop.source, threads)
    return out
end

@noinline function _assemble_matrix!(out, K, X, Y::Quadrature, threads)
    @usethreads threads for j in 1:length(Y)
        for i in 1:length(X)
            out[i, j] = K(X[i], Y[j]) * weight(Y[j])
        end
    end
    return out
end

"""
    assemble_fmm(iop; rtol)

Set up a 2D or 3D FMM for evaluating the discretized integral operator `iop` associated with
the `op`. In 2D the `FMM2D` or `FMMLIB2D` library is used (whichever was most recently
loaded) while in 3D `FMM3D` is used.

!!! warning "FMMLIB2D"
    FMMLIB2D does *no* checking for if the targets and sources coincide, and
    will return `Inf` values if `iop.target !== iop.source`, but there is a
    point `x ∈ iop.target` such that `x ∈ iop.source`.
"""
function assemble_fmm(iop::IntegralOperator; rtol)
    N = ambient_dimension(iop.source)
    if N == 2
        return _assemble_fmm2d(iop; rtol)
    elseif N == 3
        return _assemble_fmm3d(iop; rtol)
    else
        return error("Only 2D and 3D FMMs are supported")
    end
end

function _assemble_fmm2d(args...; kwargs...)
    return error("_assemble_fmm2d not found. Did you forget to import FMM2D or FMMLIB2D ?")
end

function _assemble_fmm3d(args...; kwargs...)
    return error("_assemble_fmm3d not found. Did you forget to import FMM3D ?")
end

"""
    assemble_hmatrix(iop[; atol, rank, rtol, eta])

Assemble an H-matrix representation of the discretized integral operator `iop`
using the `HMatrices.jl` library.

See the documentation of
[`HMatrices`](https://github.com/IntegralEquations/HMatrices.jl) for more
details on usage and other keyword arguments.
"""
function assemble_hmatrix(args...; kwargs...)
    return error("Inti.assemble_hmatrix not found. Did you forget to import HMatrices?")
end

"""
    _green_multiplier(x, quad)

Helper function to help determine the constant σ in the Green identity S\\[γ₁u\\](x)
- D\\[γ₀u\\](x) + σ*u(x) = 0. This can be used as a predicate to determine whether a
point is inside a domain or not.
"""
function _green_multiplier(x::SVector, Q::Quadrature{N}) where {N}
    op = Laplace(; dim = N)
    K = DoubleLayerKernel(op)
    σ = sum(Q.qnodes) do q
        return K(x, q) * weight(q)
    end
    return σ[1]
end
_green_multiplier(x::Tuple, Q::Quadrature) = _green_multiplier(SVector(x), Q)
_green_multiplier(x::QuadratureNode, Q::Quadrature) = _green_multiplier(coords(x), Q)

"""
    _green_multiplier(s::Symbol)

Return `-1.0` if `s == :inside`, `0.0` if `s == :outside`, and `-0.5` if `s ==
:on`; otherwise, throw an error. The orientation is relative to the normal of
the bounding curve/surface.
"""
function _green_multiplier(s::Symbol)
    # assume an exterior normal orientation and a smooth surface
    if s == :inside
        return -1.0
    elseif s == :outside
        return 0.0
    elseif s == :on
        return -0.5
    else
        return error("Unknown target location $s. Expected :inside, :outside, or :on")
    end
end

# Applying Laplace's double-layer to a constant will yield either 1 or -1,
# depending on whether the target point is inside or outside the obstacle.
# Assumes `quad` is the quadrature of a closed curve/surface
function isinside(x::SVector, quad::Quadrature, s = 1)
    u = _green_multiplier(x, quad)
    return s * u + 0.5 < 0
    # u < 0
end
isinside(x::Tuple, quad::Quadrature) = isinside(SVector(x), quad)
