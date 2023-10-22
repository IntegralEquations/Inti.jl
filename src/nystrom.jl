"""
    struct NystromDensity{N,T,V} <: AbstractVector{V}

Values of type `V` on a `Quadrature` in `N` dimensions.
"""
struct NystromDensity{N,T,V} <: AbstractVector{V}
    values::Vector{V}
    quadrature::Quadrature{N,T}
end

Base.size(σ::NystromDensity) = size(σ.values)
Base.getindex(σ::NystromDensity, args...) = getindex(σ.values, args...)
Base.setindex!(σ::NystromDensity, args...) = setindex!(σ.values, args...)

"""
    NystromDensity(f::Function, Q::Quadrature)

Return a `NystromDensity` with values `f(q)` at the quadrature nodes of `Q`.

Note that the argument passsed to `f` is a [`QuadratureNode`](@ref), so that `f`
may depend on quantities other than the [`coords`](@ref) of the quadrature node
(such as the [`normal`](@ref) vector).

See also: [`QuadratureNode`](@ref)
"""
function NystromDensity(f::Function, Q::Quadrature{N,T}) where {N,T}
    vals = [f(dof) for dof in Q]
    V = eltype(vals)
    return NystromDensity{N,T,V}(vals, Q)
end

function NystromDensity(pde::AbstractPDE,Q::Quadrature)
    T = default_density_eltype(pde)
    NystromDensity(i->zero(T),Q)
end

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

"""
struct IntegralPotential
    kernel::AbstractKernel
    quadrature::Quadrature
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
    single_layer_potential(pde::AbstractPDE, quad::Quadrature)

Build an [`IntegralPotential`](@ref) corresponding to the single-layer potential
over `quad` associated to a given PDE.
"""
function single_layer_potential(op::AbstractPDE, quad)
    return IntegralPotential(SingleLayerKernel(op), quad)
end

"""
    double_layer_potential(pde::AbstractPDE, quad::Quadrature)

Build an [`IntegralPotential`](@ref) corresponding to the double-layer potential
over `quad` associated to a given PDE.
"""
function double_layer_potential(op::AbstractPDE, quad)
    return IntegralPotential(DoubleLayerKernel(op), quad)
end

"""
    struct IntegralOperator{T} <: AbstractMatrix{T}

A discrete linear integral operator given by
```math
I[u](x) = \\int_{\\Gamma\\_s} K(x,y)u(y) ds_y, x \\in \\Gamma_{t}
```
where ``\\Gamma_s`` and ``\\Gamma_t`` are the source and target domains, respectively.
"""
struct IntegralOperator{T} <: AbstractMatrix{T}
    kernel::AbstractKernel
    # since the target can be as simple as a vector of points, leave it untyped
    target
    # the source, on the other hand, has to be a quadrature
    source::Quadrature
end

kernel(iop::IntegralOperator) = iop.kernel

function IntegralOperator(k, X, Y::Quadrature = X)
    T = return_type(k)
    msg = """IntegralOperator of nonbits being created"""
    isbitstype(T) || (@warn msg)
    return IntegralOperator{T}(k, X, Y)
end

Base.size(iop::IntegralOperator) = (length(iop.target), length(iop.source))

function Base.getindex(iop::IntegralOperator, i::Integer, j::Integer)
    k = kernel(iop)
    return k(iop.target[i], iop.source[j]) * weight(iop.source[j])
end

function Base.Matrix(iop::IntegralOperator{T}) where {T}
    m, n = size(iop)
    K = kernel(iop)
    out = Matrix{T}(undef, m, n)
    _iop_to_matrix!(out, K, iop.target, iop.source)
    return out
end
@noinline function _iop_to_matrix!(out, K, X, Y::Quadrature)
    Threads.@threads for j in 1:length(Y)
        for i in 1:length(X)
            out[i, j] = K(X[i], Y[j]) * weight(Y[j])
        end
    end
    return out
end

# convenience constructors
function single_layer_operator(op::AbstractPDE, X, Y = X)
    return IntegralOperator(SingleLayerKernel(op), X, Y)
end
function double_layer_operator(op::AbstractPDE, X, Y = X)
    return IntegralOperator(DoubleLayerKernel(op), X, Y)
end

function adjoint_double_layer_operator(op::AbstractPDE, X, Y = X)
    return IntegralOperator(AdjointDoubleLayerKernel(op), X, Y)
end
function hypersingular_operator(op::AbstractPDE, X, Y = X)
    return IntegralOperator(HyperSingularKernel(op), X, Y)
end

# Applying Laplace's double-layer to a constant will yield either 1 or -1,
# depending on whether the target point is inside or outside the obstacle.
# Assumes `quad` is the quadrature of a closed curve/surface
function isinside(x::SVector, quad::Quadrature, s = 1)
    N = ambient_dimension(quad)
    pde = Laplace(; dim = N)
    K = DoubleLayerKernel(pde)
    u = sum(quad.qnodes) do source
        return K(x, source) * weight(source)
    end
    return s * u + 0.5 < 0
    # u < 0
end
isinside(x::Tuple, quad::Quadrature) = isinside(SVector(x), quad)
