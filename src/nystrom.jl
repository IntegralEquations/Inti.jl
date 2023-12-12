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
    # the source, on the other hand, has to be a quadrature for our Nystrom method
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

"""
    assemble_matrix(iop::IntegralOperator; threads = true)

Assemble the dense matrix representation of an `IntegralOperator`.
"""
function assemble_matrix(iop::IntegralOperator; threads = true)
    T    = eltype(iop)
    m, n = size(iop)
    out  = Matrix{T}(undef, m, n)
    K    = kernel(iop)
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

# helper function to help determine the constant σ in the Green identity:
# S[γ₁u](x) - D[γ₀u](x) + σ*u(x) = 0
# This can be used as a predicate to determine whether a point is inside a
# domain or not
function _green_multiplier(x::SVector, Q::Quadrature{N}) where {N}
    pde = Laplace(; dim = N)
    K = DoubleLayerKernel(pde)
    σ = sum(Q.qnodes) do q
        return K(x, q) * weight(q)
    end
    return σ[1]
end
_green_multiplier(x::Tuple, Q::Quadrature) = _green_multiplier(SVector(x), Q)
_green_multiplier(x::QuadratureNode, Q::Quadrature) = _green_multiplier(coords(x), Q)

# Applying Laplace's double-layer to a constant will yield either 1 or -1,
# depending on whether the target point is inside or outside the obstacle.
# Assumes `quad` is the quadrature of a closed curve/surface
function isinside(x::SVector, quad::Quadrature, s = 1)
    u = _green_multiplier(x, quad)
    return s * u + 0.5 < 0
    # u < 0
end
isinside(x::Tuple, quad::Quadrature) = isinside(SVector(x), quad)
