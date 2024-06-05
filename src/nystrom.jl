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
struct IntegralPotential{K,Q<:Quadrature}
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
    struct IntegralOperator{T} <: AbstractMatrix{T}

A discrete linear integral operator given by
```math
I[u](x) = \\int_{\\Gamma\\_s} K(x,y)u(y) ds_y, x \\in \\Gamma_{t}
```
where ``\\Gamma_s`` and ``\\Gamma_t`` are the source and target domains, respectively.
"""
struct IntegralOperator{V,K,T,S<:Quadrature} <: AbstractMatrix{V}
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
    T = return_type(k, eltype(X), eltype(Y))
    msg = """IntegralOperator of nonbits being created: $T"""
    isbitstype(T) || (@warn msg)
    return IntegralOperator{T,typeof(k),typeof(X),typeof(Y)}(k, X, Y)
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

"""
    assemble_fmm(iop; atol)

Set up a 2D or 3D FMM for evaluating the discretized integral operator `iop`
associated with the `pde`. In 2D the `FMM2D` or `FMMLIB2D` library is used
(whichever was most recently loaded) while in 3D `FMM3D` is used.

!!! warning "FMMLIB2D"
    FMMLIB2D does *no* checking for if the targets and sources coincide, and
    will return `Inf` values if `iop.target !== iop.source`, but there is a
    point `x ∈ iop.target` such that `x ∈ iop.source`.
"""
function assemble_fmm(iop::IntegralOperator, args...; kwargs...)
    N = ambient_dimension(iop.source)
    if N == 2
        return _assemble_fmm2d(iop, args...; kwargs...)
    elseif N == 3
        return _assemble_fmm3d(iop, args...; kwargs...)
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

See the `assemble_hmatrix` function from `HMatrices.jl` for more details on the
keyword arguments.
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
    pde = Laplace(; dim = N)
    K = DoubleLayerKernel(pde)
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

"""
    farfield_distance(iop::IntegralOperator; tol, maxiter = 10)
    farfield_distance(K, Q::Quadrature; tol, maxiter = 10)

Return an estimate of the distance `d` such that the (absolute) quadrature error
of the integrand `y -> K(x,y)` is below `tol` for `x` at a distance `d` from the
center of the largest element in `Q`; when an integral operator is passed, we
have `Q::Quadrature = source(iop)` and `K = kernel(iop)`.

The estimate is computed by finding the first integer `n` such that the
quadrature error on the largest element `τ` lies below `tol` for points `x`
satisfying `dist(x,center(τ)) = n*radius(τ)`.

Note that the desired tolerance may not be achievable if the quadrature rule is
not accurate enough, or if `τ` is not sufficiently small, and therefore a
maximum number of iterations `maxiter` is provided to avoid an infinite loops.
In such cases, it is recommended that you either increase the quadrature order,
or decrease the mesh size.

**Note**: this is obviously a heuristic, and may not be accurate in all cases.
"""
function farfield_distance(iop::IntegralOperator; tol, maxiter = 10)
    return farfield_distance(kernel(iop), source(iop); tol, maxiter)
end

function farfield_distance(K, Q::Quadrature; tol, maxiter = 10)
    msh = mesh(Q)
    dict = Dict{DataType,Float64}()
    for E in element_types(msh)
        els = elements(msh, E)
        qrule = quadrature_rule(Q, E)
        # pick the biggest element as a reference
        qtags = etype2qtags(Q, E)
        a, i = @views findmax(j -> sum(weight, Q[qtags[:, j]]), 1:size(qtags, 2))
        dict[E] = _farfield_distance(els[i], K, qrule, tol, maxiter)
    end
    # TODO: it may be useful to return a maxdist per element type so that we can
    # be more efficient in cases where different elements of different orders
    # and sizes are used. That is why a dictionary is created here. For the time
    # being, however, we just return the maximum distance.
    return maximum(values(dict))
end

function _farfield_distance(el, K, qrule, tol, maxiter)
    x₀ = center(el) # center
    h = radius(el)  # reasonable scale
    f = (x, ŷ) -> begin
        y   = el(ŷ)
        jac = jacobian(el, ŷ)
        ν   = _normal(jac)
        νₓ  = (x - x₀) |> normalize
        τ′  = _integration_measure(jac)
        return K((coords = x, normal = νₓ), (coords = y, normal = ν)) * τ′
    end
    τ̂ = domain(el)
    N = length(x₀)
    er = 0.0
    n = 0
    while n < maxiter
        n += 1
        # explore a few directions and pick a maximum distance
        er = 0.0
        for dir in -N:N
            iszero(dir) && continue
            k    = abs(dir)
            x    = setindex(x₀, x₀[k] + sign(N) * n * h, k)
            I, E = adaptive_integration(ŷ -> f(x, ŷ), τ̂; atol = tol / 2)
            @assert E < tol / 2 "hcubature did not converge"
            Ia = integrate(ŷ -> f(x, ŷ), qrule)
            er = max(er, norm(Ia - I))
        end
        @debug n, er
        (er < tol / 2) && break # attained desired tolerance
    end
    msg = """failed to attained desired tolerance when computing maxdist. Your
    quadrature may not be accurate enough, or your meshsize not small enough, to
    achieve the requested tolerance on the far field."""
    er > tol / 2 && @warn msg
    return n * h
end
