"""
    abstract type ReferenceQuadrature{D}

A quadrature rule for integrating a function over the domain `D <: ReferenceShape`.

Calling `x,w = q()` returns the nodes `x`, given as `SVector`s, and weights `w`,
for performing integration over `domain(q)`.
"""
abstract type ReferenceQuadrature{D} end

"""
    domain(q::ReferenceQuadrature)

The domain of integratino for quadrature rule `q`.
"""
domain(q::ReferenceQuadrature{D}) where {D} = D()

"""
    qcoords(q)

Return the coordinate of the quadrature nodes associated with `q`.
"""
qcoords(q::ReferenceQuadrature) = q()[1]

"""
    qweights(q)

Return the quadrature weights associated with `q`.
"""
qweights(q::ReferenceQuadrature) = q()[2]

function (q::ReferenceQuadrature)()
    return interface_method(q)
end

Base.length(q::ReferenceQuadrature) = length(qweights(q))

"""
    integrate(f,q::ReferenceQuadrature)
    integrate(f,x,w)

Integrate the function `f` using the quadrature rule `q`. This is simply
`sum(f.(x) .* w)`, where `x` and `w` are the quadrature nodes and weights,
respectively.

The function `f` should take an `SVector` as input.
"""
function integrate(f, q::ReferenceQuadrature)
    x, w = q()
    if domain(q) == ReferenceLine()
        return integrate(x -> f(x[1]), x, w)
    else
        return integrate(f, x, w)
    end
end

function integrate(f, x, w)
    sum(zip(x, w)) do (x, w)
        return f(x) * prod(w)
    end
end

## Define some one-dimensional quadrature rules
"""
    struct Fejer{N}

`N`-point Fejer's first quadrature rule for integrating a function over `[0,1]`.
Exactly integrates all polynomials of degree `≤ N-1`.

```jldoctest
using Inti

q = Inti.Fejer(;order=10)

Inti.integrate(cos,q) ≈ sin(1) - sin(0)

# output

true
```

"""
struct Fejer{N} <: ReferenceQuadrature{ReferenceLine} end

Fejer(n::Int) = Fejer{n}()

Fejer(; order::Int) = Fejer(order + 1)

# N point fejer quadrature integrates all polynomials up to degree N-1
"""
    order(q::ReferenceQuadrature)

A quadrature of order `p` integrates all polynomials of degree `≤ p`.
"""
order(::Fejer{N}) where {N} = N - 1

@generated function (q::Fejer{N})() where {N}
    theta = [(2j - 1) * π / (2 * N) for j in 1:N]
    x = -cos.(theta)
    w = zero(x)
    for j in 1:N
        tmp = 0.0
        for l in 1:floor(N / 2)
            tmp += 1 / (4 * l^2 - 1) * cos(2 * l * theta[j])
        end
        w[j] = 2 / N * (1 - 2 * tmp)
    end
    xs = svector(i -> SVector(0.5 * (x[i] + 1)), N)
    ws = svector(i -> w[i] / 2, N)
    return xs, ws
end

"""
    struct Gauss{D,N} <: ReferenceQuadrature{D}

Tabulated `N`-point symmetric Gauss quadrature rule for integration over `D`.
"""
struct Gauss{D,N} <: ReferenceQuadrature{D}
    # gauss quadrature should be constructed using the order, and not the number
    # of nodes. This ensures you don't instantiate quadratures which are not
    # tabulated.
    function Gauss(; domain, order)
        domain == :triangle && (domain = ReferenceTriangle())
        domain == :tetrehedron && (domain = ReferenceTetrahedron())
        if domain isa ReferenceTriangle
            msg = "quadrature of order $order not available for ReferenceTriangle"
            haskey(TRIANGLE_GAUSS_ORDER_TO_NPTS, order) || error(msg)
            n = TRIANGLE_GAUSS_ORDER_TO_NPTS[order]
        elseif domain isa ReferenceTetrahedron
            msg = "quadrature of order $order not available for ReferenceTetrahedron"
            haskey(TETRAHEDRON_GAUSS_ORDER_TO_NPTS, order) || error(msg)
            n = TETRAHEDRON_GAUSS_ORDER_TO_NPTS[order]
        else
            error("Tabulated Gauss quadratures only available for `ReferenceTriangle` or `ReferenceTetrahedron`")
        end
        return new{typeof(domain),n}()
    end
end

function order(q::Gauss{ReferenceTriangle,N}) where {N}
    return TRIANGLE_GAUSS_NPTS_TO_ORDER[N]
end

function order(q::Gauss{ReferenceTetrahedron,N}) where {N}
    return TETRAHEDRON_GAUSS_NPTS_TO_ORDER[N]
end

@generated function (q::Gauss{D,N})() where {D,N}
    x, w = _get_gauss_qcoords_and_qweights(D, N)
    return :($x, $w)
end

"""
    _get_gauss_and_qweights(R::Type{<:ReferenceShape{D}}, N) where D

Returns the `N`-point symmetric gaussian qnodes and qweights `(x, w)` for integration over `R`.
"""
function _get_gauss_qcoords_and_qweights(R::Type{<:ReferenceShape}, N)
    D = ambient_dimension(R())
    if !haskey(GAUSS_QRULES, R) || !haskey(GAUSS_QRULES[R], N)
        error("quadrature rule not found")
    end
    qrule = GAUSS_QRULES[R][N]
    @assert length(qrule) == N
    # qnodes
    qnodestype = SVector{N,SVector{D,Float64}}
    x = qnodestype([q[1] for q in qrule])
    # qweights
    qweightstype = SVector{N,Float64}
    w = qweightstype([q[2] for q in qrule])
    return x, w
end

"""
    TensorProductQuadrature{N,Q}

A tensor-product of one-dimension quadrature rules. Integrates over `[0,1]^N`.

# Examples
```julia
qx = Fejer(10)
qy = TrapezoidalOpen(15)
q  = TensorProductQuadrature(qx,qy)
```
"""
struct TensorProductQuadrature{N,Q} <: ReferenceQuadrature{ReferenceHyperCube{N}}
    quads1d::Q
end

function TensorProductQuadrature(q...)
    N = length(q)
    Q = typeof(q)
    return TensorProductQuadrature{N,Q}(q)
end

function (q::TensorProductQuadrature{N})() where {N}
    nodes1d = ntuple(N) do i
        x1d, _ = q.quads1d[i]()
        return map(x -> x[1], x1d) # convert the `SVector{1,T}` to just `T`
    end
    weights1d = map(q -> q()[2], q.quads1d)
    nodes_iter = (SVector(x) for x in Iterators.product(nodes1d...))
    weights_iter = (prod(w) for w in Iterators.product(weights1d...))
    return nodes_iter, weights_iter
end
