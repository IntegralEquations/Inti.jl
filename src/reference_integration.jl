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

function (q::ReferenceQuadrature)(f)
    x, w = qcoords(q), qweights(q)
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

A quadrature of order `p` (sometimes called degree of precision) integrates all
polynomials of degree `≤ p` but not `≤ p + 1`.
"""
order(::Fejer{N}) where {N} = N - 1

@generated function (q::Fejer{N})() where {N}
    theta = [(2j - 1) * π / (2 * N) for j in 1:N]
    x = -cos.(theta)
    w = zero(x)
    for j in 1:N
        tmp = 0.0
        for l in 1:floor(N/2)
            tmp += 1 / (4 * l^2 - 1) * cos(2 * l * theta[j])
        end
        w[j] = 2 / N * (1 - 2 * tmp)
    end
    xs = svector(i -> SVector(0.5 * (x[i] + 1)), N)
    ws = svector(i -> w[i] / 2, N)
    return xs, ws
end

"""
    struct GaussLegendre{N,T}

`N`-point Gauss-Legendre quadrature rule for integrating a function over
`[0,1]`. Exactly integrates all polynomials of degree `≤ 2N-1`.

```jldoctest
using Inti

q = Inti.GaussLegendre(;order=10)

Inti.integrate(cos,q) ≈ sin(1) - sin(0)

# output

true
```
"""
struct GaussLegendre{N,T} <: ReferenceQuadrature{ReferenceLine}
    nodes::SVector{N,SVector{1,T}}
    weights::SVector{N,T}
end

function GaussLegendre{N,T}() where {N,T}
    x, w = gauss(T, N, 0, 1)
    V = SVector{1,T}
    return GaussLegendre(SVector{N,V}(V.(x)), SVector{N,T}(w))
end

GaussLegendre(n::Int) = GaussLegendre{n,Float64}()

GaussLegendre(; order::Int) = GaussLegendre(ceil(Int, (order + 1) / 2))

order(::GaussLegendre{N}) where {N} = 2 * N - 1

function (q::GaussLegendre)()
    return q.nodes, q.weights
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
        domain == :segment && (domain = ReferenceLine())
        domain == :triangle && (domain = ReferenceTriangle())
        domain == :tetrehedron && (domain = ReferenceTetrahedron())
        msg = "quadrature of order $order not available for $domain"
        if domain isa ReferenceLine
            # TODO: support Gauss-Legendre quadratures of arbitrary order
            order == 13 || error(msg)
            n = 7
        elseif domain isa ReferenceTriangle
            haskey(TRIANGLE_GAUSS_ORDER_TO_NPTS, order) || error(msg)
            n = TRIANGLE_GAUSS_ORDER_TO_NPTS[order]
        elseif domain isa ReferenceTetrahedron
            haskey(TETRAHEDRON_GAUSS_ORDER_TO_NPTS, order) || error(msg)
            n = TETRAHEDRON_GAUSS_ORDER_TO_NPTS[order]
        else
            error(
                "Tabulated Gauss quadratures only available for `ReferenceTriangle` or `ReferenceTetrahedron`",
            )
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
    # TODO: it makes no sense to store the tabulated rules in a format
    # different from what is needed when they are fetched.
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
qx = Inti.Fejer(10)
qy = Inti.Fejer(15)
q  = Inti.TensorProductQuadrature(qx,qy)
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
    nodes_iter = [SVector(x) for x in Iterators.product(nodes1d...)]
    weights_iter = [prod(w) for w in Iterators.product(weights1d...)]
    return nodes_iter, weights_iter
end

"""
    struct VioreanuRokhlin{D,N} <: ReferenceQuadrature{D}

Tabulated `N`-point Vioreanu-Rokhlin quadrature rule for integration over `D`.
"""
struct VioreanuRokhlin{D,N} <: ReferenceQuadrature{D}
    # VR quadrature should be constructed using the order, and not the number
    # of nodes. This ensures you don't instantiate quadratures which are not
    # tabulated.
    function VioreanuRokhlin(; domain, order)
        domain == :triangle && (domain = ReferenceTriangle())
        domain == :tetrahedron && (domain = ReferenceTetrahedron())
        if domain isa ReferenceTriangle
            msg = "VioreanuRokhlin quadrature of order $order not available for ReferenceTriangle"
            haskey(TRIANGLE_VR_ORDER_TO_NPTS, order) || error(msg)
            n = TRIANGLE_VR_ORDER_TO_NPTS[order]
        elseif domain isa ReferenceTetrahedron
            msg = "VioreanuRokhlin quadrature of order $order not available for ReferenceTetrahedron"
            haskey(TETRAHEDRON_VR_ORDER_TO_NPTS, order) || error(msg)
            n = TETRAHEDRON_VR_ORDER_TO_NPTS[order]
        else
            error(
                "Tabulated Vioreanu-Rokhlin quadratures only available for `ReferenceTriangle` or `ReferenceTetrahedron`",
            )
        end
        return new{typeof(domain),n}()
    end
end

function order(q::VioreanuRokhlin{ReferenceTriangle,N}) where {N}
    return TRIANGLE_VR_NPTS_TO_ORDER[N]
end

function order(q::VioreanuRokhlin{ReferenceTetrahedron,N}) where {N}
    return TETRAHEDRON_VR_NPTS_TO_ORDER[N]
end

function interpolation_order(q::VioreanuRokhlin{ReferenceTriangle,N}) where {N}
    return TRIANGLE_VR_QORDER_TO_IORDER[order(q)]
end

function interpolation_order(q::VioreanuRokhlin{ReferenceTetrahedron,N}) where {N}
    return TETRAHEDRON_VR_QORDER_TO_IORDER[order(q)]
end

function Triangle_VR_interpolation_order_to_quadrature_order(i::Integer)
    return TRIANGLE_VR_IORDER_TO_QORDER[i]
end

function Tetrahedron_VR_interpolation_order_to_quadrature_order(i::Integer)
    return TETRAHEDRON_VR_IORDER_TO_QORDER[i]
end

@generated function (q::VioreanuRokhlin{D,N})() where {D,N}
    x, w = _get_vioreanurokhlin_qcoords_and_qweights(D, N)
    return :($x, $w)
end

"""
    _get_vioreanurokhlin_qcoords_and_qweights(R::Type{<:ReferenceShape{D}}, N) where D

Returns the `N`-point Vioreanu-Rokhlin qnodes and qweights `(x, w)` for integration over `R`.
"""
function _get_vioreanurokhlin_qcoords_and_qweights(R::Type{<:ReferenceShape}, N)
    D = ambient_dimension(R())
    if !haskey(VR_QRULES, R) || !haskey(VR_QRULES[R], N)
        error("quadrature rule not found")
    end
    # TODO: it makes no sense to store the tabulated rules in a format
    # different from what is needed when they are fetched.
    qrule = VR_QRULES[R][N]
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
    struct EmbeddedQuadrature{L,H,D} <: ReferenceQuadrature{D}

A quadrature rule for the reference shape `D` based on a high-order quadrature
of type `H` and a low-order quadrature of type `L`. The low-order quadrature
rule is *embedded* in the sense that its `n` nodes are exactly the first `n`
nodes of the high-order quadrature rule.
"""
struct EmbeddedQuadrature{L,H,D} <: ReferenceQuadrature{D}
    low::L
    high::H
    function EmbeddedQuadrature(lquad, hquad)
        d = domain(lquad)
        @assert domain(lquad) == domain(hquad) "quadrature domains must match"
        xlow = qcoords(lquad)
        xhigh = qcoords(hquad)
        @assert length(xhigh) > length(xlow) "high-order quadrature must have more nodes than low-order"
        @assert xlow == xhigh[1:length(xlow)] "low-order nodes must exactly match the first high-order nodes. Got $(xlow) and $(xhigh)"
        return new{typeof(lquad),typeof(hquad),typeof(d)}(lquad, hquad)
    end
end

(q::EmbeddedQuadrature)() = q.high()

"""
    integrate_with_error_estimate(f, quad::EmbeddedQuadrature, norm = LinearAlgebra.norm)

Return `I, E` where `I` is the estimated integral of `f` over `domain(quad)`
using the high-order quadrature and `E` is the error estimate obtained by taking
the `norm` of the difference between the high and low-order quadratures in
`quad`.
"""
function integrate_with_error_estimate(
    f,
    quad::EmbeddedQuadrature,
    norm = LinearAlgebra.norm,
)
    x, w_high = quad.high()
    w_low = qweights(quad.low)
    nhigh, nlow = length(w_high), length(w_low)
    # assuming that nodes in quad_high are ordered so that the overlapping nodes
    # come first, add them up
    x1     = first(x)
    v1     = f(x1)
    I_high = v1 * first(w_high)
    I_low  = v1 * first(w_low)
    for i in 2:nlow
        v = f(x[i])
        I_high += v * w_high[i]
        I_low += v * w_low[i]
    end
    # now compute the rest of the high order quadrature
    for i in (nlow+1):nhigh
        v = f(x[i])
        I_high += v * w_high[i]
    end
    return I_high, norm(I_high - I_low)
end

"""
    lagrange_basis(qrule::ReferenceQuadrature)

Return a function `L : ℝᴺ → ℝᵖ` where `N` is the dimension of the domain of
`qrule`, and `p` is the number of nodes in `qrule`. The function `L` is a
polynomial in [`polynomial_space(qrule)`](@ref), and `L(xⱼ)[i] = δᵢⱼ` (i.e. the `i`th component of `L` is the `i`th
Lagrange basis).
"""
function lagrange_basis(qrule::ReferenceQuadrature{D}) where {D}
    k = interpolation_order(qrule)
    sp = PolynomialSpace{D,k}()
    nodes = qcoords(qrule)
    return lagrange_basis(nodes, sp)
end

"""
    polynomial_space(qrule::ReferenceQuadrature)

Return a [`PolynomialSpace`](@ref) associated with the
[`interpolation_order`](@ref) of the quadrature nodes of `qrule`.
"""
function polynomial_space(qrule::ReferenceQuadrature{D}) where {D}
    k = interpolation_order(qrule)
    return PolynomialSpace{D,k}()
end

"""
    interpolation_order(qrule::ReferenceQuadrature)

The interpolation order of a quadrature rule is defined as the the smallest `k`
such that there exists a unique polynomial in `PolynomialSpace{D,k}` that
minimizes the error in approximating the function `f` at the quadrature nodes.

For an `N`-point Gauss quadrature rule on the segment, the
interpolation order is `N-1` since `N` points uniquely determine a polynomial of
degree `N-1`.

For a triangular reference domain, the interpolation order is more difficult to
define. An unisolvent three-node quadrature on the triangular, for example, has
an interpolation order `k=1` since the three nodes uniquely determine a linear
polynomial, but a four-node quadrature may also have an interpolation order
`k=1` since for `k=2` there are multiple polynomials that pass through the four
nodes.
"""
function interpolation_order(qrule::ReferenceQuadrature{ReferenceLine})
    N = length(qrule)
    return N - 1
end

function interpolation_order(qrule::ReferenceQuadrature{ReferenceTriangle})
    N = length(qrule)
    # the last triangular less than or equal to N
    return floor(Int, (sqrt(8N + 1) - 3) / 2)
end

function interpolation_order(qrule::ReferenceQuadrature{ReferenceTetrahedron})
    N = length(qrule)
    # the last tetrahedral number less than or equal to N. For example P1 has at most 4
    # nodes, P2 has at most 10 nodes, P3 has at most 20 nodes, etc...
    P = 0
    while (P + 1) * (P + 2) * (P + 3) / 6 < N
        P += 1
    end
    return P - 1
end

function interpolation_order(qrule::Inti.TensorProductQuadrature)
    k1d = map(Inti.interpolation_order, qrule.quads1d)
    @assert allequal(k1d) "interpolation order must be the same in all dimensions"
    return first(k1d)
end

"""
    adaptive_quadrature(ref_domain::ReferenceShape; kwargs...)

Return a function `quad` callable as `quad(f)` that integrates the function `f` over the
reference shape `ref_domain`. The keyword arguments are passed to
`HAdaptiveIntegration.integrate`.
"""
function adaptive_quadrature(ref_domain::ReferenceLine; kwargs...)
    seg = HAdaptiveIntegration.segment(0.0, 1.0)
    return (f) -> HAdaptiveIntegration.integrate(f, seg; kwargs...)[1]
end
function adaptive_quadrature(ref_domain::ReferenceTriangle; kwargs...)
    tri = HAdaptiveIntegration.triangle((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    return (f) -> HAdaptiveIntegration.integrate(f, tri; kwargs...)[1]
end
function adaptive_quadrature(ref_domain::ReferenceSquare; kwargs...)
    sq = HAdaptiveIntegration.rectangle((0.0, 0.0), (1.0, 1.0))
    return (f) -> HAdaptiveIntegration.integrate(f, sq; kwargs...)[1]
end
