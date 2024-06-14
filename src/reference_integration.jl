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
    for i in nlow+1:nhigh
        v = f(x[i])
        I_high += v * w_high[i]
    end
    return I_high, norm(I_high - I_low)
end

"""
    adaptive_integration(f, τ̂::RefernceShape; kwargs...)
    adaptive_integration(f, qrule::EmbeddedQuadrature; kwargs...)

Use an adaptive procedure to estimate the integral of `f` over `τ̂ =
domain(qrule)`. The following optional keyword arguments are available:
- `atol::Real=0.0`: absolute tolerance for the integral estimate
- `rtol::Real=0.0`: relative tolerance for the integral estimate
- `maxsplit::Int=1000`: maximum number of times to split the domain
- `norm::Function=LinearAlgebra.norm`: norm to use for error estimates
- `buffer::BinaryHeap`: a pre-allocated buffer to use for the adaptive procedure
  (see [allocate_buffer](@ref))
"""
function adaptive_integration(
    f,
    quad::EmbeddedQuadrature;
    atol = 0.0,
    rtol = iszero(atol) ? sqrt(eps()) : 0.0,
    maxsplit = 1000,
    norm = LinearAlgebra.norm,
    buffer = nothing,
)
    return _adaptive_integration(f, quad, atol, rtol, maxsplit, norm, buffer)
end
function adaptive_integration(f, τ̂::ReferenceShape; kwargs...)
    return adaptive_integration(f, default_embedded_quadrature(τ̂); kwargs...)
end

function _adaptive_integration(f, quad::EmbeddedQuadrature, atol, rtol, maxsplit, norm, buf)
    τ̂ = domain(quad)
    nsplit = 0
    I, E = integrate_with_error_estimate(f, quad)
    # a quick check to see if splitting is really needed
    if E < atol || E < rtol * norm(I) || nsplit >= maxsplit
        return I, E
    end
    # split is needed, so prepare heap if needed, push the element to the heap
    # and begin
    heap = if isnothing(buf)
        allocate_buffer(f, quad)
    else
        empty!(buf.valtree)
        buf
    end
    # create a first (non-singleton) element, push it to the heap, and begin
    push!(heap, (LagrangeElement(τ̂), I, E))
    while E > atol && E > rtol * norm(I) && nsplit < maxsplit
        el, Ic, Ec = pop!(heap)
        I -= Ic
        E -= Ec
        for child in subdivide(el)
            μ = integration_measure(child)
            Inew, Enew = μ .* integrate_with_error_estimate(x -> f(child(x)), quad)
            I += Inew
            E += Enew
            push!(heap, (child, Inew, Enew))
        end
        nsplit += 1
    end
    # nsplit >= maxsplit && @warn "maximum number of steps reached"
    return I, E
end

"""
    allocate_buffer(f, quad::EmbeddedQuadrature)

Create the `buffer` needed for the call [`adaptive_integration(f, τ̂; buffer,
...)`](@ref adaptive_integration).
"""
function allocate_buffer(f, quad::EmbeddedQuadrature)
    T = Float64 # TODO: make this a parameter so that we can do single precision?
    τ̂ = domain(quad)
    # type of element that will be returned by by quad. Pay the cost of single
    # call to figure this out
    I, E = integrate_with_error_estimate(f, quad)
    # the heap of adaptive quadratures have elements of the form (s,I,E), where
    # I and E are the value and error estimate over the simplex s. The ordering
    # used is based the maximum error
    ord = Base.Order.By(el -> -el[3])
    # figure out the shape of the domains that will be needed for the heap
    S = if τ̂ isa ReferenceLine
        Line1D{T}
    elseif τ̂ isa ReferenceTriangle
        Triangle2D{T}
        error("not implemented")
    end
    heap = BinaryHeap{Tuple{S,typeof(I),typeof(E)}}(ord)
    return heap
end
allocate_buffer(f, τ̂::ReferenceShape) = allocate_buffer(f, default_embedded_quadrature(τ̂))

function subdivide(ln::Line1D)
    # @assert x ∈ ln
    a, b = vertices(ln)
    m = (a + b) / 2
    return LagrangeLine(a, m), LagrangeLine(m, b)
end

function default_embedded_quadrature(::ReferenceLine)
    qhigh = Kronrod{ReferenceLine,15}()
    qlow = Gauss(; domain = ReferenceLine(), order = 13)
    return EmbeddedQuadrature(qlow, qhigh)
end

# some tabulated rules used for EmbeddedQuadrature
"""
    struct Kronrod{D,N} <: ReferenceQuadrature{D}

`N`-point Kronrod rule obtained by adding `n+1` points to a Gauss quadrature
containing `n` points. The order is either `3n + 1` for `n` even or `3n + 2` for
`n` odd.
"""
struct Kronrod{D,N} <: ReferenceQuadrature{D} end

function (qrule::Gauss{ReferenceLine,7})()
    return SEGMENT_G13N7
end

function (qrule::Kronrod{ReferenceLine,15})()
    return SEGMENT_K23N15
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
