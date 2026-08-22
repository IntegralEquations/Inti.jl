"""
    abstract type ReferenceInterpolant{D,T}

Interpolating function mapping points on the domain `D<:ReferenceShape`
(of singleton type) to a value of type `T`.

Instances `el` of `ReferenceInterpolant` are expected to implement:
- `el(x̂)`: evaluate the interpolation scheme at the (reference) coordinate `x̂
  ∈ D`.
- `jacobian(el,x̂)` : evaluate the jacobian matrix of the interpolation at the
  (reference) coordinate `x ∈ D`.

!!! note
    For performance reasons, both `el(x̂)` and `jacobian(el,x̂)` should
    take as input a `StaticVector` and output a static vector or static array.
"""
abstract type ReferenceInterpolant{D, T} end

function (el::ReferenceInterpolant)(x)
    return interface_method(el)
end

geometric_dimension(::ReferenceInterpolant{D, T}) where {D, T} = geometric_dimension(D)
ambient_dimension(el::ReferenceInterpolant{D, T}) where {D, T} = length(T)

"""
    jacobian(f,x)

Given a (possibly vector-valued) functor `f : 𝐑ᵐ → 𝐅ⁿ`, return the `n × m`
matrix `Aᵢⱼ = ∂fᵢ/∂xⱼ`. By default `ForwardDiff` is used to compute the
jacobian, but you should overload this method for specific `f` if better
performance and/or precision is required.

Note: both `x` and `f(x)` are expected to be of `SVector` type.
"""
function jacobian(f, s)
    return ForwardDiff.jacobian(f, s)
end
jacobian(f, s::Real) = jacobian(f, SVector(s))

"""
    hessian(el,x)

Given a (possibly vector-valued) functor `f : 𝐑ᵐ → 𝐅ⁿ`, return the `n × m × m`
matrix `Aᵢⱼⱼ = ∂²fᵢ/∂xⱼ∂xⱼ`. By default `ForwardDiff` is used to compute the
hessian, but you should overload this method for specific `f` if better
performance and/or precision is required.

Note: both `x` and `f(x)` are expected to be of `SVector` type.
"""
function hessian(el::ReferenceInterpolant, s)
    N = ambient_dimension(el)
    M = geometric_dimension(el)
    S = Tuple{N, M, M}
    return SArray{S}(stack(i -> ForwardDiff.hessian(x -> el(x)[i], s), 1:N; dims = 1))
end

function first_fundamental_form(el::ReferenceInterpolant, x̂)
    jac = jacobian(el, x̂)
    E = dot(jac[:, 1], jac[:, 1])
    geometric_dimension(el) == 1 && return E
    F = dot(jac[:, 1], jac[:, 2])
    G = dot(jac[:, 2], jac[:, 2])
    return E, F, G
end

function second_fundamental_form(el::ReferenceInterpolant, x̂)
    jac = jacobian(el, x̂)
    ν = _normal(jac)
    hess = hessian(el, x̂)
    L = dot(hess[:, 1, 1], ν)
    geometric_dimension(el) == 1 && return L
    M = dot(hess[:, 1, 2], ν)
    N = dot(hess[:, 2, 2], ν)
    return L, M, N
end

"""
    principal_curvatures(τ, x̂; surface_type::Symbol=:extrusion)

Calculate the two principal curvatures (κ₁, κ₂) of the surface defined by element `τ` at the
parametric coordinate `x̂`.  

If `τ` is a curve in two-dimensions, it is treated as the generator of a surface of
revolution (if `surface_type=:revolution`) or of an extrusion (if
`surface_type=:extrusion`).
"""
function principal_curvatures(el::ReferenceInterpolant, x̂; surface_type::Symbol = :extrusion)
    jac = jacobian(el, x̂)
    hess = hessian(el, x̂)
    ν = _normal(jac)

    if geometric_dimension(el) == 1
        E = dot(jac[:, 1], jac[:, 1])
        L = dot(hess[:, 1, 1], ν)
        κ_meridian = L / E

        if surface_type === :revolution
            r = el(x̂)[1]
            if abs(r) < 1.0e-12
                κ_azimuthal = κ_meridian
            else
                κ_azimuthal = -ν[1] / r
            end
            return (κ_meridian, κ_azimuthal)
        else
            return (κ_meridian, zero(κ_meridian))
        end
    else
        E = dot(jac[:, 1], jac[:, 1])
        F = dot(jac[:, 1], jac[:, 2])
        G = dot(jac[:, 2], jac[:, 2])
        L = dot(hess[:, 1, 1], ν)
        M = dot(hess[:, 1, 2], ν)
        N = dot(hess[:, 2, 2], ν)

        denom = E * G - F^2
        H = (L * G - 2 * F * M + E * N) / (2 * denom)
        K = (L * N - M^2) / denom

        disc = max(zero(H), H^2 - K)
        sqrt_disc = sqrt(disc)
        return (H + sqrt_disc, H - sqrt_disc)
    end
end

"""
    curvature(τ, x̂; kwargs...)

Calculate the sum of the principal curvatures (κ₁ + κ₂) of the element `τ` 
at the parametric coordinate `x̂`. 

This quantity is commonly used in physics to compute the Laplace pressure across an interface.
"""
curvature(el::ReferenceInterpolant, x̂; kwargs...) = sum(principal_curvatures(el, x̂; kwargs...))

"""
    mean_curvature(τ, x̂; kwargs...)

Calculate the [mean curvature](https://en.wikipedia.org/wiki/Mean_curvature) of
the element `τ` at the parametric coordinate `x̂`.
"""
mean_curvature(el::ReferenceInterpolant, x̂; kwargs...) = curvature(el, x̂; kwargs...) / 2

"""
    gauss_curvature(τ, x̂; kwargs...)

Calculate the [Gaussian
curvature](https://en.wikipedia.org/wiki/Gaussian_curvature) of the element `τ`
at the parametric coordinate `x̂`.
"""
gauss_curvature(el::ReferenceInterpolant, x̂; kwargs...) = prod(principal_curvatures(el, x̂; kwargs...))

domain(::ReferenceInterpolant{D, T}) where {D, T} = D()
domain(::Type{<:ReferenceInterpolant{D, T}}) where {D, T} = D()

# TODO: deprecate `domain` in favor of `reference_domain` for clarity
reference_domain(el) = domain(el)

return_type(::ReferenceInterpolant{D, T}) where {D, T} = T
return_type(::Type{<:ReferenceInterpolant{D, T}}) where {D, T} = T
domain_dimension(t::ReferenceInterpolant{D, T}) where {D, T} = domain(t) |> center |> length
function domain_dimension(t::Type{<:ReferenceInterpolant{D, T}}) where {D, T}
    return domain(t) |> center |> length
end
function range_dimension(el::ReferenceInterpolant{R, T}) where {R, T}
    return domain(el) |> center |> el |> length
end
function range_dimension(el::Type{<:ReferenceInterpolant{R, T}}) where {R, T}
    return domain(el) |> center |> el |> length
end

center(el::ReferenceInterpolant{D}) where {D} = el(center(D()))

# FIXME: need a practical definition of an approximate "radius" of an element.
# Does not need to be very sharp, since we mostly need to put elements inside a
# bounding ball. The method below is more a of a hack, but it is valid for
# convex polygons.
function radius(el::ReferenceInterpolant{D}) where {D}
    xc = center(el)
    return maximum(x -> norm(x - xc), vertices(el))
end

vertices(el::ReferenceInterpolant{D}) where {D} = el.(vertices(D()))

"""
    struct HyperRectangle{N,T} <: ReferenceInterpolant{ReferenceHyperCube{N},T}

Axis-aligned hyperrectangle in `N` dimensions given by
`low_corner::SVector{N,T}` and `high_corner::SVector{N,T}`.
"""
struct HyperRectangle{N, T} <: ReferenceInterpolant{ReferenceHyperCube{N}, T}
    low_corner::SVector{N, T}
    high_corner::SVector{N, T}
    # check that low_corner <= high_corner
    function HyperRectangle(low_corner::SVector{N, T}, high_corner::SVector{N, T}) where {N, T}
        @assert all(low_corner .<= high_corner) "low_corner must be less than high_corner"
        return new{N, T}(low_corner, high_corner)
    end
end

low_corner(el::HyperRectangle) = el.low_corner
high_corner(el::HyperRectangle) = el.high_corner
geometric_dimension(::HyperRectangle{N, T}) where {N, T} = N
ambient_dimension(::HyperRectangle{N, T}) where {N, T} = N

function (el::HyperRectangle)(u)
    lc = low_corner(el)
    hc = high_corner(el)
    v = @. lc + (hc - lc) * u
    return v
end

"""
    ParametricElement{D,T,F} <: ReferenceInterpolant{D,T}

An element represented through a explicit function `f` mapping `D` into the
element. For performance reasons, `f` should take as input a `StaticVector` and
return a `StaticVector` or `StaticArray`.

See also: [`ReferenceInterpolant`](@ref), [`LagrangeElement`](@ref)
"""
struct ParametricElement{D <: ReferenceShape, T, F} <: ReferenceInterpolant{D, T}
    parametrization::F
    function ParametricElement{D, T}(f::F) where {F, D, T}
        return new{D, T, F}(f)
    end
end

parametrization(el::ParametricElement) = el.parametrization
domain(::ParametricElement{D, T, F}) where {D, T, F} = D()
return_type(::ParametricElement{D, T, F}) where {D, T, F} = T

ambient_dimension(p::ParametricElement) = length(return_type(p))

function (el::ParametricElement)(u)
    @assert u ∈ domain(el)
    f = parametrization(el)
    return f(u)
end

vertices_idxs(::Type{<:ParametricElement{ReferenceLine}}) = 1:2
vertices_idxs(::Type{<:ParametricElement{ReferenceTriangle}}) = 1:3
vertices_idxs(::Type{<:ParametricElement{ReferenceSquare}}) = 1:4
vertices_idxs(::Type{<:ParametricElement{ReferenceTetrahedron}}) = 1:4
vertices_idxs(::Type{<:ParametricElement{ReferenceCube}}) = 1:8
vertices_idxs(el::ParametricElement) = vertices_idxs(typeof(el))

"""
    ParametricElement(f, d::HyperRectangle)

Construct the element defined as the image of `f` over `d`.
"""
function ParametricElement(f, d::HyperRectangle{N, T}) where {N, T}
    V = return_type(f, SVector{N, T})
    D = ReferenceHyperCube{N}
    return ParametricElement{D, V}((x) -> f(d(x)))
end

"""
    struct LagrangeElement{D,Np,T} <: ReferenceInterpolant{D,T}

A polynomial `p : D → T` uniquely defined by its `Np` values on the `Np` reference nodes
of `D`.

The return type `T` should be a vector space (i.e. support addition and
multiplication by scalars). For istance, `T` could be a number or a vector, but
not a `Tuple`.
"""
struct LagrangeElement{D <: ReferenceShape, Np, T} <: ReferenceInterpolant{D, T}
    vals::SVector{Np, T}
end

vals(el::LagrangeElement) = el.vals

"""
    reference_nodes(el::LagrangeElement)
    reference_nodes(::Type{<:LagrangeElement})

Return the reference nodes on `domain(el)` used for the polynomial
interpolation. The function values on these nodes completely determines the
interpolating polynomial.
"""
function reference_nodes(el::LagrangeElement)
    return interface_method(el)
end

# infer missig information from type of vals
function LagrangeElement{D}(vals::SVector{Np, T}) where {D, Np, T}
    return LagrangeElement{D, Np, T}(vals)
end

# a more convenient syntax
LagrangeElement{D}(x1, xs...) where {D} = LagrangeElement{D}(SVector(x1, xs...))

"""
    order(el::LagrangeElement)

The order of the element's interpolating polynomial (e.g. a `LagrangeLine` with
`2` nodes defines a linear polynomial, and thus has order `1`).
"""
@generated function order(::Type{<:LagrangeElement{D, Np}})::Int where {D, Np}
    if D <: ReferenceHyperCube
        N = geometric_dimension(D)
        K = findfirst(i -> i^N == Np, 1:100) - 1
        isnothing(K) && error("Np must be a perfect $N-th root")
    elseif D <: ReferenceSimplex
        N = geometric_dimension(D)
        K = findfirst(i -> binomial(i + N, N) == Np, 0:100) - 1
        isnothing(K) && error("Np must be an $N-triangular number")
    else
        notimplemented()
    end
    return :($K)
end

"""
    const LagrangeLine = LagrangeElement{ReferenceLine}
"""
const LagrangeLine = LagrangeElement{ReferenceLine}

const Line1D{T} = LagrangeElement{ReferenceLine, 2, SVector{1, T}}
const Line2D{T} = LagrangeElement{ReferenceLine, 2, SVector{2, T}}
const Line3D{T} = LagrangeElement{ReferenceLine, 2, SVector{3, T}}
Line1D(args...) = Line1D{Float64}(args...)
Line2D(args...) = Line2D{Float64}(args...)
Line3D(args...) = Line3D{Float64}(args...)

integration_measure(l::Line1D) = norm(vals(l)[2] - vals(l)[1])

"""
    const LagrangeTriangle = LagrangeElement{ReferenceTriangle}
"""
const LagrangeTriangle = LagrangeElement{ReferenceTriangle}

const Triangle2D{T} = LagrangeElement{ReferenceTriangle, 3, SVector{2, T}}
const Triangle3D{T} = LagrangeElement{ReferenceTriangle, 3, SVector{3, T}}
Triangle2D(args...) = Triangle2D{Float64}(args...)
Triangle3D(args...) = Triangle3D{Float64}(args...)

"""
    const LagrangeTetrahedron = LagrangeElement{ReferenceTetrahedron}
"""
const LagrangeTetrahedron = LagrangeElement{ReferenceTetrahedron}

"""
    const LagrangeSquare = LagrangeElement{ReferenceSquare}
"""
const LagrangeSquare = LagrangeElement{ReferenceSquare}

const Quadrangle2D{T} = LagrangeElement{ReferenceSquare, 4, SVector{2, T}}
const Quadrangle3D{T} = LagrangeElement{ReferenceSquare, 4, SVector{3, T}}
Quadrangle2D(args...) = Quadrangle2D{Float64}(args...)
Quadrangle3D(args...) = Quadrangle3D{Float64}(args...)

"""
    const LagrangeCube = LagrangeElement{ReferenceCube}
"""
const LagrangeCube = LagrangeElement{ReferenceCube}

"""
    vertices_idxs(el::LagrangeElement)
    vertices_idxs(::Type{LagrangeElement})

The indices of the nodes in `el` that define the vertices of the element.
"""
vertices_idxs(::Type{<:LagrangeLine{N}}) where {N} = SVector(1, N)

function vertices_idxs(::Type{<:LagrangeSquare{N2}}) where {N2}
    N = order(LagrangeSquare{N2}) + 1
    return SVector(1, N, N2, N2 - N + 1)
end

function vertices_idxs(::Type{<:LagrangeCube{N3}}) where {N3}
    N = order(LagrangeCube{N3}) + 1
    N2 = N * N
    low_face = SVector(1, N, N2 - N + 1, N2)
    up_face = (N3 - N2) .+ low_face
    return SVector(low_face..., up_face...)
end

function vertices_idxs(::Type{<:LagrangeTriangle{Np}}) where {Np}
    N = order(LagrangeTriangle{Np}) + 1
    return SVector(1, N, Np)
end

function vertices_idxs(::Type{<:LagrangeTetrahedron{Np}}) where {Np}
    N = order(LagrangeTetrahedron{Np}) + 1
    return SVector(1, N, N * (N + 1) ÷ 2, Np)
end

vertices_idxs(el::LagrangeElement) = vertices_idxs(typeof(el))

"""
    vertices(el::LagrangeElement)

Coordinates of the vertices of `el`.
"""
vertices(el::LagrangeElement) = view(vals(el), vertices_idxs(el))

"""
    boundary_vertex_idxs(::Type{<:ReferenceInterpolant})

For each face of the element, the indices *into its vertices* of that face's
vertices. This is the single definition of face ordering and winding: the
winding is chosen so that `_normal` gives the outward normal, and both
[`boundary_idxs`](@ref) (the topological view) and [`boundary_element`](@ref)
(the geometric view) are derived from it, so the two cannot drift apart.
"""
boundary_vertex_idxs(el::ReferenceInterpolant) = boundary_vertex_idxs(typeof(el))
boundary_vertex_idxs(::Type{<:ReferenceInterpolant{ReferenceLine}}) = ((1,), (2,))
boundary_vertex_idxs(::Type{<:ReferenceInterpolant{ReferenceTriangle}}) =
    ((1, 2), (2, 3), (3, 1))
boundary_vertex_idxs(::Type{<:ReferenceInterpolant{ReferenceSquare}}) =
    ((1, 2), (2, 3), (3, 4), (4, 1))
boundary_vertex_idxs(::Type{<:ReferenceInterpolant{ReferenceTetrahedron}}) =
    ((3, 2, 1), (1, 4, 3), (2, 3, 4), (1, 2, 4))

"""
    boundary_idxs(el)
    boundary_idxs(::Type{<:ReferenceInterpolant})

The indices of the nodes in `el` that define the boundary of the element,
returned as a tuple of faces, each face itself a tuple of node indices, wound as
in [`boundary_vertex_idxs`](@ref).

Indices are expressed through [`vertices_idxs`](@ref) rather than hard-coded, so
the methods hold for elements of any order (and for the curved
[`ParametricElement`](@ref)s, whose connectivity stores only the vertices).

This is a purely *topological* view — it identifies a face by its nodes, which is
what makes shared faces comparable. To obtain a face's geometry, use
[`boundary_element`](@ref): a face of a high-order or curved element is not the
straight span of its vertices.
"""
boundary_idxs(el::ReferenceInterpolant) = boundary_idxs(typeof(el))

function boundary_idxs(T::Type{<:ReferenceInterpolant})
    I = vertices_idxs(T)
    return map(f -> map(i -> I[i], f), boundary_vertex_idxs(T))
end

"""
    boundary_element(el, k)

The `k`-th face of `el` (ordered and wound as in [`boundary_idxs`](@ref)), as a
`ReferenceInterpolant` of one geometric dimension less.

The face is the element map restricted to the corresponding face of the
*reference* shape, so it follows the element's true geometry: high-order
`LagrangeElement`s and curved `ParametricElement`s alike.
"""
function boundary_element end

# the element map, without `ParametricElement`'s domain-membership assertion,
# which points sitting exactly on ∂(reference shape) can trip
_element_map(el::ParametricElement) = parametrization(el)
_element_map(el::ReferenceInterpolant) = el

function boundary_element(
        el::ReferenceInterpolant{D, T}, k::Integer,
    ) where {D <: Union{ReferenceTriangle, ReferenceSquare}, T}
    v = vertices(D())
    i, j = boundary_vertex_idxs(typeof(el))[k]
    ra, rb = v[i], v[j]
    f = _element_map(el)
    return ParametricElement{ReferenceHyperCube{1}, T}(u -> f(ra + u[1] * (rb - ra)))
end

function boundary_element(el::ReferenceInterpolant{ReferenceTetrahedron, T}, k::Integer) where {T}
    v = vertices(ReferenceTetrahedron())
    i, j, l = boundary_vertex_idxs(typeof(el))[k]
    ra, rb, rc = v[i], v[j], v[l]
    f = _element_map(el)
    return ParametricElement{ReferenceSimplex{2}, T}(
        u -> f(ra + u[1] * (rb - ra) + u[2] * (rc - ra)),
    )
end

# A simplex with only vertex nodes is affine, so its faces are exactly the
# straight spans of their vertices. Keep them as `LagrangeElement`s: the generic
# path above would wrap them in a closure and take its jacobian by AD.
function boundary_element(el::LagrangeElement{ReferenceTriangle, 3, T}, k::Integer) where {T}
    i, j = boundary_vertex_idxs(typeof(el))[k]
    V = vertices(el)
    return LagrangeElement{ReferenceHyperCube{1}, 2, T}(SVector(V[i], V[j]))
end

function boundary_element(el::LagrangeElement{ReferenceTetrahedron, 4, T}, k::Integer) where {T}
    i, j, l = boundary_vertex_idxs(typeof(el))[k]
    V = vertices(el)
    return LagrangeElement{ReferenceSimplex{2}, 3, T}(SVector(V[i], V[j], V[l]))
end

# generic ℚₖ elements for ReferenceHyperCube
function reference_nodes(T::Type{<:LagrangeElement{ReferenceHyperCube{D}, Np}}) where {D, Np}
    n = order(T) + 1
    @assert abs(n - Np^(1 / D)) < 1.0e-8 "Np must be a perfect power of D"
    nodes1d = ntuple(i -> n == 1 ? 0.5 : range(0, 1, n), D)
    nodes = map(Iterators.product(nodes1d...)) do x
        return SVector(x...)
    end
    return SVector{Np}(nodes)
end

@generated function (el::LagrangeElement{ReferenceHyperCube{D}, Np})(u) where {D, Np}
    n = order(el) + 1
    dims = ntuple(i -> n, D)
    # fetch references nodes on format expected by `lagrange_interp`
    nodes1d = n == 1 ? [0.5] : collect(range(0, 1, n))
    weights1d = barycentric_lagrange_weights(nodes1d)
    nodes = ntuple(i -> nodes1d, D)
    weights = ntuple(i -> weights1d, D)
    return quote
        v = reshape(vals(el), $dims)
        return tensor_lagrange_interp(SVector(u), v, $nodes, $weights, Val(D), 1, Np)
    end
end

"""
    tensor_lagrange_interp(
        x::SVector{N,Td},
        vals::AbstractArray{<:Any,N},
        nodes::NTuple{N},
        weights::NTuple{N},
        ::Val{dim},
        i1,
        len,
        ::Val{SKIP} = Val(false)
    ) where {N,Td,dim,SKIP}

Low-level function performing tensor-product Lagrange interpolation of an N-dimensional
function at the point `x`.

# Arguments
- `x::SVector{N,Td}`: The point at which to interpolate, given as a static vector of length `N`.
- `vals::AbstractArray{<:Any,N}`: The array of function values at the interpolation nodes, with `N` dimensions.
- `nodes::NTuple{N}`: A tuple containing the interpolation nodes for each dimension.
- `weights::NTuple{N}`: A tuple containing the barycentric weights for each dimension.
- `::Val{dim}`: A type-level value indicating the current dimension for recursion.
- `i1`: The starting index for the current slice of `vals`.
- `len`: The stride length for the current dimension.

# Returns
- The interpolated value at the point `x`, of the same type as the elements of `vals`.
"""
@inline function tensor_lagrange_interp(
        x::SVector{N, Td},
        vals::AbstractArray{<:Any, N},
        nodes::NTuple{N},
        weights::NTuple{N},
        ::Val{dim},
        i1,
        len,
        ::Val{SKIP} = Val(false),
    ) where {N, Td, dim, SKIP}
    T = eltype(vals)
    n = size(vals, dim)
    @inbounds xd = x[dim]
    @inbounds W = weights[dim]
    @inbounds X = nodes[dim]
    num = zero(T)
    l = one(Td)
    res = zero(T)
    δ = one(Td)
    # although the modified lagrange formula below is backward stable, autodiffing through
    # it is not if we are close to an interpolation node. The fix here is to switch to a
    # slightly different representation, which is more stable, if `x` is ever close to an
    # interpolation node. The `thres` variable below was chosen empirically for `Float64`
    # types.
    thres = 1.0e-3
    if dim == 1
        for i in 1:n
            @inbounds ci = vals[i1 + (i - 1)]
            @inbounds wi = W[i]
            @inbounds x_m_xi = xd - X[i]
            if SKIP || abs(x_m_xi) > thres || (!iszero(res))
                l *= x_m_xi
                num += (wi / x_m_xi) * ci
            else
                δ = x_m_xi
                res = ci * wi
            end
        end
        return res * l + l * δ * num
    else
        Δi = len ÷ n # column-major stride of current dimension
        # recurse down on dimension
        dim′ = Val{dim - 1}()
        @inbounds for i in 1:n
            ci =
                tensor_lagrange_interp(x, vals, nodes, weights, dim′, i1 + (i - 1) * Δi, Δi)
            wi = W[i]
            x_m_xi = xd - X[i]
            if SKIP || abs(x_m_xi) > thres || (!iszero(res))
                l *= x_m_xi
                num += (wi / x_m_xi) * ci
            else
                δ = x_m_xi
                res = ci * wi
            end
        end
        return res * l + l * δ * num
    end
end

function barycentric_lagrange_weights(x::AbstractVector)
    n = length(x)
    w = map(1:n) do i
        xᵢ = x[i]
        prod(Iterators.filter(j -> j ≠ i, 1:n); init = one(eltype(x))) do j
            xⱼ = x[j]
            return xᵢ - xⱼ
        end
    end
    w .= 1 ./ w
    return w
end

# generic ℙₖ elements for ReferenceSimplex
function reference_nodes(T::Type{<:LagrangeElement{ReferenceSimplex{D}, Np}}) where {D, Np}
    k = order(T)
    k == 0 && return SVector{1}((svector(i -> 1 / (D + 1), D),))
    nodes = SVector{D, Float64}[]
    for I in Iterators.product(ntuple(i -> 0:k, D)...)
        sum(I) > k && continue # skip if sum of indices exceeds n
        x = svector(i -> I[i] / k, D)
        push!(nodes, x)
    end
    return SVector{Np}(nodes)
end

# Based on a formula found in
# ``On a class of finite elements generated by Lagrange Interpolation''
# Nicolaides. SINUM 1972.
function (el::LagrangeElement{ReferenceSimplex{D}, Np})(u) where {D, Np}
    T = eltype(u)
    k = order(typeof(el))::Int
    iszero(k) && return vals(el)[1] # constant element
    u = SVector{D}(u)
    x = push(u, 1 - sum(u)) # add the last coordinate
    lags1d = MMatrix{k + 1, D + 1, T}(undef)
    @inbounds for dim in 1:(D + 1)
        xk = k * x[dim] # scaled coordinate
        lags1d[1, dim] = 1.0 # constant term
        for j in 1:k
            lags1d[j + 1, dim] = lags1d[j, dim] * (xk - (j - 1)) / j
        end
    end
    v = vals(el)
    acc = zero(eltype(v))
    @inbounds for (n, I) in enumerate(_barycentric_iterator(el))
        l = v[n] # value at the current node
        for d in 1:(D + 1)
            i = I[d] + 1
            l *= lags1d[i, d]
        end
        acc += l
    end
    return acc
end

@generated function _barycentric_iterator(
        el::LagrangeElement{ReferenceSimplex{D}, Np},
    ) where {D, Np}
    k = order(el)
    idxs = MVector{Np, NTuple{D + 1, Int}}(undef)
    cc = 0
    for I in Iterators.product(ntuple(i -> 0:k, D)...)
        sum(I) > k && continue # skip if sum of indices exceeds n
        cc += 1
        idxs[cc] = (I..., k - sum(I)) # add the last coordinate
    end
    return :($idxs)
end

"""
    lagrange_basis(E::Type{<:LagrangeElement})

Return the Lagrange basis `B` for the element `E`. Evaluating `B(x)` yields the
value of each basis function at `x`.
"""
function lagrange_basis(::Type{LagrangeElement{D, N, T}}) where {D, N, T}
    vals = svector(i -> svector(j -> i == j, N), N)
    return LagrangeElement{D}(vals)
end

"""
    translation_and_scaling(el) -> (c, r)

Center and radius of a ball containing `el`, the frame the scaled coordinate `x̃ = (x - c)/r`
of [`lvdim_correction`](@ref) is normalized against: the circumscribed ball of the element's
straight-sided vertices where its circumcenter lies inside it, and the ball on the longest
edge otherwise. Sharper than [`radius`](@ref) on a simplex, which is what wants it.
"""
function translation_and_scaling end

# fallback for elements with no sharper formula (e.g. the curved quadrilaterals `meshgen`
# produces): the element's own center and radius
translation_and_scaling(el::ReferenceInterpolant) = (center(el), radius(el))

function translation_and_scaling(el::LagrangeTriangle)
    vertices = el.vals[1:3]
    l1 = norm(vertices[1] - vertices[2])
    l2 = norm(vertices[2] - vertices[3])
    l3 = norm(vertices[3] - vertices[1])
    if ((l1^2 + l2^2 >= l3^2) && (l2^2 + l3^2 >= l1^2) && (l3^2 + l1^2 > l2^2))
        acuteright = true
    else
        acuteright = false
    end

    if acuteright
        # Compute the circumcenter and circumradius
        Bp = vertices[2] - vertices[1]
        Cp = vertices[3] - vertices[1]
        Dp = 2 * (Bp[1] * Cp[2] - Bp[2] * Cp[1])
        Upx = 1 / Dp * (Cp[2] * (Bp[1]^2 + Bp[2]^2) - Bp[2] * (Cp[1]^2 + Cp[2]^2))
        Upy = 1 / Dp * (Bp[1] * (Cp[1]^2 + Cp[2]^2) - Cp[1] * (Bp[1]^2 + Bp[2]^2))
        Up = SVector{2}(Upx, Upy)
        r = norm(Up)
        c = Up + vertices[1]
    else
        if (l1 >= l2) && (l1 >= l3)
            c = (vertices[1] + vertices[2]) / 2
            r = l1 / 2
        elseif (l2 >= l1) && (l2 >= l3)
            c = (vertices[2] + vertices[3]) / 2
            r = l2 / 2
        else
            c = (vertices[1] + vertices[3]) / 2
            r = l3 / 2
        end
    end
    return c, r
end

function translation_and_scaling(el::ParametricElement{ReferenceSimplex{3}})
    straight_nodes =
        [el([0.0, 0.0, 0.0]), el([1.0, 0.0, 0.0]), el([0.0, 1.0, 0.0]), el([0.0, 0.0, 1.0])]
    return translation_and_scaling(
        LagrangeElement{ReferenceSimplex{3}, 4, SVector{3, Float64}}(straight_nodes),
    )
end

function translation_and_scaling(el::ParametricElement{ReferenceSimplex{2}})
    straight_nodes = [el([1.0e-18, 1.0e-18]), el([1.0, 0.0]), el([0.0, 1.0])]
    return translation_and_scaling(
        LagrangeElement{ReferenceSimplex{2}, 3, SVector{2, Float64}}(straight_nodes),
    )
end

function translation_and_scaling(el::LagrangeTetrahedron)
    vertices = el.vals[1:4]
    # Compute the circumcenter in barycentric coordinates
    # formulas here are due to: https://math.stackexchange.com/questions/2863613/tetrahedron-centers
    a = norm(vertices[4] - vertices[1])
    b = norm(vertices[2] - vertices[4])
    c = norm(vertices[3] - vertices[4])
    d = norm(vertices[3] - vertices[2])
    e = norm(vertices[3] - vertices[1])
    f = norm(vertices[2] - vertices[1])
    f² = f^2
    a² = a^2
    b² = b^2
    c² = c^2
    d² = d^2
    e² = e^2

    ρ =
        a² * d² * (-d² + e² + f²) + b² * e² * (d² - e² + f²) + c² * f² * (d² + e² - f²) -
        2 * d² * e² * f²
    α =
        a² * d² * (b² + c² - d²) + e² * b² * (-b² + c² + d²) + f² * c² * (b² - c² + d²) -
        2 * b² * c² * d²
    β =
        b² * e² * (a² + c² - e²) + d² * a² * (-a² + c² + e²) + f² * c² * (a² - c² + e²) -
        2 * a² * c² * e²
    γ =
        c² * f² * (a² + b² - f²) + d² * a² * (-a² + b² + f²) + e² * b² * (a² - b² + f²) -
        2 * a² * b² * f²
    if (ρ >= 0 && α >= 0 && β >= 0 && γ >= 0)
        # circumcenter lays inside `el`
        center =
            (α * vertices[1] + β * vertices[2] + γ * vertices[3] + ρ * vertices[4]) /
            (ρ + α + β + γ)
        # ref: https://math.stackexchange.com/questions/1087011/calculating-the-radius-of-the-circumscribed-sphere-of-an-arbitrary-tetrahedron
        R = sqrt(1 / 2 * (β * f² + γ * e² + ρ * a²) / (ρ + α + β + γ))
    else
        if (a >= b && a >= c && a >= d && a >= e && a >= f)
            center = (vertices[1] + vertices[4]) / 2
            R = a / 2
        elseif (b >= a && b >= c && b >= d && b >= e && b >= f)
            center = (vertices[2] + vertices[4]) / 2
            R = b / 2
        elseif (c >= a && c >= b && c >= d && c >= e && c >= f)
            center = (vertices[3] + vertices[4]) / 2
            R = c / 2
        elseif (d >= a && d >= b && d >= c && d >= e && d >= f)
            center = (vertices[3] + vertices[2]) / 2
            R = d / 2
        elseif (e >= a && e >= b && e >= c && e >= d && e >= f)
            center = (vertices[3] + vertices[1]) / 2
            R = e / 2
        else
            center = (vertices[2] + vertices[1]) / 2
            R = f / 2
        end
    end
    return center, R
end
