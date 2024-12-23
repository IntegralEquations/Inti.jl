"""
    abstract type ReferenceInterpolant{D,T}

Interpolanting function mapping points on the domain `D<:ReferenceShape`
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
abstract type ReferenceInterpolant{D,T} end

function (el::ReferenceInterpolant)(x)
    return interface_method(el)
end

geometric_dimension(::ReferenceInterpolant{D,T}) where {D,T} = geometric_dimension(D)
ambient_dimension(el::ReferenceInterpolant{D,T}) where {D,T} = length(T)

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
    hesssian(el,x)

Given a (possibly vector-valued) functor `f : 𝐑ᵐ → 𝐅ⁿ`, return the `n × m × m`
matrix `Aᵢⱼⱼ = ∂²fᵢ/∂xⱼ∂xⱼ`. By default `ForwardDiff` is used to compute the
hessian, but you should overload this method for specific `f` if better
performance and/or precision is required.

Note: both `x` and `f(x)` are expected to be of `SVector` type.
"""
function hessian(el::ReferenceInterpolant, s)
    N = ambient_dimension(el)
    M = geometric_dimension(el)
    S = Tuple{N,M,M}
    return SArray{S}(stack(i -> ForwardDiff.hessian(x -> el(x)[i], s), 1:N; dims = 1))
end

function first_fundamental_form(el::ReferenceInterpolant, x̂)
    jac = jacobian(el, x̂)
    # first fundamental form
    E = dot(jac[:, 1], jac[:, 1])
    F = dot(jac[:, 1], jac[:, 2])
    G = dot(jac[:, 2], jac[:, 2])
    return E, F, G
end

function second_fundamental_form(el::ReferenceInterpolant, x̂)
    jac = jacobian(el, x̂)
    ν = _normal(jac)
    # second fundamental form
    hess = hessian(el, x̂)
    L = dot(hess[:, 1, 1], ν)
    M = dot(hess[:, 1, 2], ν)
    N = dot(hess[:, 2, 2], ν)

    return L, M, N
end

"""
    mean_curvature(τ, x̂)

Calculate the [mean curvature](https://en.wikipedia.org/wiki/Mean_curvature) of
the element `τ` at the parametric coordinate `x̂`.
"""
function mean_curvature(el::ReferenceInterpolant, x̂)
    E, F, G = first_fundamental_form(el, x̂)
    L, M, N = second_fundamental_form(el, x̂)
    # mean curvature
    κ = (L * G - 2 * F * M + E * N) / (2 * (E * G - F^2))
    return κ
end

"""
    gauss_curvature(τ, x̂)

Calculate the [Gaussian
curvature](https://en.wikipedia.org/wiki/Gaussian_curvature) of the element `τ`
at the parametric coordinate `x̂`.
"""
function gauss_curvature(el::ReferenceInterpolant, x̂)
    E, F, G = first_fundamental_form(el, x̂)
    L, M, N = second_fundamental_form(el, x̂)
    # Guassian curvature
    κ = (L * N - M^2) / (E * G - F^2)
    return κ
end

"""
    curvature(Q::Quadrature{2})

Compute the curvature at each quadrature node in `Q`, where `Q` is the quadrature of curve
in 2D.
"""
curvature(Q::Quadrature{2}) = _curvature(curvature, Q)

# helper function for computing curvature
function _curvature(f, Q)
    msh = mesh(Q)
    curv = zeros(length(Q))
    for (E, tags) in Q.etype2qtags
        qrule = quadrature_rule(Q, E)
        q̂, _ = qrule()
        els = elements(msh, E)
        for n in 1:size(tags, 2)
            el = els[n]
            for i in 1:size(tags, 1)
                qtag = tags[i, n]
                curv[qtag] = f(el, q̂[i])
            end
        end
    end
    return curv
end

domain(::ReferenceInterpolant{D,T}) where {D,T} = D()
domain(::Type{<:ReferenceInterpolant{D,T}}) where {D,T} = D()
return_type(::ReferenceInterpolant{D,T}) where {D,T} = T
return_type(::Type{<:ReferenceInterpolant{D,T}}) where {D,T} = T
domain_dimension(t::ReferenceInterpolant{D,T}) where {D,T} = domain(t) |> center |> length
function domain_dimension(t::Type{<:ReferenceInterpolant{D,T}}) where {D,T}
    return domain(t) |> center |> length
end
function range_dimension(el::ReferenceInterpolant{R,T}) where {R,T}
    return domain(el) |> center |> el |> length
end
function range_dimension(el::Type{<:ReferenceInterpolant{R,T}}) where {R,T}
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
struct HyperRectangle{N,T} <: ReferenceInterpolant{ReferenceHyperCube{N},T}
    low_corner::SVector{N,T}
    high_corner::SVector{N,T}
    # check that low_corner <= high_corner
    function HyperRectangle(low_corner::SVector{N,T}, high_corner::SVector{N,T}) where {N,T}
        @assert all(low_corner .<= high_corner) "low_corner must be less than high_corner"
        return new{N,T}(low_corner, high_corner)
    end
end

low_corner(el::HyperRectangle) = el.low_corner
high_corner(el::HyperRectangle) = el.high_corner
geometric_dimension(::HyperRectangle{N,T}) where {N,T} = N
ambient_dimension(::HyperRectangle{N,T}) where {N,T} = N

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
struct ParametricElement{D<:ReferenceShape,T,F} <: ReferenceInterpolant{D,T}
    parametrization::F
    function ParametricElement{D,T}(f::F) where {F,D,T}
        return new{D,T,F}(f)
    end
end

parametrization(el::ParametricElement) = el.parametrization
domain(::ParametricElement{D,T,F}) where {D,T,F} = D()
return_type(::ParametricElement{D,T,F}) where {D,T,F} = T

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
function ParametricElement(f, d::HyperRectangle{N,T}) where {N,T}
    V = return_type(f, SVector{N,T})
    D = ReferenceHyperCube{N}
    return ParametricElement{D,V}((x) -> f(d(x)))
end

"""
    struct LagrangeElement{D,Np,T} <: ReferenceInterpolant{D,T}

A polynomial `p : D → T` uniquely defined by its `Np` values on the `Np` reference nodes
of `D`.

The return type `T` should be a vector space (i.e. support addition and
multiplication by scalars). For istance, `T` could be a number or a vector, but
not a `Tuple`.
"""
struct LagrangeElement{D<:ReferenceShape,Np,T} <: ReferenceInterpolant{D,T}
    vals::SVector{Np,T}
end

vals(el::LagrangeElement) = el.vals

"""
    reference_nodes(el::LagrangeElement)
    reference_nodes(::Type{<:LagrangeElement})

Return the reference nodes on `domain(el)` used for the polynomial
interpolation. The function values on these nodes completely determines the
interpolating polynomial.

We use the same convention as `gmsh` for defining the reference nodes and their
order (see [node
ordering](https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering) on `gmsh`
documentation).
"""
function reference_nodes(el::LagrangeElement)
    return interface_method(el)
end

# infer missig information from type of vals
function LagrangeElement{D}(vals::SVector{Np,T}) where {D,Np,T}
    return LagrangeElement{D,Np,T}(vals)
end

# a more convenient syntax
LagrangeElement{D}(x1, xs...) where {D} = LagrangeElement{D}(SVector(x1, xs...))

# construct based on a function
function LagrangeElement{D}(f::Function) where {D}
    ref_nodes = reference_nodes(D())
    vals = svector(i -> f(ref_nodes[i]), length(ref_nodes))
    return LagrangeElement{D}(vals)
end

"""
    order(el::LagrangeElement)

The order of the element's interpolating polynomial (e.g. a `LagrangeLine` with
`2` nodes defines a linear polynomial, and thus has order `1`).
"""
function order(::Type{<:LagrangeElement{D,Np}})::Int where {D,Np}
    if D == ReferenceLine
        return Np - 1
    elseif D == ReferenceTriangle
        K = (-3 + sqrt(1 + 8 * Np)) / 2
        isinteger(K) || error("Np must be triangular number")
        return Int(K)
    elseif D == ReferenceTetrahedron
        if Np == 4
            return 1
        elseif Np == 10
            return 2
        else
            # TODO: general case of tetrahedron
            notimplemented()
        end
    elseif D == ReferenceSquare
        K = sqrt(Np) - 1
        isinteger(K) || error("Np must be square number")
        return Int(K)
    elseif D == ReferenceCube
        K = Np^(1 / 3) - 1
        isinteger(K) || error("Np must be cubic number")
        return Int(K)
    else
        notimplemented()
    end
end

"""
    const LagrangeLine = LagrangeElement{ReferenceLine}
"""
const LagrangeLine = LagrangeElement{ReferenceLine}

const Line1D{T} = LagrangeElement{ReferenceLine,2,SVector{1,T}}
const Line2D{T} = LagrangeElement{ReferenceLine,2,SVector{2,T}}
const Line3D{T} = LagrangeElement{ReferenceLine,2,SVector{3,T}}
Line1D(args...) = Line1D{Float64}(args...)
Line2D(args...) = Line2D{Float64}(args...)
Line3D(args...) = Line3D{Float64}(args...)

integration_measure(l::Line1D) = norm(vals(l)[2] - vals(l)[1])

"""
    const LagrangeTriangle = LagrangeElement{ReferenceTriangle}
"""
const LagrangeTriangle = LagrangeElement{ReferenceTriangle}

const Triangle2D{T} = LagrangeElement{ReferenceTriangle,3,SVector{2,T}}
const Triangle3D{T} = LagrangeElement{ReferenceTriangle,3,SVector{3,T}}
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

const Quadrangle2D{T} = LagrangeElement{ReferenceSquare,4,SVector{2,T}}
const Quadrangle3D{T} = LagrangeElement{ReferenceSquare,4,SVector{3,T}}
Quadrangle2D(args...) = Quadrangle2D{Float64}(args...)
Quadrangle3D(args...) = Quadrangle3D{Float64}(args...)

"""
    const LagrangeSquare = LagrangeElement{ReferenceSquare}
"""
const LagrangeCube = LagrangeElement{ReferenceCube}

"""
    vertices_idxs(el::LagrangeElement)

The indices of the nodes in `el` that define the vertices of the element.
"""
vertices_idxs(::Type{<:LagrangeLine}) = 1:2
vertices_idxs(::Type{<:LagrangeTriangle}) = 1:3
vertices_idxs(::Type{<:LagrangeSquare}) = 1:4
vertices_idxs(::Type{<:LagrangeTetrahedron}) = 1:4
vertices_idxs(::Type{<:LagrangeCube}) = 1:8
vertices_idxs(el::LagrangeElement) = vertices_idxs(typeof(el))

"""
    vertices(el::LagrangeElement)

Coordinates of the vertices of `el`.
"""
vertices(el::LagrangeElement) = view(vals(el), vertices_idxs(el))

"""
    boundary_idxs(el::LagrangeElement)

The indices of the nodes in `el` that define the boundary of the element.
"""
function boundary_idxs(el::LagrangeLine)
    return 1, length(vals(el))
end

function boundary_idxs(el::LagrangeTriangle{3})
    return (1, 2), (2, 3), (3, 1)
end

function boundary_idxs(el::LagrangeTriangle{6})
    return (1, 2), (2, 3), (3, 1)
end

#=
Hardcode some basic elements.
TODO: Eventually this could/should be automated.
=#

# P0 for ReferenceLine
function reference_nodes(::Type{<:LagrangeLine{1}})
    return SVector(SVector(0.5))
end

function (el::LagrangeLine{1})(u)
    return vals(el)[1]
end

# P1 for ReferenceLine
function reference_nodes(::Type{<:LagrangeLine{2}})
    return SVector(SVector(0.0), SVector(1.0))
end

function (el::LagrangeLine{2})(u)
    v = vals(el)
    return v[1] + (v[2] - v[1]) * u[1]
end

# P2 for ReferenceLine
function reference_nodes(::Type{<:LagrangeLine{3}})
    return SVector(SVector(0.0), SVector(1.0), SVector(0.5))
end

function (el::LagrangeLine{3})(u)
    v = vals(el)
    return v[1] +
           (4 * v[3] - 3 * v[1] - v[2]) * u[1] +
           2 * (v[2] + v[1] - 2 * v[3]) * u[1]^2
end

# P3 for ReferenceLine
function reference_nodes(::Type{<:LagrangeLine{4}})
    return SVector(SVector(0.0), SVector(1.0), SVector(1 / 3), SVector(2 / 3))
end

function (el::LagrangeLine{4})(u)
    v1, v2, v3, v4 = vals(el)
    # Calculate the coefficients based on the values
    a = -9 * v1 / 2 + 9 * v2 / 2 + 27 * v3 / 2 - 27 * v4 / 2
    b = 9 * v1 - 9 * v2 / 2 - 45 * v3 / 2 + 18 * v4
    c = -11 * v1 / 2 + v2 + 9 * v3 - 9 * v4 / 2
    d = v1
    # Evaluate the cubic polynomial at u
    return d + c * u[1] + b * u[1]^2 + a * u[1]^3
end

# P1 for ReferenceTriangle
function reference_nodes(::Type{<:LagrangeTriangle{3}})
    return SVector(SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0))
end

function (el::LagrangeTriangle{3})(u)
    v = vals(el)
    return v[1] + (v[2] - v[1]) * u[1] + (v[3] - v[1]) * u[2]
end

# P2 for ReferenceTriangle
function reference_nodes(::Type{<:LagrangeTriangle{6}})
    return SVector(
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
        SVector(0.5, 0.0),
        SVector(0.5, 0.5),
        SVector(0.0, 0.5),
    )
end

function (el::LagrangeTriangle{6})(u)
    v = vals(el)
    return (1 + u[2] * (-3 + 2u[2]) + u[1] * (-3 + 2u[1] + 4u[2])) * v[1] +
           u[1] *
           (-v[2] + u[1] * (2v[2] - 4v[4]) + 4v[4] + u[2] * (-4v[4] + 4v[5] - 4v[6])) +
           u[2] * (-v[3] + u[2] * (2v[3] - 4v[6]) + 4v[6])
end

# P3 for ReferenceTriangle
# source: https://www.math.uci.edu/~chenlong/iFEM/doc/html/dofP3doc.html
function reference_nodes(::LagrangeTriangle{10})
    return SVector(
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
        SVector(1 / 3, 0.0),
        SVector(2 / 3, 0.0),
        SVector(2 / 3, 1 / 3),
        SVector(1 / 3, 2 / 3),
        SVector(0.0, 2 / 3),
        SVector(0.0, 1 / 3),
        SVector(1 / 3, 1 / 3),
    )
end

function (el::LagrangeTriangle{10})(u)
    λ₁ = 1 - u[1] - u[2]
    λ₂ = u[1]
    λ₃ = u[2]
    ϕ₁ = 0.5 * (3λ₁ - 1) * (3λ₁ - 2) * λ₁
    ϕ₂ = 0.5 * (3λ₂ - 1) * (3λ₂ - 2) * λ₂
    ϕ₃ = 0.5 * (3λ₃ - 1) * (3λ₃ - 2) * λ₃
    ϕ₄ = 4.5 * λ₁ * λ₂ * (3λ₁ - 1)
    ϕ₅ = 4.5 * λ₁ * λ₂ * (3λ₂ - 1)
    ϕ₆ = 4.5 * λ₃ * λ₂ * (3λ₂ - 1)
    ϕ₇ = 4.5 * λ₃ * λ₂ * (3λ₃ - 1)
    ϕ₈ = 4.5 * λ₁ * λ₃ * (3λ₃ - 1)
    ϕ₉ = 4.5 * λ₁ * λ₃ * (3λ₁ - 1)
    ϕ₁₀ = 27 * λ₁ * λ₂ * λ₃
    v = vals(el)
    return v[1] * ϕ₁ +
           v[2] * ϕ₂ +
           v[3] * ϕ₃ +
           v[4] * ϕ₄ +
           v[5] * ϕ₅ +
           v[6] * ϕ₆ +
           v[7] * ϕ₇ +
           v[8] * ϕ₈ +
           v[9] * ϕ₉ +
           v[10] * ϕ₁₀
end

# P1 for ReferenceSquare
function reference_nodes(::Type{<:LagrangeSquare{4}})
    return SVector(SVector(0, 0), SVector(1, 0), SVector(1, 1), SVector(0, 1))
end

function (el::LagrangeElement{ReferenceSquare,4})(u)
    v = vals(el)
    return v[1] +
           (v[2] - v[1]) * u[1] +
           (v[4] - v[1]) * u[2] +
           (v[3] + v[1] - v[2] - v[4]) * u[1] * u[2]
end

# P1 for ReferenceTetrahedron
function reference_nodes(::LagrangeTetrahedron{4})
    return SVector(SVector(0, 0, 0), SVector(1, 0, 0), SVector(0, 1, 0), SVector(0, 0, 1))
end

function (el::LagrangeElement{ReferenceTetrahedron,4})(u)
    v = vals(el)
    return v[1] + (v[2] - v[1]) * u[1] + (v[3] - v[1]) * u[2] + (v[4] - v[1]) * u[3]
end

"""
    degree(el::LagrangeElement)
    degree(el::Type{<:LagrangeElement})

The polynomial degree `el`.
"""
function degree(::Type{<:LagrangeElement{D,Np}})::Int where {D,Np}
    if D == ReferenceLine
        return Np - 1
    elseif D == ReferenceTriangle
        K = (-3 + sqrt(1 + 8 * Np)) / 2
        return K
    elseif D == ReferenceTetrahedron
        notimplemented()
    elseif D == ReferenceSquare
        return sqrt(Np) - 1
    elseif D == ReferenceCube
        return Np^(1 / 3) - 1
    else
        notimplemented()
    end
end
degree(el::LagrangeElement) = typeof(el) |> degree

"""
    lagrange_basis(E::Type{<:LagrangeElement})

Return the Lagrange basis `B` for the element `E`. Evaluating `B(x)` yields the
value of each basis function at `x`.
"""
function lagrange_basis(::Type{LagrangeElement{D,N,T}}) where {D,N,T}
    vals = svector(i -> svector(j -> i == j, N), N)
    return LagrangeElement{D}(vals)
end

# construct a LagrangeElement from a reference shape
function LagrangeElement(::ReferenceLine)
    v = SVector(SVector(0.0), SVector(1.0))
    return LagrangeLine(v)
end
