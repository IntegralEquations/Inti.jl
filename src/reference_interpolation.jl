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
abstract type ReferenceInterpolant{D<:ReferenceShape,T} end

function (el::ReferenceInterpolant)(x)
    return interface_method(el)
end

"""
    jacobian(f,x)

Given a (possibly vector-valued) functor `f : 𝐑ᵐ → 𝐅ⁿ`, return the `n × m`
matrix `Aᵢⱼ = ∂fᵢ/∂xⱼ`.Both `x` and `f(x)` are expected to be of `SVector` type.
"""
function jacobian(f, x)
    return interface_method(f)
end

domain(::ReferenceInterpolant{D,T}) where {D,T} = D()
return_type(::ReferenceInterpolant{D,T}) where {D,T} = T
domain_dimension(t::ReferenceInterpolant) = domain(t) |> center |> length
range_dimension(el::ReferenceInterpolant{R,T}) where {R,T} = domain(el) |> center |> el |> length

"""
    struct LagrangeElement{D,Np,T} <: ReferenceInterpolant{D,T}

A polynomial `p : D → T` uniquely defined by its `Np` values on the `Np` reference nodes
of `D`.

The return type `T` should be a vector space (i.e. support addition and
multiplication by scalars). For istance, `T` could be a number or a vector, but
not a `Tuple`.
"""
struct LagrangeElement{D,Np,T} <: ReferenceInterpolant{D,T}
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
LagrangeElement{D}(x1,xs...) where {D} = LagrangeElement{D}(SVector(x1,xs...))

# construct based on a function
function LagrangeElement{D}(f::Function) where {D}
    ref_nodes = reference_nodes(D())
    vals = svector(i -> f(ref_nodes[i]), length(ref_nodes))
    return LagrangeElement{D}(vals)
end

"""
    const LagrangeLine = LagrangeElement{ReferenceLine}
"""
const LagrangeLine = LagrangeElement{ReferenceLine}

"""
    const LagrangeTriangle = LagrangeElement{ReferenceTriangle}
"""
const LagrangeTriangle = LagrangeElement{ReferenceTriangle}

"""
    const LagrangeTetrahedron = LagrangeElement{ReferenceTetrahedron}
"""
const LagrangeTetrahedron = LagrangeElement{ReferenceTetrahedron}

"""
    const LagrangeSquare = LagrangeElement{ReferenceSquare}
"""
const LagrangeSquare = LagrangeElement{ReferenceSquare}

"""
    const LagrangeSquare = LagrangeElement{ReferenceSquare}
"""
const LagrangeCube = LagrangeElement{ReferenceCube}

#=
Hardcode some basic elements.
TODO: Eventually this could/should be automated.
=#

# P1 for ReferenceLine
function reference_nodes(::Type{LagrangeLine{2}})
    return SVector(SVector(0.0), SVector(1.0))
end

function (el::LagrangeLine{2})(u)
    v = vals(el)
    return v[1] + (v[2] - v[1]) * u[1]
end

function jacobian(el::LagrangeLine{2}, u)
    v = vals(el)
    return hcat(v[2] - v[1])
end

# P2 for ReferenceLine
function reference_nodes(::Type{LagrangeLine{3}})
    return SVector(SVector(0.0), SVector(1.0), SVector(0.5))
end

function (el::LagrangeLine{3})(u)
    v = vals(el)
    return v[1] + (4 * v[3] - 3 * v[1] - v[2]) * u[1] +
           2 * (v[2] + v[1] - 2 * v[3]) * u[1]^2
end

function jacobian(el::LagrangeLine{3}, u)
    v = vals(el)
    return hcat(4 * v[3] - 3 * v[1] - v[2] + 4 * (v[2] + v[1] - 2 * v[3]) * u[1])
end

# P1 for ReferenceTriangle
function reference_nodes(::Type{LagrangeTriangle{3}})
    return SVector(SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0))
end

function (el::LagrangeTriangle{3})(u)
    v = vals(el)
    return v[1] + (v[2] - v[1]) * u[1] + (v[3] - v[1]) * u[2]
end

function jacobian(el::LagrangeTriangle{3}, u)
    v = vals(el)
    jac = hcat(v[2] - v[1], v[3] - v[1])
    return jac
end

# P2 for ReferenceTriangle
function reference_nodes(::LagrangeTriangle{6})
    return SVector(SVector(0.0, 0.0),
                   SVector(1.0, 0.0),
                   SVector(0.0, 1.0),
                   SVector(0.5, 0.0),
                   SVector(0.5, 0.5),
                   SVector(0.0, 0.5))
end

function (el::LagrangeTriangle{6})(u)
    v = vals(el)
    return (1 + u[2] * (-3 + 2u[2]) + u[1] * (-3 + 2u[1] + 4u[2])) * v[1] + u[1] *
           (-v[2] + u[1] * (2v[2] - 4v[4]) + 4v[4] + u[2] * (-4v[4] + 4v[5] - 4v[6])) +
           u[2] * (-v[3] + u[2] * (2v[3] - 4v[6]) + 4v[6])
end

function jacobian(el::LagrangeTriangle{6}, u)
    v = vals(el)
    return hcat((-3 + 4u[1] + 4u[2]) * v[1] - v[2] +
                u[1] * (4v[2] - 8v[4]) +
                4v[4] +
                u[2] * (-4v[4] + 4v[5] - 4v[6]),
                (-3 + 4u[1] + 4u[2]) * v[1] - v[3] +
                u[2] * (4v[3] - 8v[6]) +
                u[1] * (-4v[4] + 4v[5] - 4v[6]) +
                4v[6])
end

# P1 for ReferenceSquare
function reference_nodes(::Type{LagrangeSquare{4}})
    return SVector(SVector(0, 0),
                   SVector(1, 0),
                   SVector(1, 1),
                   SVector(0, 1))
end

function (el::LagrangeElement{ReferenceSquare,4})(u)
    v = vals(el)
    return v[1] +
           (v[2] - v[1]) * u[1] +
           (v[4] - v[1]) * u[2] +
           (v[3] + v[1] - v[2] - v[4]) * u[1] * u[2]
end

function jacobian(el::LagrangeElement{ReferenceSquare,4}, u)
    v = vals(el)
    return hcat(((v[2] - v[1]) + (v[3] + v[1] - v[2] - v[4]) * u[2]),
                ((v[4] - v[1]) + (v[3] + v[1] - v[2] - v[4]) * u[1]))
end

# P1 for ReferenceTetrahedron
function reference_nodes(::LagrangeTetrahedron{4})
    return SVector(SVector(0, 0, 0),
                   SVector(1, 0, 0),
                   SVector(0, 1, 0),
                   SVector(0, 0, 1))
end

function (el::LagrangeElement{ReferenceTetrahedron,4})(u)
    v = vals(el)
    return v[1] + (v[2] - v[1]) * u[1] + (v[3] - v[1]) * u[2] + (v[4] - v[1]) * u[3]
end

function jacobian(el::LagrangeElement{ReferenceTetrahedron,4}, u)
    v = vals(el)
    return hcat((v[2] - v[1]), (v[3] - v[1]), (v[4] - v[1]))
end

"""
    degree(el::LagrangeElement)
    degree(el::Type{<:LagrangeElement})

The polynomial degree `el`.
"""
function degree(::Type{LagrangeElement{D,Np}})::Int where {D,Np}
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