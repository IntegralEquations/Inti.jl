"""
    abstract type ReferenceShape

A fixed reference domain/shape. Used mostly for defining more complex shapes as
transformations mapping an `ReferenceShape` to some region of `ℜᴹ`.

See e.g. [`ReferenceLine`](@ref) or [`ReferenceTriangle`](@ref) for some
examples of concrete subtypes.
"""
abstract type ReferenceShape end

"""
    struct ReferenceSimplex{N}

Singleton type representing the N-simplex with N+1 vertices
`(0,...,0),(0,...,0,1),(0,...,0,1,0),(1,0,...,0)`
"""
struct ReferenceSimplex{N} <: ReferenceShape end
geometric_dimension(::ReferenceSimplex{N}) where {N} = N
ambient_dimension(::ReferenceSimplex{N}) where {N} = N
function Base.in(x, ::ReferenceSimplex{N}) where {N}
    for i in 1:N
        0 ≤ x[i] ≤ 1 - sum(view(x, 1:i-1); init = zero(x[i])) || return false
    end
    return true
end
center(::ReferenceSimplex{N}) where {N} = svector(i -> 1 / (N + 1), N)

"""
    struct ReferenceTriangle

Singleton type representing the triangle with vertices `(0,0),(1,0),(0,1)`
"""
const ReferenceTriangle = ReferenceSimplex{2}

vertices(::ReferenceTriangle) = Point2D(0, 0), Point2D(1, 0), Point2D(0, 1)

"""
    struct ReferenceTetrahedron

Singleton type representing the tetrahedron with vertices
`(0,0,0),(0,0,1),(0,1,0),(1,0,0)`
"""
const ReferenceTetrahedron = ReferenceSimplex{3}

function vertices(::ReferenceTetrahedron)
    return Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0), Point3D(0, 0, 1)
end

"""
    struct ReferenceHyperCube{N} <: ReferenceShape{N}

Singleton type representing the axis-aligned hypercube in `N` dimensions with
the lower corner at the origin and the upper corner at `(1,1,…,1)`.
"""
struct ReferenceHyperCube{N} <: ReferenceShape end
geometric_dimension(::ReferenceHyperCube{N}) where {N} = N
ambient_dimension(::ReferenceHyperCube{N}) where {N} = N
Base.in(x, ::ReferenceHyperCube{N}) where {N} = all(0 .≤ x .≤ 1)
center(::ReferenceHyperCube{N}) where {N} = svector(i -> 0.5, N)

"""
    const ReferenceLine = ReferenceHyperCube{1}

Singleton type representing the `[0,1]` segment.
"""
const ReferenceLine = ReferenceHyperCube{1}

vertices(::ReferenceLine) = Point1D(0), Point1D(1)

"""
    const ReferenceSquare = ReferenceHyperCube{2}

Singleton type representing the unit square `[0,1]²`.
"""
const ReferenceSquare = ReferenceHyperCube{2}

vertices(::ReferenceSquare) = Point2D(0, 0), Point2D(1, 0), Point2D(1, 1), Point2D(0, 1)

"""
    const ReferenceCube = ReferenceHyperCube{3}

Singleton type representing the unit cube `[0,1]³`.
"""
const ReferenceCube = ReferenceHyperCube{3}

# since ReferenceShapes are singletons, define methods on the type to be
# equivalent to methods on instantiation of the type
geometric_dimension(E::Type{<:ReferenceShape}) = geometric_dimension(E())
ambient_dimension(E::Type{<:ReferenceShape}) = ambient_dimension(E())
vertices(E::Type{<:ReferenceShape}) = vertices(E())
center(E::Type{<:ReferenceShape}) = center(E())
Base.in(x, E::Type{<:ReferenceShape}) = in(x, E())
