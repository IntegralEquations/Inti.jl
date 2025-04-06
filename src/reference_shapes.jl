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
        0 ≤ x[i] ≤ 1 - sum(x[1:(i-1)]) || return false
    end
    return true
end
function vertices(::ReferenceSimplex{N}) where {N}
    return svector(
        i -> i == 1 ? zero(SVector{N,Int64}) : standard_basis_vector(i - 1, Val(N)),
        N + 1,
    )
end
center(::ReferenceSimplex{N}) where {N} = svector(i -> 1 / (N + 1), N)

"""
    struct ReferenceTriangle

Singleton type representing the triangle with vertices `(0,0),(1,0),(0,1)`
"""
const ReferenceTriangle = ReferenceSimplex{2}

"""
    struct ReferenceTetrahedron

Singleton type representing the tetrahedron with vertices
`(0,0,0),(0,0,1),(0,1,0),(1,0,0)`
"""
const ReferenceTetrahedron = ReferenceSimplex{3}

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

vertices(::ReferenceLine) = (SVector(0), SVector(1))

"""
    const ReferenceSquare = ReferenceHyperCube{2}

Singleton type representing the unit square `[0,1]²`.
"""
const ReferenceSquare = ReferenceHyperCube{2}

function vertices(::ReferenceSquare)
    return (SVector(0, 0), SVector(1, 0), SVector(1, 1), SVector(0, 1))
end

"""
    const ReferenceCube = ReferenceHyperCube{3}

Singleton type representing the unit cube `[0,1]³`.
"""
const ReferenceCube = ReferenceHyperCube{3}

function vertices(::ReferenceCube)
    return (
        SVector(0, 0, 0),
        SVector(1, 0, 0),
        SVector(1, 1, 0),
        SVector(0, 1, 0),
        SVector(0, 0, 1),
        SVector(1, 0, 1),
        SVector(1, 1, 1),
        SVector(0, 1, 1),
    )
end

# since ReferenceShapes are singletons, define methods on the type to be
# equivalent to methods on instantiation of the type
geometric_dimension(E::Type{<:ReferenceShape}) = geometric_dimension(E())
ambient_dimension(E::Type{<:ReferenceShape}) = ambient_dimension(E())
vertices(E::Type{<:ReferenceShape}) = vertices(E())
center(E::Type{<:ReferenceShape}) = center(E())
Base.in(E::Type{<:ReferenceShape}) = in(E())
