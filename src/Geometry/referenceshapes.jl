"""
    abstract type AbstractReferenceShape

A fixed reference domain/shape. Used mostly for defining more complex shapes as
transformations mapping an `AbstractReferenceShape` to some region of `ℜᴹ`.

See e.g. [`ReferenceLine`](@ref) or [`ReferenceTriangle`](@ref) for some
examples of concrete subtypes.
"""
abstract type AbstractReferenceShape end

"""
    struct ReferenceTriangle

Singleton type representing the triangle with vertices `(0,0),(1,0),(0,1)`
"""
struct ReferenceTriangle <: AbstractReferenceShape end
geometric_dimension(::ReferenceTriangle) = 2
ambient_dimension(::ReferenceTriangle) = 2
vertices(::ReferenceTriangle) = SVector(0, 0), SVector(1, 0), SVector(0, 1)
Base.in(x, ::ReferenceTriangle) = 0 ≤ x[1] ≤ 1 && 0 ≤ x[2] ≤ 1 - x[1]
center(::ReferenceTriangle) = svector(i -> 1 / 3, 3)

"""
    struct ReferenceTetrahedron

Singleton type representing the tetrahedron with vertices
`(0,0,0),(0,0,1),(0,1,0),(1,0,0)`
"""
struct ReferenceTetrahedron <: AbstractReferenceShape end
geometric_dimension(::ReferenceTetrahedron) = 3
ambient_dimension(::ReferenceTetrahedron) = 3
vertices(::ReferenceTetrahedron) = SVector(0, 0, 0), SVector(1, 0, 0), SVector(0, 1, 0), SVector(0, 0, 1)
Base.in(x,::ReferenceTetrahedron) = 0 ≤ x[1] ≤ 1 && 0 ≤ x[2] ≤ 1 - x[1] && 0 ≤ x[3] ≤ 1 - x[1] - x[2]
center(::ReferenceTetrahedron) = svector(i -> 1 / 4, 4)


# TODO: generalize structs above to `ReferenceSimplex{N}` and

"""
    struct ReferenceHyperCube{N} <: AbstractReferenceShape{N}

Singleton type representing the axis-aligned hypercube in `N` dimensions with
the lower corner at the origin and the upper corner at `(1,1,…,1)`.
"""
struct ReferenceHyperCube{N} <: AbstractReferenceShape end
geometric_dimension(::ReferenceHyperCube{N}) where {N} = N
ambient_dimension(::ReferenceHyperCube{N}) where {N} = N
vertices(::ReferenceHyperCube{N}) where {N} = ntuple(i -> SVector(ntuple(j -> (i >> j) & 1, N)), 2^N)
Base.in(x, ::ReferenceHyperCube{N}) where {N} = all(0 .≤ x .≤ 1)
center(::ReferenceHyperCube{N}) where {N} = svector(i -> 0.5, N)

"""
    const ReferenceLine = ReferenceHyperCube{1}

Singleton type representing the `[0,1]` segment.
"""
const ReferenceLine = ReferenceHyperCube{1}

"""
    const ReferenceSquare = ReferenceHyperCube{2}

Singleton type representing the unit square `[0,1]²`.
"""
const ReferenceSquare = ReferenceHyperCube{2}

"""
    const ReferenceCube = ReferenceHyperCube{3}

Singleton type representing the unit cube `[0,1]³`.
"""
const ReferenceCube = ReferenceHyperCube{3}
