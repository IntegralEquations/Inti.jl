"""
    abstract type AbstractReferenceShape

A fixed reference domain/shape. Used mostly for defining more complex shapes as
transformations mapping an `AbstractReferenceShape` to some region of `ℜᴹ`.

See e.g. [`ReferenceLine`](@ref) or [`ReferenceTriangle`](@ref) for some
examples of concrete subtypes.
"""
abstract type AbstractReferenceShape end

"""
    struct ReferenceSimplex{N}

Singleton type representing the N-simplex with N+1 vertices
`(0,...,0),(0,...,0,1),(0,...,0,1,0),(1,0,...,0)`
"""
struct ReferenceSimplex{N} <: AbstractReferenceShape end
geometric_dimension(::ReferenceSimplex{N}) where {N} = N
ambient_dimension(::ReferenceSimplex{N}) where {N} = N
function Base.in(x, ::ReferenceSimplex{N}) where {N}
    for i in 1:N
        0 ≤ x[i] ≤ 1 - sum(x[1:i-1]) || return false
    end
    return true
end
vertices(::ReferenceSimplex{N}) where {N} = svector(i -> i == 1 ? zero(SVector{N, Int64}) : standard_basis_vector(i-1, Val(N)), N+1)
center(::ReferenceSimplex{N}) where {N} = svector(i -> 1 / (N + 1), N )

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
