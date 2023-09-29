#=

Definition of basic geometrical concepts.

=#


"""
    ambient_dimension(x)

Dimension of the ambient space where `x` lives. For geometrical objects this can
differ from its [`geometric_dimension`](@ref); for example a triangle in `ℝ³`
has ambient dimension `3` but geometric dimension `2`, while a curve in `ℝ³` has
ambient dimension 3 but geometric dimension 1.
"""
function ambient_dimension end

"""
    geometric_dimension(x)

Number of independent coordinates needed to describe `x`. Lines have geometric
dimension 1, triangles have geometric dimension 2, etc.
"""
function geometric_dimension end

include("referenceshapes.jl")
