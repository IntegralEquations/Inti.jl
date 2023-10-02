module Inti

const PROJECT_ROOT = pkgdir(Inti)

using StaticArrays

# helper functions
include("utils.jl")

# basic interpolation and integration
include("reference_shapes.jl")
include("reference_interpolation.jl")
# include("reference_integration.jl")

# # geometry and meshes
# include("domain.jl")
# include("mesh.jl")

# # integral operators
# include("integral_potentials.jl")
# include("integral_operators.jl")


end
