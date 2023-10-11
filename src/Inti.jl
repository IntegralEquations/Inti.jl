module Inti

const PROJECT_ROOT = pkgdir(Inti)

using LinearAlgebra
using StaticArrays
using SpecialFunctions
using Printf

# helper functions
include("utils.jl")

# basic interpolation and integration
include("reference_shapes.jl")
include("reference_interpolation.jl")
include("quad_rules_tables.jl")
include("reference_integration.jl")

# geometry meshes, and quadratures
include("entities.jl")
include("domain.jl")
include("mesh.jl")
include("quadrature.jl")

# Nyström methods
include("kernels.jl")
include("nystrom.jl")

# # integral operators
# include("integral_potentials.jl")
# include("integral_operators.jl")

end
