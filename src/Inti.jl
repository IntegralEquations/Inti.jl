module Inti

const PROJECT_ROOT = pkgdir(Inti)

using LinearAlgebra
using NearestNeighbors
using SparseArrays
using StaticArrays
using SpecialFunctions
using Printf

# helper functions
include("utils.jl")
include("blockmatrix.jl")

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
include("dim.jl")

# some zero-argument methods for the Inti's gmsh extension
include("gmsh_api.jl")

end
