module Inti

const PROJECT_ROOT = pkgdir(Inti)

using LinearAlgebra
using NearestNeighbors
using SparseArrays
using StaticArrays
using LinearMaps
using SpecialFunctions
using Printf

# PolynomialSolutions module. TODO: decide if we register this module as a
# package (in which case we may want to rename it), or just keep it here as a
# submodule.
include("PolynomialSolutions.jl")

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
include("dim.jl")

# some zero-argument methods for the Inti's gmsh api
include("gmsh_api.jl")

end
