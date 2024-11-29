"""
    module Inti

Library for solving integral equations using Nyström methods.
"""
module Inti

const PROJECT_ROOT = pkgdir(Inti)

using DataStructures
using ForwardDiff
using LinearAlgebra
using LinearMaps
using NearestNeighbors
using Pkg
using QuadGK
using Scratch
using SparseArrays
using StaticArrays
using Printf
using TOML

import ElementaryPDESolutions
import SpecialFunctions
import Bessels # faster than SpecialFunctions for Bessel functions with real args

# helper functions
include("utils.jl")
include("blockarray.jl")

# basic interpolation and integration
include("reference_shapes.jl")
include("polynomials.jl")
include("reference_interpolation.jl")
include("quad_rules_tables.jl")
include("reference_integration.jl")

# geometry meshes, and quadratures
include("entities.jl")
include("domain.jl")
include("simpleshapes.jl")
include("mesh.jl")
include("quadrature.jl")

# Nyström methods
include("kernels.jl")
include("nystrom.jl")
include("bdim.jl")
include("lbdim.jl")
include("vdim.jl")
include("adaptive.jl")

# some zero-argument methods for the Inti's gmsh extension
include("gmsh_api.jl")

# API
include("api.jl")

end
