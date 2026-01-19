"""
    module Inti

Library for solving integral equations using Nyström methods.
"""
module Inti

const PROJECT_ROOT = pkgdir(Inti)

import Bessels
import ElementaryPDESolutions
import HAdaptiveIntegration
import SpecialFunctions

import ElementaryPDESolutions: Polynomial

using DataStructures
using ForwardDiff
using LinearAlgebra
using LinearMaps
using NearestNeighbors
using Pkg
using Printf
using QuadGK
using Richardson
using Scratch
using SparseArrays
using StaticArrays
using TOML

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
include("adaptive_correction.jl")
include("bdim.jl")
include("vdim.jl")

# some zero-argument methods for the Inti's gmsh extension
include("gmsh_api.jl")

# API
include("api.jl")

end
