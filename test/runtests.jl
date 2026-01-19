using Inti
using SafeTestsets
using Test
using Aqua

@safetestset "Code quality" include("aqua_test.jl")

@safetestset "Utility functions" include("utils_test.jl")

@safetestset "Block array" include("blockarray_test.jl")

@safetestset "Reference shapes" include("reference_shapes_test.jl")

@safetestset "Polynomials" include("polynomials_test.jl")

@safetestset "Reference interpolation" include("reference_interpolation_test.jl")

@safetestset "Reference integration" include("reference_integration_test.jl")

@safetestset "Native mesh generation" include("meshgen_test.jl")

@safetestset "Quadrature" include("quadrature_test.jl")

@safetestset "Normal orientation" include("normal_orientation_test.jl")

@safetestset "Kernels" include("kernels_test.jl")

@safetestset "Integral operators" include("integral_operator_test.jl")

@safetestset "Guiggiani" include("guiggiani_test.jl")

@testset verbose = true "Corrections (Green identities)" begin
    include("green_identities_test.jl")
    include("volume_potential_test.jl")
end

@testset "Accelerated density interpolation" include("dim_test.jl")

@safetestset "Gmsh extension" include("gmsh_test.jl")

@safetestset "Meshes extension" include("meshes_test.jl")

@safetestset "HMatrices extension" include("hmatrices_test.jl")

@safetestset "FMM2D extension" include("fmm2d_test.jl")

@safetestset "FMM3D extension" include("fmm3d_test.jl")

@safetestset "Curve 2D Mesh" include("curved_test_2d.jl")

@safetestset "Curve 3D Mesh" include("curved_test_3d.jl")
