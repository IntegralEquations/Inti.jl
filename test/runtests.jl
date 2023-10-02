using Inti
using SafeTestsets
using Aqua

@safetestset "Code quality" include("aqua_test.jl")

@safetestset "Utility functions" include("utils_test.jl")

@safetestset "Reference shapes" include("reference_shapes_test.jl")

@safetestset "Reference interpolation" include("reference_interpolation_test.jl")

@safetestset "Reference integration" include("reference_integration_test.jl")
