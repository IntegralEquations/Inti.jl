using Inti
using SafeTestsets
using Aqua

@safetestset "Code quality" include("aqua_test.jl")

@safetestset "Reference shapes" include("reference_shapes_test.jl")
