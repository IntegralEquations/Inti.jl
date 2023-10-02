using Inti
using SafeTestsets
using Aqua

@safetestset "Code quality" begin
    include("aqua_test.jl")
end

@safetestset "Geometry" begin
    @safetestset "Reference shapes" include("Geometry/referenceshapes_test.jl")
end
