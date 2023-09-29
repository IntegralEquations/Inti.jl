using Inti
using Test
using Aqua

@testset "Inti.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Inti)
    end
    # Write your tests here.
end
