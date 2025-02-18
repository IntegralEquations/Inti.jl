using Inti
using Test
using StaticArrays

@testset "Yukawa" begin
    for dim in (2, 3)
        x = @SVector rand(dim)
        y = @SVector rand(dim)
        nx = @SVector rand(dim)
        ny = @SVector rand(dim)
        target = (; coords = x, normal = nx)
        source = (; coords = y, normal = ny)
        λ = rand()
        k = im * λ
        yuka = Inti.Yukawa(; dim, λ)
        helm = Inti.Helmholtz(; dim, k)
        @testset "$dim dimensions" begin
            for kernel in (
                Inti.SingleLayerKernel,
                Inti.DoubleLayerKernel,
                Inti.AdjointDoubleLayerKernel,
                Inti.HyperSingularKernel,
            )
                @test kernel(yuka)(target, source) ≈ kernel(helm)(target, source)
            end
        end
    end
end
