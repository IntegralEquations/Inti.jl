using Inti
using Test
using StaticArrays
using ForwardDiff
using QPGreen

## Extend QuadGK to support ForwardDiff.Dual types (see https://github.com/JuliaMath/QuadGK.jl/issues/122)
## (only needed if computing derivatives of QPGreen using ForwardDiff)
# using QuadGK
# function QuadGK.cachedrule(
#     ::Type{<:ForwardDiff.Dual{<:Any,T}},
#     n::Integer,
# ) where {T<:Number}
#     return QuadGK._cachedrule(typeof(float(real(one(T)))), Int(n))
# end

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

@testset "Laplace periodic 1d" begin
    dim     = 2
    x       = @SVector rand(dim)
    y       = @SVector rand(dim)
    nx      = @SVector rand(dim)
    ny      = @SVector rand(dim)
    target  = (; coords = x, normal = nx)
    source  = (; coords = y, normal = ny)
    period  = rand()
    op      = Inti.LaplacePeriodic1D(; dim, period)
    G       = Inti.SingleLayerKernel(op)
    dGdny   = Inti.DoubleLayerKernel(op)
    dGdnx   = Inti.AdjointDoubleLayerKernel(op)
    d2Gdnxy = Inti.HyperSingularKernel(op)
    # test that the normal derivatives are correct
    @test ForwardDiff.derivative(t -> G((coords = x + t * nx,), y), 0) ≈
          dGdnx((coords = x, normal = nx), y)
    @test ForwardDiff.derivative(t -> G(x, (coords = y + t * ny,)), 0) ≈
          dGdny(x, (coords = y, normal = ny))
    @test ForwardDiff.derivative(
        t -> dGdny((coords = x + t * nx,), (coords = y, normal = ny)),
        0,
    ) ≈ d2Gdnxy((coords = x, normal = nx), (coords = y, normal = ny))
    # test periodicity
    @test G(x .+ (period, 0), y) ≈ G(x, y)
    @test G(x .+ (-3 * period, 0), y) ≈ G(x, y)
end

@testset "Helmholtz periodic 1d" begin
    dim     = 2
    x       = @SVector rand(dim)
    y       = @SVector rand(dim)
    nx      = @SVector rand(dim)
    ny      = @SVector rand(dim)
    target  = (; coords = x, normal = nx)
    source  = (; coords = y, normal = ny)
    alpha   = rand()
    k       = rand()
    op      = Inti.HelmholtzPeriodic1D(; alpha, k, dim)
    G       = Inti.SingleLayerKernel(op)
    dGdny   = Inti.DoubleLayerKernel(op)
    dGdnx   = Inti.AdjointDoubleLayerKernel(op)
    d2Gdnxy = Inti.HyperSingularKernel(op)
    # test that the normal derivatives are correct
    # @test isapprox(
    #     ForwardDiff.derivative(t -> G((coords = x + t * nx,), y), 0),
    #     dGdnx((coords = x, normal = nx), y);
    #     atol = 1e-6,
    # )
    # @test isapprox(
    #     ForwardDiff.derivative(t -> G(x, (coords = y + t * ny,)), 0),
    #     dGdny(x, (coords = y, normal = ny));
    #     atol = 1e-6,
    # )
    # @test ForwardDiff.derivative(
    #     t -> dGdny((coords = x + t * nx,), (coords = y, normal = ny)),
    #     0,
    # ) ≈ d2Gdnxy((coords = x, normal = nx), (coords = y, normal = ny))
    # test quasi-periodicity
    period = 2π
    @test G(x .+ (period, 0), y) ≈ exp(im * alpha * period) * G(x, y)
    @test G(x .+ (-3 * period, 0), y) ≈ exp(im * alpha * -3 * period) * G(x, y)
end
