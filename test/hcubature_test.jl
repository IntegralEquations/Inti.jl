using Test
using LinearAlgebra
using Inti
using Random
using HCubature

include("test_utils.jl")

Random.seed!(1)
atol = 1e-10
meshsize = 0.1

fevals = Ref(0) # counter for function evaluations

@testset "Segment integration" begin
    # test ∫₀¹ log(x - s) dx. For s ∈ [0,1] the integrand is weakly singular. For
    # s > 1, the integrand is analytic on the interval.
    # NOTE: for consistency with higher-dimensional cases, scalars are
    # represented as one-dimensional static vectors.
    τ̂ = Inti.ReferenceLine()
    f = (x, s) -> (fevals[] += 1; log(norm(x - s)))
    I = (s) -> -1 + s[1] * log(s[1]) + (1 - s[1]) * log(norm(1 - s[1]))
    # not singular
    s = Inti.Point1D(2)
    fevals[] = 0
    I1, E1 = Inti.hcubature(x -> f(x, s), τ̂; atol)
    @test fevals[] < 50
    @test norm(I(s) - I1) < atol
    # singular
    s = Inti.Point1D(1 / 3)
    fevals[] = 0
    I1, E1 = Inti.hcubature(x -> f(x, s), τ̂; atol) # naive
    n1 = fevals[]
    @test norm(I(s) - I1) < atol
    fevals[] = 0
    I1, E1 = Inti.hcubature(x -> f(x, s), τ̂, s; atol) #
    n2 = fevals[]
    @test norm(I(s) - I1) < atol
    @test n2 < n1
end

@testset "Triangle integration" begin
    τ̂ = Inti.ReferenceTriangle()
    # not singular
    @test Inti.hcubature(x -> 1, τ̂)[1] ≈ 0.5
    f = x -> (fevals[] += 1; cos(x[1]) * sin(x[2]))
    Ie = 1 / 2 * (sin(1) - cos(1))
    fevals[] = 0
    Ia, Ea = Inti.hcubature(f, τ̂)
    n1 = fevals[]
    @test Ie ≈ Ia
    # singular kernel at right vertex
    s = Inti.Point2D(1, 0)
    f = (x, s) -> (fevals[] += 1; 1 / norm(x - s))
    fevals[] = 0
    Ia, Ea = Inti.hcubature(x -> f(x, s), τ̂)
    n1 = fevals[]
    Ie = acosh(sqrt(2))
    @test Ia ≈ Ie
    # singular kernel in the middle
    s = Inti.Point2D(1 / 3, 1 / 3)
    fevals[] = 0
    I1, E1 = Inti.hcubature(x -> f(x, s), τ̂; atol)
    n1 = fevals[]
    fevals[] = 0
    I2, E2 = Inti.hcubature(x -> f(x, s), τ̂, s; atol)
    n2 = fevals[]
    @test I1 ≈ I2
    @test n2 < n1
end

@testset "Square integration" begin
    τ̂ = Inti.ReferenceSquare()
    # not singular
    @test Inti.hcubature(x -> 1, τ̂)[1] ≈ 1.0
    @test Inti.hcubature(x -> 1, τ̂, Inti.Point2D(1 / 3, 1 / 3))[1] ≈ 1.0
    f = x -> (fevals[] += 1; cos(x[1]) * sin(x[2]))
    Ie = -sin(1) * (cos(1) - 1)
    fevals[] = 0
    Ia, Ea = Inti.hcubature(f, τ̂; atol)
    n1 = fevals[]
    @test norm(Ia - Ie) < atol
    # singular kernel at interior point
    s = Inti.Point2D(1 / 3, 1 / 3)
    els = Inti.decompose(τ̂, s)
    f = (x, s) -> (fevals[] += 1; 1 / norm(x - s))
    fevals[] = 0
    I1, E1 = Inti.hcubature(x -> f(x, s), τ̂; atol)
    n1 = fevals[]
    fevals[] = 0
    I2, E2 = Inti.hcubature(x -> f(x, s), τ̂, s; atol)
    n2 = fevals[]
    @test I1 ≈ I2
    @test n2 < n1
end
