using Inti
using HMatrices
using FMM3D
using LinearAlgebra
using StaticArrays
using Test

@testset "Issue 118" begin
    op = Inti.Stokes(; dim = 3, μ = 2.0)
    # op = Inti.Elastostatic(; dim = 3, μ = 2.0, λ = 1.0)
    Ω = Inti.ellipsoid() |> Inti.Domain
    Γ = Inti.boundary(Ω)
    Q = Inti.Quadrature(Γ; meshsize = 0.2, qorder = 2)
    S, D = Inti.single_double_layer(;
        op,
        target = Q,
        source = Q,
        compression = (method = :none,),
        correction = (method = :dim,),
    )
    Shmat, Dhmat = Inti.single_double_layer(;
        op,
        target = Q,
        source = Q,
        compression = (method = :hmatrix, tol = 1e-10),
        correction = (method = :dim,),
    )
    Sfmm, Dfmm = Inti.single_double_layer(;
        op,
        target = Q,
        source = Q,
        compression = (method = :fmm, tol = 1e-10),
        correction = (method = :dim,),
    )
    x = rand(SVector{3,Float64}, size(S, 2))
    @test Sfmm * x ≈ S * x
    @test Dfmm * x ≈ D * x
    @test Shmat * x ≈ S * x
    @test Dhmat * x ≈ D * x
end
