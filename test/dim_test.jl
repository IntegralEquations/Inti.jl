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
        compression = (method = :hmatrix, tol = 1.0e-10),
        correction = (method = :dim,),
    )
    Sfmm, Dfmm = Inti.single_double_layer(;
        op,
        target = Q,
        source = Q,
        compression = (method = :fmm, tol = 1.0e-10),
        correction = (method = :dim,),
    )
    x = rand(SVector{3, Float64}, size(S, 2))
    @test Sfmm * x ≈ S * x
    @test Dfmm * x ≈ D * x
    @test Shmat * x ≈ S * x
    @test Dhmat * x ≈ D * x
end

@testset "Issue 136 - mixed targets" begin
    Γ = Inti.parametric_curve(0, 2π) do t
        SVector(cos(t), sin(t))
    end |> Inti.Domain

    op = Inti.Laplace(; dim = 2)
    source = Inti.Quadrature(Γ; meshsize = 0.2, qorder = 3)
    target = [SVector(x, y) for x in -1.5:0.1:1.5, y in -1.5:0.1:1.5] |> vec
    idx_in = filter(i -> norm(target[i]) < 1.0, eachindex(target))
    idx_out = filter(i -> norm(target[i]) ≥ 1.0, eachindex(target))
    green_multiplier = zeros(length(target))
    green_multiplier[idx_in] .= -1.0
    green_multiplier[idx_out] .= 0.0

    # assemble the whole thing at one
    S, D = Inti.single_double_layer(;
        op,
        target,
        source,
        correction = (method = :dim, green_multiplier, maxdist = Inf)
    )
    # or do it separately for inside and outside
    S⁺, D⁺ = Inti.single_double_layer(;
        op,
        target = target[idx_out],
        source,
        correction = (method = :dim, target_location = :outside, maxdist = Inf)
    )
    S⁻, D⁻ = Inti.single_double_layer(;
        op,
        target = target[idx_in],
        source,
        correction = (method = :dim, target_location = :inside, maxdist = Inf)
    )

    # now test
    u = ones(length(source))
    @test norm(D⁺ * u, Inf) < 1.0e-3
    @test norm((D⁻ * u .+ 1), Inf) < 1.0e-3

    v = D * u
    @show norm(v[idx_out], Inf) # should be zero (not OK)
    @show norm(v[idx_in] .+ 1, Inf) # close to -1 (not OK)

    # the SL also does not match the one computed separately
    v = S * u
    @test norm(v[idx_in] - S⁻ * u, Inf) < 1.0e-3
    @test norm(v[idx_out] - S⁺ * u, Inf) < 1.0e-3

end
