using Test
using LinearAlgebra
using Inti
using Random
using StaticArrays

include("test_utils.jl")

Random.seed!(1)
atol = 1e-4
meshsize = 0.2

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
    s = SVector(2)
    fevals[] = 0
    I1, E1 = Inti.adaptive_integration(x -> f(x, s), τ̂; atol)
    @test fevals[] == 15 # default Gauss-Kronrod rule has 15 points
    @test norm(I(s) - I1) < atol
    # singular at 0
    s = SVector(0)
    fevals[] = 0
    I1, E1 = Inti.adaptive_integration(x -> f(x, s), τ̂; atol)
    @test norm(-1 - I1) < atol
    # singular inside
    s = SVector(1 / 3)
    fevals[] = 0
    I1, E1 = Inti.adaptive_integration(x -> f(x, s), τ̂; atol) # naive
    n1 = fevals[]
    @test E1 < atol
    @test norm(I(s) - I1) > atol
    # manually indicate the location of singularity
    s = SVector(1 / 2)
    fevals[] = 0
    I1, E1 = Inti.adaptive_integration_singular(x -> f(x, s), τ̂, s; atol)
    @test norm(I(s) - I1) < atol
    n2 = fevals[]
end

@testset "Boundary integral operators" begin
    for N in (2,)
        # create geometry
        Inti.clear_entities!()
        if N == 2
            Ω, msh = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize)
        else
            Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize)
        end
        Γ = Inti.external_boundary(Ω)
        Γ_msh = view(msh, Γ)
        Γ_quad = Inti.Quadrature(view(msh, Γ); qorder = 5)
        for t in (:interior, :exterior)
            σ = t == :interior ? 1 / 2 : -1 / 2
            ops = (
                Inti.Laplace(; dim = N),
                Inti.Helmholtz(; k = 1.2, dim = N),
                # Inti.Stokes(; μ = 1.2, dim = N),
            )
            for op in ops
                @testset "Greens identity ($t) $(N)d $op" begin
                    xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
                    T = Inti.default_density_eltype(op)
                    c = rand(T)
                    u = (qnode) -> Inti.SingleLayerKernel(op)(qnode, xs) * c
                    dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(op)(qnode, xs) * c
                    γ₀u = map(u, Γ_quad)
                    γ₁u = map(dudn, Γ_quad)
                    γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
                    γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
                    # single and double layer
                    G = Inti.SingleLayerKernel(op)
                    S = Inti.IntegralOperator(G, Γ_quad)
                    S0 = Inti.assemble_matrix(S)
                    dG = Inti.DoubleLayerKernel(op)
                    D = Inti.IntegralOperator(dG, Γ_quad)
                    D0 = Inti.assemble_matrix(D)
                    e0 = norm(S0 * γ₁u - D0 * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                    # @test d > meshsize
                    maxdist = Inti.farfield_distance(S; tol = atol)
                    δS = Inti.adaptive_correction(S; maxdist, tol = atol)
                    maxdist = Inti.farfield_distance(D; tol = atol)
                    δD = Inti.adaptive_correction(D; maxdist, tol = atol)
                    Smat, Dmat = S0 + δS, D0 + δD
                    e1 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                    @testset "Single/double layer $(string(op))" begin
                        @test norm(e0, Inf) > 10 * norm(e1, Inf)
                        @test norm(e1, Inf) < 10 * atol
                    end
                end
            end
        end
    end
end
