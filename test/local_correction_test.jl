using Test
using LinearAlgebra
using Inti
using Random
using StaticArrays

include("test_utils.jl")

Random.seed!(1)
atol = 1e-4
@testset "Boundary integral operators" begin
    for N in (2, 3)
        # create geometry
        Inti.clear_entities!()
        meshsize = N == 2 ? 0.2 : 0.4
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
                Inti.Stokes(; μ = 1.2, dim = N),
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
                    maxdist = 2 * meshsize
                    # single and double layer
                    @testset "Single/double layer $(string(op))" begin
                        G = Inti.SingleLayerKernel(op)
                        S = Inti.IntegralOperator(G, Γ_quad)
                        S0 = Inti.assemble_matrix(S)
                        dG = Inti.DoubleLayerKernel(op)
                        D = Inti.IntegralOperator(dG, Γ_quad)
                        D0 = Inti.assemble_matrix(D)
                        e0 = norm(S0 * γ₁u - D0 * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                        δS = Inti.local_correction(S; maxdist, tol = atol)
                        δD = Inti.local_correction(D; maxdist, tol = atol)
                        Smat, Dmat = S0 + δS, D0 + δD
                        e1 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                        @test norm(e0, Inf) > 10 * norm(e1, Inf)
                        @test norm(e1, Inf) < 10 * atol
                    end
                    @testset "Adjoint double-layer/hypersingular $(string(op))" begin
                        op isa Inti.Stokes && continue # TODO: implement hypersingular for Stokes?
                        K = Inti.IntegralOperator(Inti.AdjointDoubleLayerKernel(op), Γ_quad)
                        Kmat = Inti.assemble_matrix(K)
                        H = Inti.IntegralOperator(Inti.HyperSingularKernel(op), Γ_quad)
                        Hmat = Inti.assemble_matrix(H)
                        e0 = norm(Kmat * γ₁u - Hmat * γ₀u - σ * γ₁u, Inf) / γ₁u_norm
                        δK = Inti.local_correction(K; maxdist, tol = atol)
                        δH = Inti.local_correction(H; maxdist, tol = atol)
                        Kmat, Hmat = Kmat + δK, Hmat + δH
                        e1 = norm(Kmat * γ₁u - Hmat * γ₀u - σ * γ₁u, Inf) / γ₁u_norm
                    end
                end
            end
        end
    end
end
