using Test
using LinearAlgebra
using Inti
using Random
using StaticArrays

include("test_utils.jl")

Random.seed!(1)
atol = 1e-4

@testset "Laurent coefficients" begin
    f = ρ -> ρ^2 + 2ρ + 1
    f₋₂, f₋₁, f₀ = @inferred Inti.laurent_coefficients(f, 1e-2, Val(2))
    @test norm((f₋₂, f₋₁, f₀) .- (0, 0, 1)) < 1e-10
    f₋₂, f₋₁, f₀ = @inferred Inti.laurent_coefficients(f, 1e-2, Val(0))
    @test norm((f₋₂, f₋₁, f₀) .- (0, 0, 1)) < 1e-10
    f = ρ -> cos(ρ) / ρ^2 + exp(ρ) / ρ + exp(ρ)
    f₋₂, f₋₁, f₀ = Inti.laurent_coefficients(
        f,
        1e-0,
        Val(2);
        atol = 1e-12,
        breaktol = 2,
        contract = 1 / 2,
    )
    @test norm((f₋₂, f₋₁, f₀) .- (1, 1, 1.5)) < 1e-10

    f = ρ -> SVector(cos(ρ), sin(ρ)) / ρ^2 + SVector(exp(ρ), 0.2) / ρ
    f₋₂, f₋₁, f₀ = Inti.laurent_coefficients(f, 1e-1, Val(2))
    @test f₋₂ ≈ SVector(1.0, 0.0)
    @test f₋₁ ≈ SVector(1.0, 1.2)

    ## Laplace kernel
    v1 = SVector(0.0, 0.0, 0.0)
    v2 = SVector(1.0, 0.0, 0.0)
    v3 = SVector(1.0, 1.0, 0.0)
    v4 = SVector(0.0, 1.0, 0.0)

    el = Inti.LagrangeSquare(v1, v2, v3, v4)

    K = Inti.HyperSingularKernel(Inti.Laplace(; dim = 3))
    x̂ = SVector(0.5, 0.2)
    x = el(x̂)
    nx = Inti.normal(el, x̂)
    qx = (coords = x, normal = nx)

    F = let K = K, qx = qx, el = el, x̂ = x̂
        (ρ, θ) -> begin
            s, c = sincos(θ)
            ŷ = x̂ + ρ * SVector(c, s)
            y = el(ŷ)
            jac = Inti.jacobian(el, ŷ)
            ny = Inti._normal(jac)
            μ = Inti._integration_measure(jac)
            qy = (coords = y, normal = ny)
            ρ * K(qx, qy) * μ
            # qy = (coords = x, normal = nx)
            # ρ * K(qx, qx)
        end
    end
    g = let F = F
        (ρ) -> F(ρ, 1)
    end
    @inferred Inti.laurent_coefficients(g, 1e-3, (Val(2)))
end

@testset "Plane distorted element" begin
    # Guiggiani plane distoreted element (table 1)
    δ          = 0.5
    z          = 0.0
    y¹         = SVector(-1.0, -1.0, z)
    y²         = SVector(1.0 + δ, -1.0, z)
    y³         = SVector(1.0 - δ, 1.0, z)
    y⁴         = SVector(-1.0, 1.0, z)
    nodes      = (y¹, y², y³, y⁴)
    el         = Inti.LagrangeSquare(nodes)
    K          = (p, q) -> begin
        x = Inti.coords(p)
        y = Inti.coords(q)
        d = norm(x - y)
        1 / (d^3)
    end
    û         = (x̂) -> 1
    a          = SVector(0.5, 0.5)
    b          = SVector(1.66 / 2, 0.5)
    quad_rho   = Inti.GaussLegendre(; order = 10)
    quad_theta = Inti.GaussLegendre(; order = 20)
    va         = Inti.guiggiani_singular_integral(K, û, a, el, quad_rho, quad_theta)
    vb         = Inti.guiggiani_singular_integral(K, û, b, el, quad_rho, quad_theta)
    @test isapprox(va, -5.749237; atol = 1e-4)
    @test isapprox(vb, -9.154585; atol = 1e-4)
    # TODO: add point c and more tests from table 2
end

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
                        δS = Inti.local_correction(S)
                        δD = Inti.local_correction(D)
                        Smat, Dmat = S0 + δS, D0 + δD
                        e2 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                        @test norm(e2, Inf) < 10 * atol
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
