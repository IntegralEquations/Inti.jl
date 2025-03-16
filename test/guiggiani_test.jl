using Test
using Inti
using StaticArrays
using Gmsh
using LinearAlgebra

@testset "Laurent coefficients" begin
    f = ρ -> ρ^2 + 2ρ + 1
    f₋₂, f₋₁, f₀ = @inferred Inti.laurent_coefficients(f, 1e-2)
    @test norm(f₋₂) < 1e-10
    @test norm(f₋₁) < 1e-10
    @test norm(f₀ - 1) < 1e-10
    f = ρ -> cos(ρ) / ρ^2 + exp(ρ) / ρ + exp(ρ)
    f₋₂, f₋₁, f₀ = Inti.laurent_coefficients(
        f,
        1e-1,
        Val(2);
        atol = 1e-20,
        breaktol = 2,
        contract = 1 / 4,
    )
    @test f₋₂ ≈ 1.0
    @test f₋₁ ≈ 1.0
    @test f₀ ≈ 1.5

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

@testset "Polar decomposition" begin
    # FIXME: write a test for the polar decomposition. E.g. test that it a quadrature in
    # rho/theta correctly integrates all some function over the square.
    ref = Inti.ReferenceSquare()
    x̂  = SVector(0.2, 0.5)

    # quad_rho   = Inti.GaussLegendre(; order = 10)
    # quad_theta = Inti.GaussLegendre(; order = 10)
    # fig        = Figure()
    # ax         = Axis(fig[1, 1])
    # for (θmin, θmax, ρ) in Inti.polar_decomposition(ref, x̂)
    #     @test θmin ≤ θmax
    #     x = SVector{2,Float64}[]
    #     for θ in θmin:0.1:θmax
    #         ρmax = ρ(θ)
    #         @show θ, ρmax
    #         for ρ in 0:0.1:ρmax
    #             push!(x, x̂ + ρ * SVector(cos(θ), sin(θ)))
    #         end
    #     end
    #     scatter!(ax, x; label = "")
    # end
    # fig
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

@testset "Green's identity" begin
    ## create a mesh and quadrature
    meshsize = 0.4
    gmsh.initialize(String[], false)
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()
    msh = Inti.import_mesh(; dim = 3)
    gmsh.finalize()
    Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
    Γ_msh = msh[Γ]
    Γ_quad = Inti.Quadrature(Γ_msh; qorder = 3)

    ##
    op = Inti.Laplace(; dim = 3)
    K = Inti.HyperSingularKernel(op)
    T = Inti.IntegralOperator(K, Γ_quad, Γ_quad)
    T₀ = Inti.assemble_matrix(T)
    δT = Inti.local_correction(T; nearfield_distance = 4 * meshsize, nearfield_qorder = 40)

    Tnew = T₀ + δT
    rhs = ones(Float64, size(T, 1))
    @test norm(Tnew * rhs, Inf) < 1e-2

    ##
    op = Inti.Elastostatic(; dim = 3, μ = 1, λ = 1)
    K = Inti.HyperSingularKernel(op)
    T = Inti.IntegralOperator(K, Γ_quad, Γ_quad)
    T₀ = Inti.assemble_matrix(T)
    δT = Inti.local_correction(T; nearfield_distance = 3 * meshsize, nearfield_qorder = 40)
    Tnew = T₀ + δT
    rhs = [SVector(1.0, 1.0, 1.0) for _ in 1:size(T, 1)]
    @test norm(Tnew * rhs, Inf) < 1e-2
end
