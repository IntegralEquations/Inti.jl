using Test
using LinearAlgebra
using StaticArrays
using Inti

@testset "2D circle, linear function" begin
    Inti.clear_entities!()
    Γ = Inti.parametric_curve(t -> SVector(cos(t), sin(t)), 0, 2π) |> Inti.Domain
    quad = Inti.Quadrature(Γ; meshsize = 0.1, qorder = 5)
    # f(x,y) = x on unit circle
    u = map(q -> q.coords[1], quad)
    grad = Inti.surface_gradient(u, quad)
    # analytic: ∇_Γ(x) = (y², -xy) on unit circle
    grad_exact = map(quad) do q
        x, y = q.coords
        SVector(y^2, -y * x)
    end
    @test norm(grad .- grad_exact, Inf) < 1.0e-2
end

@testset "3D ellipsoid, linear function" begin
    Inti.clear_entities!()
    Ω = Inti.GeometricEntity("ellipsoid") |> Inti.Domain
    Γ = Inti.external_boundary(Ω)
    quad = Inti.Quadrature(Γ; meshsize = 0.2, qorder = 4)
    # f(x) = a ⋅ x with known a
    a = SVector(1.0, 0.5, -0.3)
    u = map(q -> dot(a, q.coords), quad)
    grad = Inti.surface_gradient(u, quad)
    # analytic: ∇_Γ f = a - (a⋅n̂)n̂
    grad_exact = map(quad) do q
        n̂ = q.normal
        a - dot(a, n̂) * n̂
    end
    err = maximum(norm(gc - ga) for (gc, ga) in zip(grad, grad_exact))
    @test err < 5.0e-2
end

@testset "matrix vs direct consistency" begin
    Inti.clear_entities!()
    Γ = Inti.parametric_curve(t -> SVector(cos(t), sin(t)), 0, 2π) |> Inti.Domain
    quad = Inti.Quadrature(Γ; meshsize = 0.1, qorder = 3)
    u = map(q -> sin(q.coords[1]), quad)
    G = Inti.tangential_gradient_matrix(quad)
    grad1 = G * u
    grad2 = Inti.surface_gradient(u, quad)
    @test grad1 == grad2
end

@testset "convergence under refinement (2D circle)" begin
    # f(x,y) = x² on unit circle
    # ∇_Γ f = (2x - 2x³, -2x²y)
    errs = Float64[]
    for h in [0.4, 0.2, 0.1, 0.05]
        Inti.clear_entities!()
        Γ = Inti.parametric_curve(t -> SVector(cos(t), sin(t)), 0, 2π) |> Inti.Domain
        quad = Inti.Quadrature(Γ; meshsize = h, qorder = 5)
        u = map(q -> q.coords[1]^2, quad)
        grad = Inti.surface_gradient(u, quad)
        grad_exact = map(quad) do q
            x, y = q.coords
            SVector(2x - 2x^3, -2x^2 * y)
        end
        push!(errs, maximum(norm(gc - ga) for (gc, ga) in zip(grad, grad_exact)))
    end
    # check monotonic convergence
    ratios = errs[1:(end - 1)] ./ errs[2:end]
    @test all(r > 1.5 for r in ratios)
end

@testset "surface divergence (3D torus)" begin
    Inti.clear_entities!()
    Ω = Inti.torus(R = 2.0, r = 0.5) |> Inti.Domain
    Γ = Inti.external_boundary(Ω)
    quad = Inti.Quadrature(Γ; meshsize = 0.4, qorder = 4)

    # Vector field v(x) = x. ∇_Γ ⋅ x = 2
    v = map(q -> q.coords, quad)
    div_v = Inti.surface_divergence(v, quad)

    div_exact = fill(2.0, length(quad))
    err = norm(div_v .- div_exact, Inf)
    @test err < 5.0e-2
end

@testset "surface laplacian (3D torus)" begin
    Inti.clear_entities!()
    Ω = Inti.torus(R = 2.0, r = 0.5) |> Inti.Domain
    Γ = Inti.external_boundary(Ω)

    # Use a finer mesh and higher order to demonstrate convergence
    quad = Inti.Quadrature(Γ; meshsize = 0.2, qorder = 8)

    # u(x) = x[1] (linear coordinate)
    # Δ_Γ u = 2 * mean_curvature * n_1
    u = map(q -> q.coords[1], quad)
    lap_u = Inti.surface_laplacian(u, quad)

    curv = Inti.mean_curvature(quad)
    lap_exact = map(quad, curv) do q, H
        2.0 * H * q.normal[1]
    end

    err = norm(lap_u .- lap_exact, Inf)
    @test err < 5.0e-3
end

@testset "surface laplacian (3D ellipsoid)" begin
    Inti.clear_entities!()
    Ω = Inti.GeometricEntity("ellipsoid") |> Inti.Domain
    Γ = Inti.external_boundary(Ω)

    # The ellipsoid parametrisation is non-polynomial (cubed-sphere).
    # We must use a high quadrature order to correctly interpolate the mapping Hessian.
    quad = Inti.Quadrature(Γ; meshsize = 0.2, qorder = 8)

    u = map(q -> q.coords[1], quad)
    lap_u = Inti.surface_laplacian(u, quad)

    curv = Inti.mean_curvature(quad)
    lap_exact = map(quad, curv) do q, H
        2.0 * H * q.normal[1]
    end

    err = norm(lap_u .- lap_exact, Inf)
    @test err < 2.0e-2
end
