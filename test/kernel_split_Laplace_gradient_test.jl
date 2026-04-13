using Inti
using Gmsh
using HMatrices
using IterativeSolvers
using LinearAlgebra
using StaticArrays
using Test

function build_curved_unit_disk_setup(; meshsize, qorder, n_quad_pts, smoothness)
    f_circle = s -> SVector(cos(2π * s), sin(2π * s))

    msh = let
        gmsh.initialize()
        try
            gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
            gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)

            bnd1 = Inti.gmsh_curve(f_circle, 0, 1; meshsize)
            cl = gmsh.model.occ.addCurveLoop([bnd1])
            gmsh.model.occ.addPlaneSurface([cl])
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            Inti.import_mesh(; dim = 2)
        finally
            gmsh.finalize()
        end
    end

    Ω = Inti.Domain(Inti.entities(msh)) do ent
        Inti.geometric_dimension(ent) == 2
    end
    Γ = Inti.external_boundary(Ω)

    # Geometry tests should use the curved mesh rather than the straight gmsh import.
    crvmsh = Inti.curve_mesh(msh, f_circle, smoothness)
    Ω_quad = Inti.Quadrature(crvmsh[Ω]; qorder)
    Γ_quad = Inti.Quadrature(crvmsh[Γ], Inti.GaussLegendre(n_quad_pts))
    Γ_elements = collect(Inti.elements(crvmsh[Γ]))
    Γ_connectivity = Inti.etype2qtags(Γ_quad, first(Inti.element_types(crvmsh[Γ])))

    return (;
        Ω_quad,
        Γ_quad,
        Γ_elements,
        Γ_connectivity,
    )
end

function exterior_point_source_solution(source_point)
    u = x -> -1 / (2π) * log(norm(x - source_point))
    grad_u = x -> begin
        r = x - source_point
        return -1 / (2π) * r / dot(r, r)
    end
    return u, grad_u
end

@testset "Laplace kernel-split double-layer gradient" begin
    meshsize = 0.2
    qorder = 12
    n_quad_pts = 16
    smoothness = 5
    hmatrix_tol = 1.0e-12
    gmres_tol = 1.0e-14

    op = Inti.Laplace(; dim = 2)
    angle_mod = x -> mod(angle(x), 2π)
    velocity = s -> 2π * (-sin(2π * s) + im * cos(2π * s))
    curvature = s -> 1.0
    boundary_inv = x -> angle_mod(x[1] + im * x[2]) / (2π)

    setup = build_curved_unit_disk_setup(;
        meshsize,
        qorder,
        n_quad_pts,
        smoothness,
    )
    @test isapprox(Inti.integrate(x -> 1.0, setup.Ω_quad), π; atol = 1.0e-12)

    source_point = SVector(1.5, 0.0)
    u_exact, grad_u_exact = exterior_point_source_solution(source_point)

    geometry_data = (
        connectivity = setup.Γ_connectivity,
        elements = setup.Γ_elements,
        velocity_fn = velocity,
        curvature_fn = curvature,
        boundary_inv = boundary_inv,
        parametric_length = 1.0,
    )

    boundary_data = map(q -> u_exact(Inti.coords(q)), setup.Γ_quad)
    gradient_exact = map(q -> grad_u_exact(Inti.coords(q)), setup.Ω_quad)

    _, D_b2b = Inti.single_double_layer(;
        op,
        target = setup.Γ_quad,
        source = setup.Γ_quad,
        compression = (method = :hmatrix, tol = hmatrix_tol),
        correction = merge(geometry_data, (method = :ksplit, n_panel_corr = 3)),
    )

    σ = gmres(-I / 2 + D_b2b, boundary_data; reltol = gmres_tol, abstol = gmres_tol, restart = 1000)

    Dx_b2d, Dy_b2d = Inti.double_layer_gradient(;
        op,
        target = setup.Ω_quad,
        source = setup.Γ_quad,
        compression = (method = :hmatrix, tol = hmatrix_tol),
        correction = merge(
            geometry_data,
            (
                method = :ksplit,
                maxdist = 1.1 * meshsize,
                target_location = :inside,
            ),
        ),
    )

    gradient_x = Dx_b2d * σ
    gradient_y = Dy_b2d * σ
    gradient_num = [SVector(gradient_x[i], gradient_y[i]) for i in eachindex(setup.Ω_quad)]
    relerr =
        maximum(norm.(gradient_num .- gradient_exact)) / maximum(norm.(gradient_exact))

    @info "Relative Linf gradient error" relerr
    @test relerr < 1.0e-10
end
