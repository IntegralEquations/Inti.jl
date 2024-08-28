using Test
using LinearAlgebra
using Gmsh
using StaticArrays
using Inti

@testset "Gmsh backend" begin
    @testset "Area/volume" begin
        @testset "Cube" begin
            # generate a mesh
            (lx, ly, lz) = widths = (1.0, 1.0, 2.0)
            Inti.clear_entities!()
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 2)
            gmsh.model.occ.addBox(0, 0, 0, lx, ly, lz)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)
            M = Inti.import_mesh(; dim = 3)
            gmsh.finalize()
            Ω = Inti.Domain(Inti.entities(M)) do e
                return Inti.geometric_dimension(e) == 3
            end
            Γ = Inti.external_boundary(Ω)
            Γ_quad = Inti.Quadrature(view(M, Γ); qorder = 1)
            A = 2 * (lx * ly + lx * lz + ly * lz)
            @test A ≈ Inti.integrate(x -> 1, Γ_quad)
            #
            Γ_quad = Inti.Quadrature(M[Γ]; qorder = 1)
            A = 2 * (lx * ly + lx * lz + ly * lz)
            @test A ≈ Inti.integrate(x -> 1, Γ_quad)
            # generate a Nystrom mesh for volume
            Ω_quad = Inti.Quadrature(M[Ω]; qorder = 1)
            V = prod(widths)
            # sum only weights corresponding to tetras
            @test V ≈ Inti.integrate(x -> 1, Ω_quad)
        end
        @testset "Ball" begin
            r = 0.5
            Inti.clear_entities!()
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 2)
            # meshsize
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
            gmsh.model.occ.addSphere(0, 0, 0, r)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(1)
            M = Inti.import_mesh(; dim = 3)
            gmsh.finalize()
            f = x -> x[1] + x[2] - 2 * x[3] + 1
            Ω = Inti.Domain(Inti.entities(M)) do e
                return Inti.geometric_dimension(e) == 3
            end
            Γ = Inti.external_boundary(Ω)
            quad = Inti.Quadrature(M[Γ]; qorder = 4) # NystromMesh of surface Γ
            area = Inti.integrate(x -> 1, quad)
            @test isapprox(area, 4 * π * r^2, rtol = 5e-2)
            exact = map(f, M[Γ].nodes)
            approx = Inti.quadrature_to_node_vals(quad, map(q -> f(q.coords), quad))
            @test exact ≈ approx
            quad = Inti.Quadrature(M[Ω]; qorder = 4) # Nystrom mesh of volume Ω
            volume = Inti.integrate(x -> 1, quad)
            @test isapprox(volume, 4 / 3 * π * r^3, atol = 1e-2)
            exact = map(f, M[Ω].nodes)
            approx = Inti.quadrature_to_node_vals(quad, map(q -> f(q.coords), quad))
            @test exact ≈ approx
        end
        @testset "Sphere with P2" begin
            r = 0.5
            Inti.clear_entities!()
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 2)
            # meshsize
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.025)
            gmsh.model.occ.addSphere(0, 0, 0, r)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(2)
            M = Inti.import_mesh(; dim = 3)
            gmsh.finalize()
            f = x -> x[1] + x[2] - 2 * x[3] + 1
            Ω = Inti.Domain(Inti.entities(M)) do e
                return Inti.geometric_dimension(e) == 3
            end
            Γ = Inti.external_boundary(Ω)
            quad = Inti.Quadrature(M[Γ]; qorder = 4) # NystromMesh of surface Γ
            area = Inti.integrate(x -> 1, quad)
            mean_curv = Inti.mean_curvature(quad)
            κ = -1 / r
            @test all(x -> norm(x - κ) < 0.01, mean_curv)
            gauss_curv = Inti.gauss_curvature(quad)
            κ = 1 / r^2
            @test all(x -> norm(x - κ) < 0.05, gauss_curv)
            @test isapprox(area, 4 * π * r^2, atol = 1e-3)
            exact = map(f, M[Γ].nodes)
            approx = Inti.quadrature_to_node_vals(quad, map(q -> f(q.coords), quad))
            @test exact ≈ approx
        end
        @testset "Circle" begin
            r = rx = ry = 0.5
            Inti.clear_entities!()
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 2)
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
            Inti.clear_entities!()
            gmsh.model.occ.addDisk(0, 0, 0, rx, ry)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            M = Inti.import_mesh(; dim = 2)
            gmsh.finalize()
            f = x -> x[1] + x[2] - 3
            Ω = Inti.Domain(Inti.entities(M)) do e
                return Inti.geometric_dimension(e) == 2
            end
            Γ = Inti.external_boundary(Ω)
            quad = Inti.Quadrature(M[Ω]; qorder = 2)
            A = π * r^2
            # test area
            @test isapprox(A, Inti.integrate(x -> 1, quad); atol = 1e-2)
            exact = map(f, M[Ω].nodes)
            approx = Inti.quadrature_to_node_vals(quad, map(q -> f(q.coords), quad))
            @test exact ≈ approx
            # test perimeter
            quad = Inti.Quadrature(M[Γ]; qorder = 2)
            exact = map(f, quad.mesh.nodes)
            approx = Inti.quadrature_to_node_vals(quad, map(q -> f(q.coords), quad))
            @test exact ≈ approx
            P = 2π * r
            @test isapprox(P, Inti.integrate(x -> 1, quad); atol = 1e-2)
        end
        @testset "Parametric circle" begin
            geo = Inti.parametric_curve(θ -> SVector(cos(θ), sin(θ)), 0, 2π)
            Γ = Inti.Domain(geo)
            Q = Inti.Quadrature(Γ; qorder = 2, meshsize = 0.1)
            @test Inti.integrate(x -> 1, Q) ≈ 2π
        end
    end
end
