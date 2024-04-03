using Test
using LinearAlgebra
using StaticArrays
using Gmsh
using Inti

@testset "Gmsh backend" begin
    @testset "Area/volume" begin
        @testset "Cube" begin
            # generate a mesh
            (lx, ly, lz) = widths = (1.0, 1.0, 2.0)
            Inti.clear_entities!()
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 2)
            Inti.clear_entities!()
            gmsh.model.occ.addBox(0, 0, 0, lx, ly, lz)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)
            Ω, M = Inti.import_mesh_from_gmsh_model(; dim = 3)
            gmsh.finalize()
            # lazy
            Γ = Inti.external_boundary(Ω)
            Γ_quad = Inti.Quadrature(view(M, Γ); qorder = 1)
            A = 2 * (lx * ly + lx * lz + ly * lz)
            @test A ≈ Inti.integrate(x -> 1, Γ_quad)
            #
            Γ_quad = Inti.Quadrature(M[Γ]; qorder = 5)
            A = 2 * (lx * ly + lx * lz + ly * lz)
            @test A ≈ Inti.integrate(x -> 1, Γ_quad)
            d = Inti.farfield_distance(Γ_quad, K, 1e-5)
            # generate a Nystrom mesh for volume
            Ω_quad = Inti.Quadrature(M[Ω]; qorder = 1)
            V = prod(widths)
            # sum only weights corresponding to tetras
            @test V ≈ Inti.integrate(x -> 1, Ω_quad)
        end
        @testset "Sphere" begin
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
            Ω, M = Inti.import_mesh_from_gmsh_model(; dim = 3)
            gmsh.finalize()
            f = x -> x[1] + x[2] - 2 * x[3] + 1
            Γ = Inti.external_boundary(Ω)
            quad = Inti.Quadrature(M[Γ]; qorder = 4) # NystromMesh of surface Γ
            area = Inti.integrate(x -> 1, quad)
            @test isapprox(area, 4 * π * r^2, atol = 5e-2)
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
        @testset "Circle" begin
            r = rx = ry = 0.5
            Inti.clear_entities!()
            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 2)
            Inti.clear_entities!()
            gmsh.model.occ.addDisk(0, 0, 0, rx, ry)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            Ω, M = Inti.import_mesh_from_gmsh_model(; dim = 2)
            gmsh.finalize()
            f = x -> x[1] + x[2] - 3
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
    end
end

@testset "farfield distance" begin
    pde = Inti.Laplace(; dim = 2)
    K = Inti.SingleLayerKernel(pde)
    p1, p2, p3 = SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0)
    h = 1e-1
    el = Inti.LagrangeTriangle(h.* (p1, p2, p3))
    qrule = Inti.VioreanuRokhlin(; domain = :triangle, order = 4)
    maxiter = 10
    d = Inti._farfield_distance(el, K, qrule, 1e-10, maxiter)
    @test d > h
end
