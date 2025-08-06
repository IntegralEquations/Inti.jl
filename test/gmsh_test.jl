using Inti
using Gmsh
using Test

@testset "Circle area and perimeter" begin
    for order in [1, 2, 3]
        Inti.clear_entities!()
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
        gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.set_order(order)
        msh = Inti.import_mesh(; dim = 2)
        Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
        Γ = Inti.boundary(Ω)
        gmsh.finalize()
        vol_quad = Inti.Quadrature(msh[Ω]; qorder = 3)
        bnd_quad = Inti.Quadrature(msh[Γ]; qorder = 3)
        @test abs(Inti.integrate(x -> 1, vol_quad) - π) < 1e-2
        @test abs(Inti.integrate(x -> 1, bnd_quad) - 2π) < 1e-2
    end
end

@testset "Square area and perimeter" begin
    for order in [1, 2, 3]
        Inti.clear_entities!()
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
        gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.set_order(order)
        msh = Inti.import_mesh(; dim = 2)
        Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
        Γ = Inti.boundary(Ω)
        gmsh.finalize()
        vol_quad = Inti.Quadrature(msh[Ω]; qorder = 3)
        bnd_quad = Inti.Quadrature(msh[Γ]; qorder = 3)
        @test Inti.integrate(x -> 1, vol_quad) ≈ 4
        @test Inti.integrate(x -> 1, bnd_quad) ≈ 8
    end
end

@testset "Sphere area and volume" begin
    for order in [2, 3]
        Inti.clear_entities!()
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
        gmsh.model.occ.addSphere(0, 0, 0, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.set_order(order)
        msh = Inti.import_mesh(; dim = 3)
        gmsh.finalize()
        Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, msh)
        Γ = Inti.boundary(Ω)
        vol_quad = Inti.Quadrature(msh[Ω]; qorder = 3)
        bnd_quad = Inti.Quadrature(msh[Γ]; qorder = 3)
        @test abs(Inti.integrate(x -> 1, vol_quad) - 4 / 3 * π) < 1e-2
        @test abs(Inti.integrate(x -> 1, bnd_quad) - 4 * π) < 1e-2
    end
end

@testset "Cube area and volume" begin
    for order in [1, 2, 3]
        Inti.clear_entities!()
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
        gmsh.model.occ.addBox(-1, -1, -1, 2, 2, 2)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.set_order(order)
        msh = Inti.import_mesh(; dim = 3)
        gmsh.finalize()
        Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, msh)
        Γ = Inti.boundary(Ω)
        vol_quad = Inti.Quadrature(msh[Ω]; qorder = 3)
        bnd_quad = Inti.Quadrature(msh[Γ]; qorder = 3)
        @test Inti.integrate(x -> 1, vol_quad) ≈ 8
        @test Inti.integrate(x -> 1, bnd_quad) ≈ 24
    end
end
