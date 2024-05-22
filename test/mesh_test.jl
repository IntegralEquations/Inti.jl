using Test
using LinearAlgebra
using Gmsh
using Inti
using StaticArrays

@testset "Native mesh generation" begin
    Inti.clear_entities!()
    r = rx = ry = 0.5
    d1 = Inti.HyperRectangle(SVector(0.0), SVector(0.5))
    d2 = Inti.HyperRectangle(SVector(0.5), SVector(1.0))
    parametrization = let r = r
        (x) -> SVector(r * cos(2π * x[1]), r * sin(2π * x[1]))
    end
    arc1 = Inti.GeometricEntity(; domain = d1, parametrization)
    arc2 = Inti.GeometricEntity(; domain = d2, parametrization)
    Γ = Inti.Domain(Inti.key.(Set([arc1, arc2])))
    msh = Inti.meshgen(Γ, (100,))
    quad = Inti.Quadrature(msh; qorder = 2)
    @test Inti.integrate(x -> 1, quad) ≈ 2 * π * r
end

@testset "Gmsh backend" begin
    r = rx = ry = 0.5
    Inti.clear_entities!()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.005)
    Inti.clear_entities!()
    gmsh.model.occ.addDisk(0, 0, 0, rx, ry)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    Ω, M = Inti.import_mesh_from_gmsh_model(; dim = 2)
    gmsh.finalize()
    Γ = Inti.external_boundary(Ω)
    dict = Inti.topological_neighbors(M[Γ])
    _, nei = first(dict)
    dict = Inti.topological_neighbors(M[Ω])
end
