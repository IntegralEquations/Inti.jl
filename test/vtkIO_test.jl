using Test
using Gmsh
using Inti
using WriteVTK

@test begin
# This test should simply not throw an error
# TODO Replace the gmsh code with reading from a vtk file; currently tests writing
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("Sphere")
    gmsh.model.occ.addSphere(0,0,0,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    ents = gmsh.model.getEntities()
    Ω = Inti.gmsh_import_domain(;dim=3)
    M = Inti.gmsh_import_mesh(Ω;dim=3)
    vtk_save(Inti.vtk_mesh_file(M, joinpath(Inti.PROJECT_ROOT, "test", "ball")))
    rm(joinpath(Inti.PROJECT_ROOT, "test", "ball.vtu"))
    vtk_save(Inti.vtk_mesh_file(M, Ω, joinpath(Inti.PROJECT_ROOT, "test", "ball")))
    rm(joinpath(Inti.PROJECT_ROOT, "test", "ball.vtu"))
    vtk_save(Inti.vtk_mesh_file(M,
                               Inti.external_boundary(Ω),
                               joinpath(Inti.PROJECT_ROOT, "test", "sphere")))
    rm(joinpath(@__DIR__, "sphere.vtu"))
    gmsh.finalize()
    true == true
end