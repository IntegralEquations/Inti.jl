using Test
using Gmsh
using Inti
using WriteVTK

@test begin
    # This test should simply not throw an error
    # TODO Replace the gmsh code with reading from a vtk file; currently tests
    # writing
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("Sphere")
    gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    ents = gmsh.model.getEntities()
    Ω = Inti.gmsh_import_domain(; dim = 3)
    M = Inti.import_mesh_from_gmsh_model(Ω; dim = 3)
    fname = joinpath(Inti.PROJECT_ROOT, "test", "ball")
    vtk = vtk_grid(fname, M)
    vtk_save(vtk)
    rm(joinpath(Inti.PROJECT_ROOT, "test", "ball.vtu"))
    vtk_save(vtk_grid(joinpath(Inti.PROJECT_ROOT, "test", "ball"), M, Ω))
    rm(joinpath(Inti.PROJECT_ROOT, "test", "ball.vtu"))
    vtk_save(
        vtk_grid(
            joinpath(Inti.PROJECT_ROOT, "test", "sphere"),
            M,
            Inti.external_boundary(Ω),
        ),
    )
    rm(joinpath(@__DIR__, "sphere.vtu"))
    gmsh.finalize()
    true == true
end
