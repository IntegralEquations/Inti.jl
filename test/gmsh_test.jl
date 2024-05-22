using Inti
using Gmsh
using Test

# for now that check that the code below does not error
@test begin
    Inti.clear_entities!()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
    gmsh.model.add("Disk")
    gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    Ω, msh = Inti.import_mesh_from_gmsh_model(; dim = 2)
    gmsh.finalize()
    true == true
end

@test begin
    Inti.clear_entities!()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
    gmsh.model.add("Sphere")
    gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    ents = gmsh.model.getEntities()
    Ω, msh = Inti.import_mesh_from_gmsh_model(; dim = 3)
    gmsh.finalize()
    true == true
end
