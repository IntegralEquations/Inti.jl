using Inti
using Gmsh
using Test

# for now that check that the code below does not error
@test begin
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("Disk")
    gmsh.model.occ.addDisk(0,0,0,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    Ω   = Inti.gmsh_import_domain(;dim=2)
    msh = Inti.gmsh_import_mesh(Ω;dim=2)
    gmsh.finalize()
    return true
end

@test begin
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("Sphere")
    gmsh.model.occ.addSphere(0,0,0,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    ents = gmsh.model.getEntities()
    Ω   = Inti.gmsh_import_domain(;dim=3)
    msh = Inti.gmsh_import_mesh(Ω;dim=3)
    gmsh.finalize()
    return true
end
