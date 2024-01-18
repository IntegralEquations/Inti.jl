using Inti
using Gmsh

function gmsh_disk(; center, rx, ry, meshsize)
    try
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("disk")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addDisk(center[1], center[2], 0, rx, ry)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        return Inti.import_mesh_from_gmsh_model(; dim = 2)
    finally
        gmsh.finalize()
    end
end

function gmsh_ball(; center, radius, meshsize)
    try
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("ball")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addSphere(center[1], center[2], center[3], radius)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        return Inti.import_mesh_from_gmsh_model(; dim = 3)
    finally
        gmsh.finalize()
    end
end
