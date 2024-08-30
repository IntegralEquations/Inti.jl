using Inti
using Gmsh

function gmsh_disk(; center, rx, ry, meshsize, order = 1)
    msh = try
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("disk")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addDisk(center[1], center[2], 0, rx, ry)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        Inti.import_mesh(; dim = 2)
    finally
        gmsh.finalize()
    end
    Ω = Inti.Domain(Inti.entities(msh)) do e
        return Inti.geometric_dimension(e) == 2
    end
    return Ω, msh
end

function gmsh_disks(disks; meshsize, order = 1)
    msh = try
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("disk")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        for (center, rx, ry) in disks
            gmsh.model.occ.addDisk(center[1], center[2], 0, rx, ry)
        end
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        Inti.import_mesh(; dim = 2)
    finally
        gmsh.finalize()
    end
    Ω = Inti.Domain(Inti.entities(msh)) do e
        return Inti.geometric_dimension(e) == 2
    end
    return Ω, msh
end

function gmsh_ball(; center, radius, meshsize)
    msh = try
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("ball")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addSphere(center[1], center[2], center[3], radius)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    Ω = Inti.Domain(Inti.entities(msh)) do e
        return Inti.geometric_dimension(e) == 3
    end
    return Ω, msh
end
