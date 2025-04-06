using Inti
using Gmsh
using LaTeXStrings

function gmsh_disk(; center, rx, ry, meshsize, order)
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

function Lpower(e, i, times=nothing)
    if i == 0
        return ""
    elseif i == 1
        res = L"{%$e}"
    else
        res = L"{%$e}^{%$i}"
    end
    if times == :x
        res = string(res, L"\times")
    end
    return res
end