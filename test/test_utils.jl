using Inti
using Gmsh
const gmsh_ext = Inti.get_gmsh_extension()

function gmsh_disk(;center,rx,ry,meshsize)
    try
        gmsh.initialize()
        gmsh.model.add("disk")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax",meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin",meshsize)
        gmsh.model.occ.addDisk(center[1],center[2],0,rx,ry)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        Ω   = gmsh_ext.import_domain(;dim=2)
        msh = gmsh_ext.import_mesh(Ω;dim=2)
        return Ω,msh
    finally
        gmsh.finalize()
    end
end

function gmsh_ball(;center,radius,meshsize)
    try
        gmsh.initialize()
        gmsh.model.add("ball")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax",meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin",meshsize)
        gmsh.model.occ.addSphere(center[1],center[2],center[3],radius)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        Ω   = gmsh_ext.import_domain(;dim=3)
        msh = gmsh_ext.import_mesh(Ω;dim=3)
        return Ω,msh
    finally
        gmsh.finalize()
    end
end
