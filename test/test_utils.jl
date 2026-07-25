using Inti
using Gmsh

function gmsh_disk(; center, rx, ry, meshsize, order = 1)
    msh = try
        gmsh.initialize(String[], false)
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
        gmsh.initialize(String[], false)
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

function gmsh_torus(; center, r1, r2, meshsize)
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("ball")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addTorus(center[1], center[2], center[3], r1, r2)
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

function gmsh_cut_ball(; center, radius, meshsize, cutelevation)
    msh = try
        xmin = -1.1 * radius
        ymin = -1.1 * radius
        xmax = 1.1 * radius
        ymax = 1.1 * radius
        zmin = -(1 - cutelevation) * radius
        zmax = (1 - cutelevation) * radius
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("ball")
        # Construct the intersecting box
        p1 = gmsh.model.occ.addPoint(xmin, ymax, zmin)
        p2 = gmsh.model.occ.addPoint(xmax, ymax, zmin)
        p3 = gmsh.model.occ.addPoint(xmax, ymin, zmin)
        p4 = gmsh.model.occ.addPoint(xmin, ymin, zmin)
        p5 = gmsh.model.occ.addPoint(xmin, ymax, zmax)
        p6 = gmsh.model.occ.addPoint(xmax, ymax, zmax)
        p7 = gmsh.model.occ.addPoint(xmax, ymin, zmax)
        p8 = gmsh.model.occ.addPoint(xmin, ymin, zmax)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        l5 = gmsh.model.occ.addLine(p5, p6)
        l6 = gmsh.model.occ.addLine(p6, p7)
        l7 = gmsh.model.occ.addLine(p7, p8)
        l8 = gmsh.model.occ.addLine(p8, p5)
        l9 = gmsh.model.occ.addLine(p1, p5)
        l10 = gmsh.model.occ.addLine(p2, p6)
        l11 = gmsh.model.occ.addLine(p3, p7)
        l12 = gmsh.model.occ.addLine(p4, p8)
        cl1 = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        cl2 = gmsh.model.occ.addCurveLoop([l5, l6, l7, l8])
        cl3 = gmsh.model.occ.addCurveLoop([l1, l10, l5, l9])
        cl4 = gmsh.model.occ.addCurveLoop([l3, l11, l7, l12])
        cl5 = gmsh.model.occ.addCurveLoop([l10, l2, l11, l6])
        cl6 = gmsh.model.occ.addCurveLoop([l4, l9, l8, l12])
        f1 = gmsh.model.occ.addPlaneSurface([cl1])
        f2 = gmsh.model.occ.addPlaneSurface([cl2])
        f3 = gmsh.model.occ.addPlaneSurface([cl3])
        f4 = gmsh.model.occ.addPlaneSurface([cl4])
        f5 = gmsh.model.occ.addPlaneSurface([cl5])
        f6 = gmsh.model.occ.addPlaneSurface([cl6])
        loo = gmsh.model.occ.addSurfaceLoop([f1, f2, f3, f4, f5, f6])
        vol = gmsh.model.occ.addVolume([loo])
        sph = gmsh.model.occ.addSphere(center[1], center[2], center[3], radius)
        gmsh.model.occ.intersect([3, vol], [3, sph])

        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
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
