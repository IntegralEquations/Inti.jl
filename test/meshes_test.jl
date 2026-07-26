using Inti
using GLMakie
using Gmsh
using Test
using StaticArrays

# NOTE: this script simply checks that running the code below does not error. It
# exercises the `MeshPlot` recipe (reached via `Makie.plot`/`plot!` through
# `Makie.plottype`) and the `Makie.convert_arguments` overloads defined in the
# `IntiMakieExt` extension.

## Lines
a, b = SVector(0.0, 0.0), SVector(1.0, 1.0)
el = Inti.LagrangeLine(a, b)
plot(el)

el1 = Inti.LagrangeLine(SVector(0.0, 0.0), SVector(1.0, 1.0))
el2 = Inti.LagrangeLine(SVector(0.0, 1.0), SVector(1.0, 0.0))
plot(el1)
plot!(el2)
plot([el1, el2])

a, b = SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0)
el = Inti.LagrangeLine(a, b)
plot(el)

## Triangles
a, b, c = SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0)
el = Inti.LagrangeTriangle(a, b, c)
plot(el)
el2 = Inti.LagrangeTriangle(a .+ 2, b .+ 2, c .+ 2)
plot([el, el2])

## Quadrilaterals
a, b, c, d = SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)
el = Inti.LagrangeSquare(a, b, c, d)
plot(el)

a, b, c, d = SVector(0.0, 0.0, 0.0),
    SVector(1.0, 0.0, 0.5),
    SVector(1.0, 1.0, 1.0),
    SVector(0.0, 1.0, 1.0)
el = Inti.LagrangeSquare(a, b, c, d)
plot(el; strokewidth = 1)

# Tetrahedron
a, b, c, d = SVector(0.0, 0.0, 0.0),
    SVector(1.0, 0.0, 0.0),
    SVector(0.0, 1.0, 0.0),
    SVector(0.0, 0.0, 1.0)
el = Inti.LagrangeTetrahedron(a, b, c, d)
plot(el)

# Cube
p1, p2, p3, p4, p5, p6, p7, p8 = SVector(0.0, 0.0, 0.0),
    SVector(1.0, 0.0, 0.0),
    SVector(1.0, 1.0, 0.0),
    SVector(0.0, 1.0, 0.0),
    SVector(0.0, 0.0, 1.0),
    SVector(1.0, 0.0, 1.0),
    SVector(1.0, 1.0, 1.0),
    SVector(0.0, 1.0, 1.0)
el = Inti.LagrangeCube(p1, p2, p3, p4, p5, p6, p7, p8)
plot(el; strokewidth = 1, alpha = 0.1)

## native Makie verbs through `convert_arguments`
mesh(Inti.LagrangeTriangle(a, b, c); color = :blue)
scatter(Inti.LagrangeLine(SVector(0.0, 0.0), SVector(1.0, 1.0)))

## meshes in 2d
Inti.clear_entities!()
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 2)
gmsh.model.add("Disk")
gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
msh = Inti.import_mesh(; dim = 2)
gmsh.finalize()

Ω = Inti.Domain(Inti.entities(msh)) do ent
    return Inti.geometric_dimension(ent) == 2
end
M = view(msh, Ω)
plot(M; strokewidth = 1)
# nodal data with and without interpolation
nodevals = [sin(p[1]) for p in Inti.nodes(M)]
plot(M; color = nodevals, interpolate = true)
plot(M; color = nodevals, interpolate = false, strokewidth = 1)
M = view(msh, Inti.boundary(Ω))
plot(M; strokewidth = 1)

# meshes in 3d
Inti.clear_entities!()
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 2)
gmsh.model.add("Sphere")
gmsh.model.occ.addSphere(0, 0, 0, 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
msh = Inti.import_mesh(; dim = 3)
gmsh.finalize()

Ω = Inti.Domain(Inti.entities(msh)) do ent
    return Inti.geometric_dimension(ent) == 3
end
Γ = Inti.boundary(Ω)
plot(msh[Γ]; strokewidth = 1)
plot(msh[Ω]; strokewidth = 1, alpha = 0.5)
