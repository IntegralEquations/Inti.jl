using Inti
using Meshes
import GLMakie
using Gmsh
using Test
using StaticArrays

# NOTE: this script simply checks that running the code below does not error

## Lines
a, b = SVector(0.0, 0.0), SVector(1.0, 1.0)
el = Inti.LagrangeLine(a, b)
viz(el)

el1 = Inti.LagrangeLine(SVector(0.0, 0.0), SVector(1.0, 1.0))
el2 = Inti.LagrangeLine(SVector(0.0, 1.0), SVector(1.0, 0.0))
viz(el1)
viz!(el2)
viz([el1, el2])

a, b = SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0)
el = Inti.LagrangeLine(a, b)
viz(el)

## Triangles
a, b, c = SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0)
el = Inti.LagrangeTriangle(a, b, c)
viz(el)
el2 = Inti.LagrangeTriangle(a .+ 2, b .+ 2, c .+ 2)
viz([el, el2])

## Quadrilaterals
a, b, c, d = SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0)
el = Inti.LagrangeSquare(a, b, c, d)
viz(el)

a, b, c, d = SVector(0.0, 0.0, 0.0),
SVector(1.0, 0.0, 0.5),
SVector(1.0, 1.0, 1.0),
SVector(0.0, 1.0, 1.0)
el = Inti.LagrangeSquare(a, b, c, d)
rec = Quadrangle(Tuple.(el.vals)...)
viz(rec; showsegments = true)

# Tetrahedron
a, b, c, d = SVector(0.0, 0.0, 0.0),
SVector(1.0, 0.0, 0.0),
SVector(0.0, 1.0, 0.0),
SVector(0.0, 0.0, 1.0)
el = Inti.LagrangeTetrahedron(a, b, c, d)
viz(el)

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
viz(el; showsegments = true, alpha = 0.1)

## meshes in 2d
Inti.clear_entities!()
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 2)
gmsh.model.add("Disk")
gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
Ω, msh = Inti.import_mesh_from_gmsh_model(; dim = 2)
gmsh.finalize()

M = view(msh, Ω)
viz(M; showsegments = true)
M = view(msh, Inti.boundary(Ω))
viz(M; showsegments = true)

# meshes in 3d
Inti.clear_entities!()
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 2)
gmsh.model.add("Sphere")
gmsh.model.occ.addSphere(0, 0, 0, 1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
ents = gmsh.model.getEntities()
Ω, msh = Inti.import_mesh_from_gmsh_model(; dim = 3)
gmsh.finalize()

Γ = Inti.boundary(Ω)
viz(msh[Γ]; showsegments = true)
viz(msh[Ω]; showsegments = true)
