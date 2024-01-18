# [Geometry and meshes](@id geometry-and-meshes-section)

## Overview

## [Gmsh](@id gmsh-section)

```@example gmsh-sphere
using Inti
using Gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 2)
gmsh.model.add("Sphere")
# set mesh size
gmsh.model.occ.addSphere(0,0,0,1)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
gmsh.model.mesh.generate(3)
ents = gmsh.model.getEntities()
Ω   = Inti.gmsh_import_domain(;dim=3)
msh = Inti.import_mesh_from_gmsh_model(Ω;dim=3)
gmsh.finalize()
```

- You can take views of the mesh `msh` by passing a domain to the `view` function
- You can plot a `msh` if you have a `Makie` backend (e.g. `GLMakie`)

```@example gmsh-sphere
using CairoMakie # or GLMakie for interactivity if supported
Γ = Inti.external_boundary(Ω)
color = [cos(10*x[1]) for x in Inti.nodes(msh)]
poly(view(msh,Γ);strokewidth=0.2,color, transparency=true)
```

You can also plot the volume mesh (see the Documentation of `Makie.poly` for
more details on possible arguments):

```@example gmsh-sphere
poly(view(msh,Ω);color,transparency=true,strokecolor=:lightgray, alpha = 0.1)
```

Two-dimensional meshes are very similar:

```@example gmsh-disk
    using Inti
    using Gmsh
    using CairoMakie
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
    gmsh.model.add("Disk")
    gmsh.model.occ.addDisk(0,0,0,3,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    Ω   = Inti.gmsh_import_domain(;dim=2)
    msh = Inti.import_mesh_from_gmsh_model(Ω;dim=2)
    gmsh.finalize()
    color = [cos(20*x[1]) for x in Inti.nodes(msh)]
    fig,ax,p = poly(view(msh,Ω);strokewidth=1,color)
    colsize!(fig.layout, 1, Aspect(1, 3))
    resize_to_layout!(fig)
    fig
```
