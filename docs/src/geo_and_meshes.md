# [Geometry and meshes](@id geometry-and-meshes-section)

## Overview

## [Gmsh](@id gmsh-section)

```@example gmsh-sphere
using Inti
using Gmsh

gmsh.initialize()
gmsh.model.add("Sphere")
gmsh.model.occ.addSphere(0,0,0,1)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
ents = gmsh.model.getEntities()
Ω   = Inti.gmsh_import_domain(;dim=3)
msh = Inti.gmsh_import_mesh(Ω;dim=3)
gmsh.finalize()
```

- You can take views of the mesh `msh` by passing a domain to the `view` function
- You can plot a `msh` if you have a `Makie` backend (e.g. `GLMakie`)

```@example gmsh-sphere
using GLMakie
Γ = Inti.external_boundary(Ω)
poly(view(msh,Γ);strokewidth=1,color=:lightgray, transparency=true)
```

You can also plot the volume mesh (see the Documentation of `Makie.poly` for
more details on possible arguments):

```@example gmsh-sphere
poly(view(msh,Ω);strokewidth=1,color=:lightgray, transparency=true)
```

Two-dimensional meshes are very similar:

```@example gmsh-disk
    using Inti
    using Gmsh
    using GLMakie
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("Disk")
    gmsh.model.occ.addDisk(0,0,0,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    Ω   = Inti.gmsh_import_domain(;dim=2)
    msh = Inti.gmsh_import_mesh(Ω;dim=2)
    gmsh.finalize()
    poly(view(msh,Ω);strokewidth=2)
```
