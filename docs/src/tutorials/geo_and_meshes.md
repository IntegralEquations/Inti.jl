# Geometry and meshes

!!! note "Important points covered in this tutorial"
      - Combine simple shapes to create domains
      - Import a mesh from a Gmsh file
      - ...

In the [getting started](@ref Getting started) tutorial, we saw how to solve a
simple Helmholtz scattering problem in 2D. Since domains and meshes are
fundamental to solving boundary and volume integral equations, we will now dig
deeper into how to create and manipulate them in Inti.jl.

For more complex geometries,
the recommended approach is to use *Gmsh* to create the geometry and mesh, and
then import it into Inti.jl.

## Parametric curves

Inti.jl offers a limited support for creating simple shapes for which an
explicit parametric representation is available. The simplest of such shapes are
`parametric_curve`s, which are defined by a function that maps a scalar
parameter `t` to a point in 2D or 3D space. Parametric curves are expected to
return an `SVector`, and can be created as follows:

```@example geo-and-meshes
using Inti, StaticArrays, Meshes, GLMakie
l1 = Inti.parametric_curve(x->SVector(x, 0.1 * sin(2π * x)), 0.0, 1.0, labels = ["l₁"])
l2 = Inti.parametric_curve(x->SVector(1 + 0.1 * sin(2π * x), x), 0.0, 1.0, labels = ["l₂"])
l3 = Inti.parametric_curve(x->SVector(1 - x, 1 - 0.1 * sin(2π * x)), 0.0, 1.0, labels = ["l₃"])
l4 = Inti.parametric_curve(x->SVector(0.1 * sin(2π * x), 1 - x), 0.0, 1.0, labels = ["l₄"])
```

Each variable above represents a geometrical entity, and entities can be
combined to form a [`Domain`](@ref) object that can be passed to the
[`meshgen`](@ref) function. For instance, the following code creates a domain
formed by three of the four lines, and generates a mesh for it:

```@example geo-and-meshes
Γ1 = Inti.Domain(l1, l3)
Γ2 = Inti.Domain(l2, l4)
Γ1_msh = Inti.meshgen(Γ1; meshsize = 0.05)
Γ2_msh = Inti.meshgen(Γ2; meshsize = 0.05)
fig, ax, pl = viz(Γ1_msh; segmentsize = 4,  label = "Γ₁")
viz!(Γ2_msh; segmentsize = 4, color = :red, label = "Γ₂")
axislegend()
fig # hide
```

`Domain`s can be manipulated using basic set operations, such as `union`, and
`intersect`:

```@example geo-and-meshes
# Γ = Γ1 ∩ Γ2
Γ  = Γ1 ∪ Γ2
```

## Transfinite squares

Note that you can also combine the lines to form a transfinite square, which
inherits its parametrization from the lines that form it:

```@example geo-and-meshes
surf = Inti.transfinite_square(l1, l2, l3, l4; labels = ["Ω"])
Ω = Inti.Domain(surf)
msh = Inti.meshgen(Ω; meshsize = 0.05)
viz(msh; showsegments = true)
```

Note that the `msh` object contains all entities used to construct `Ω`,
including the boundary segments:

```@example geo-and-meshes
Inti.entities(msh)
```

This allows you to index the mesh by `Domain`s, extracting either a new mesh:

```@example geo-and-meshes
msh[Γ1]
```

or a view of the mesh:

```@example geo-and-meshes
view(msh, Γ1)
```

```@example geo-and-meshes
viz(view(msh,Γ1); segmentsize = 4, label = "view of Γ₁")
axislegend()
```