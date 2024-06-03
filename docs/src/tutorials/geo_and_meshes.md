# Geometry and meshes

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this tutorial"
      - Combine simple shapes to create domains
      - Import a mesh from a Gmsh file
      - ...

In the [getting started](@ref "Getting started") tutorial, we saw how to solve a
simple Helmholtz scattering problem in 2D. Since domains and meshes are
fundamental to solving boundary and volume integral equations, we will now dig
deeper into how to create and manipulate them in Inti.jl.

## Parametric curves

Inti.jl offers a limited support for creating simple shapes for which an
explicit parametric representation is available. The simplest of such shapes are
[`parametric_curve`](@ref)s, which are defined by a function that maps a scalar
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
combined to form a [`Domain`](@ref) object which can be passed to the
[`meshgen`](@ref) function. For instance, the following code creates two domains
comprised of two curves each, and meshes them:

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

`Domain`s can be manipulated using basic set operations, such as `intersect`

```@example geo-and-meshes
@assert isempty(Γ1 ∩ Γ2) # hide
Γ = Γ1 ∩ Γ2 # empty
```

and `union`:

```@example geo-and-meshes
@assert Γ1 ∪ Γ2 == Γ # hide
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
fig, ax, p = viz(view(msh,Γ1); segmentsize = 4, label = "view of Γ₁")
axislegend()
fig # hide
```

## Importing meshes

While creating simple geometries natively in Inti.jl is useful for testing and
academic problems, most real-world applications require manipulating complex
geometries through CAD and meshing software. Inti.jl possesses a
[Gmsh](https://gmsh.info) extension that will load additional functionality if
Gmsh.jl is loaded. This extension allows you to import meshes from Gmsh files:

```@example geo-and-meshes
using Gmsh
filename = joinpath(@__DIR__, "piece.msh")
msh = Inti.import_mesh(filename)
```

Note that the `msh` object contains all entities used to construct the mesh:

```@example geo-and-meshes
ents = Inti.entities(msh)
```

You need to filter entities satisfying a certain condition, e.g., entities of a
given dimension of containing a certain label, in order to construct a domain:

```@example geo-and-meshes
filter = e -> Inti.geometric_dimension(e) == 2
Γ = Inti.Domain(filter, ents)
```

As before, you can visualize the mesh using Meshes.jl:

```@example geo-and-meshes
using Meshes, GLMakie
viz(view(msh,Γ); showsegments = true, alpha = 0.5)
```

## Elements of a mesh

Although we have created several meshes this far in the tutorial, we have not
done much with them except for visualizing. Conceptually, a mesh is simply a
collection of elements, of possibly different type. To iterate over the elements
of a mesh and perform some computation, you can simply use the `elements` function:

```@example geo-and-meshes
els = Inti.elements(view(msh, Γ))
centers = map(el -> Inti.center(el), els)
scatter([c[1] for c in centers], [c[2] for c in centers], [c[3] for c in centers], markersize = 5)
```

This example shows how to extract the centers of the elements in the mesh, and
of course you can perform any computation you like on the elements.

!!! tip "Type-stable iteration over elements"
      Since a mesh in Inti.jl can contain elements of various types, the
      `elements` function above is not type-stable. For a type-stable iterator
      approach, you should first iterate over the element types using
      [`element_types`](@ref), and then use `elements(msh, E)` to iterate over a
      specific element type `E`.

Under the hood, each element is simply a functor which maps points `x̂` from a
[`ReferenceShape`](@ref) into the physical space:

```@example geo-and-meshes
el = first(els)
x̂ = SVector(1/3,1/3)
el(x̂)
```

You can compute the Jacobian of an element

```@example geo-and-meshes
Inti.jacobian(el, x̂)
```

or its normal vector if the element is of co-dimension one:

```@example geo-and-meshes
Inti.normal(el, x̂)
```
