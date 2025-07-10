# Geometry and meshes

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this tutorial"
      - Combine simple shapes to create domains
      - Import a mesh from a file
      - Iterative over mesh elements

In the [getting started](@ref "Getting started") tutorial, we saw how to solve a
simple Helmholtz scattering problem in 2D. We will now dig deeper into how to
create and manipulate more complex geometrical shapes, as well the associated
meshes.

## Overview

Inti.jl provides a flexible way to define geometrical entities and their
associated meshes. Simply put, the `GeometricEntity` type is the atomic building
block of geometries: they can represent points, curves, surfaces, or volumes.
Geometrical entities of the same dimension can be combined to form
[`Domain`](@ref), and domains can be manipulated using basic set operations such
union and intersection. Meshes on the other hand are collections of (simple)
elements that approximate the geometrical entities. A mesh element is a just a
function that maps points from a [`ReferenceShape`](@ref) to the physical space.

In most applications involving complex three-dimensional surfaces, an external
meshing software is used to generate a mesh, and the mesh is imported using the
`import_mesh` function (which relies on [Gmsh](https://gmsh.info)). The entities
can then be extracted from the mesh based on e.g. their dimension or label.
Here is an example of how to import a mesh from a file:

```@example geo-and-meshes
using Inti
using Gmsh 
filename = joinpath(Inti.PROJECT_ROOT,"docs", "assets", "piece.msh")
msh = Inti.import_mesh(filename)
```

The imported mesh contains elements of several types, used to represent the
segments, triangles, and tetras used to approximate the geometry:

```@example geo-and-meshes
Inti.element_types(msh)
```

Note that the `msh` object contains all entities used to construct the mesh,
usually defined in a `.geo` file, which can be extracted using the `entities`:

```@example geo-and-meshes
ents = Inti.entities(msh)
nothing # hide
```

Filtering of entities satisfying a certain condition, e.g., entities of a given
dimension or containing a certain label, can also be performed in order to
construct a domain:

```@example geo-and-meshes
filter = e -> Inti.geometric_dimension(e) == 3
Ω = Inti.Domain(filter, ents)
```

`Domain`s can be used to index the mesh, creating either a new object
containing only the necessary elements:

```@example geo-and-meshes
Γ = Inti.boundary(Ω)
msh[Γ]
```

or a [`SubMesh`](@ref) containing a view of the mesh:

```@example geo-and-meshes
Γ_msh = view(msh, Γ)
```

Finally, we can visualize the mesh using:

```@example geo-and-meshes
using Meshes, GLMakie
fig = Figure(; size = (800,400))
ax = Axis3(fig[1, 1]; aspect = :data)
viz!(Γ_msh; showsegments = true, alpha = 0.5)
fig
```

!!! warning "Mesh visualization"
    Note that although the mesh may be of high order and/or conforming, the
    *visualization* of a mesh is always performed on the underlying first order
    mesh, and therefore elements may look flat even if the problem is solved on
    a curved mesh.

## Parametric entities and `meshgen`

In the previous section we saw an example of how to import a mesh from a file,
and how to extract the entities from the mesh. For simple geometries for which
an explicit parametrization is available, Inti.jl provides a way to create and
manipulate geometrical entities and their associated meshes.

### Parametric curves

The simplest parametric shapes are [`parametric_curve`](@ref)s, which are
defined by a function that maps a scalar parameter `t` to a point in 2D or 3D
space. Parametric curves are expected to return an `SVector`, and can be created
as follows:

```@example geo-and-meshes
using StaticArrays
l1 = Inti.parametric_curve(x->SVector(x, 0.1 * sin(2π * x)), 0.0, 1.0, labels = ["l₁"])
```

The object `l1` represents a `GeometricEntity` with a known push-forward map:

```@example geo-and-meshes
Inti.pushforward(l1)
```

For the sake of this example, let's create three more curves, and group them together to
form a `Domain`:

```@example geo-and-meshes
l2 = Inti.parametric_curve(x->SVector(1 + 0.1 * sin(2π * x), x), 0.0, 1.0, labels = ["l₂"])
l3 = Inti.parametric_curve(x->SVector(1 - x, 1 - 0.1 * sin(2π * x)), 0.0, 1.0, labels = ["l₃"])
l4 = Inti.parametric_curve(x->SVector(0.1 * sin(2π * x), 1 - x), 0.0, 1.0, labels = ["l₄"])
Γ  = l1 ∪ l2 ∪ l3 ∪ l4
```

`Domain`s for which a parametric representation is available can be passed to
the [`meshgen`](@ref) function:

```@example geo-and-meshes
msh = Inti.meshgen(Γ; meshsize = 0.05)
nothing # hide
```

We can use the [`Meshes.viz`](@extref) function to visualize the mesh, and use
domains to index the mesh:

```@example geo-and-meshes
Γ₁ = l1 ∪ l3
Γ₂ = l2 ∪ l4
fig, ax, pl = viz(view(msh, Γ₁); segmentsize = 4,  label = "Γ₁")
viz!(view(msh, Γ₂); segmentsize = 4, color = :red, label = "Γ₂")
fig # hide
```

Note that the orientation of the curve determines the direction of the
[`normal`](@ref) vector. The normal points to the right of the curve when moving
in the direction of increasing parameter `t`:

```@example geo-and-meshes
pts, tangents, normals = Makie.Point2f[], Makie.Vec2f[], Makie.Vec2f[]
for l in [l1, l2, l3, l4]
      push!(pts, l(0.5)) # mid-point of the curve 
      push!(tangents, vec(Inti.jacobian(l, 0.5)))
      push!(normals,Inti.normal(l, 0.5))
end
arrows!(pts, tangents, color = :blue, linewidth = 2, linestyle = :dash, lengthscale = 1/4, label = "tangent")
arrows!(pts, normals, color = :black, linewidth = 2, linestyle = :dash, lengthscale = 1/4, label = "normal")
axislegend()
fig # hide
```

### Parametric surfaces

Like parametric curves, parametric surfaces are defined by a function that maps
a reference domain ``D \subset \mathbb{R}^2`` to a surface in 3D space. They can
be constructed using the [`parametric_surface`](@ref) function:

```@example geo-and-meshes
# a patch of the unit sphere
lc = SVector(-1.0, -1.0)
hc = SVector(1.0, 1.0)
f = (u,v) -> begin
      x = SVector(1.0, u, v)   # a face of the cube
      x ./ sqrt(u^2 + v^2 + 1) # project to the sphere
end
patch = Inti.parametric_surface(f, lc, hc, labels = ["patch1"])
Γ  = Inti.Domain(patch)
msh = Inti.meshgen(Γ; meshsize = 0.1)
viz(msh[Γ]; showsegments = true, figure = (; size = (400,400),))
```

Since creating parametric surfaces that form a closed volume can be a bit more
involved, Inti.jl provide a few helper functions to create simple shapes:

```@example geo-and-meshes
fig = Figure(; size = (600,400))
nshapes = Inti.length(Inti.PREDEFINED_SHAPES)
ncols = 3; nrows = ceil(Int, nshapes/ncols)
for (n,shape) in enumerate(Inti.PREDEFINED_SHAPES)
      Ω = Inti.GeometricEntity(shape) |> Inti.Domain
      Γ = Inti.boundary(Ω)
      msh = Inti.meshgen(Γ; meshsize = 0.1)
      i,j = (n-1) ÷ ncols + 1, (n-1) % ncols + 1
      ax = Axis3(fig[i,j]; aspect = :data, title = shape)
      hidedecorations!(ax)
      viz!(msh; showsegments = true)
end
fig # hide
```

See [`GeometricEntity(shape::String)`](@ref) for a list of predefined geometries.

!!! warning "Mesh quality"
      The quality of the generated mesh created through `meshgen` depends
      heavily on the quality of the underlying parametrization. For surfaces
      containing a degenerate parametrization, or for complex shapes, one is
      better off using a suitable CAD (Computer-Aided Design) software in
      conjunction with a mesh generator.

### Transfinite domains

It is possible to combine parametric curves/surfaces to form a transfinite
domain where the parametrization is inherited from the curves/surfaces that
form its boundary. At present, Inti.jl only supports transfinite squares, which
are defined by four parametric curves:

```@example geo-and-meshes
l1 = Inti.parametric_curve(x->SVector(x, 0.1 * sin(2π * x)), 0.0, 1.0, labels = ["l₁"])
l2 = Inti.parametric_curve(x->SVector(1 + 0.1 * sin(2π * x), x), 0.0, 1.0, labels = ["l₂"])
l3 = Inti.parametric_curve(x->SVector(1 - x, 1 - 0.1 * sin(2π * x)), 0.0, 1.0, labels = ["l₃"])
l4 = Inti.parametric_curve(x->SVector(0.1 * sin(2π * x), 1 - x), 0.0, 1.0, labels = ["l₄"])
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

This allows us to probe the `msh` object to extract e.g. the boundary mesh:

```@example geo-and-meshes
viz(msh[Inti.boundary(Ω)]; color = :red)
```

!!! warning "Limitations"
      At present only the transfinite interpolation for the logically
      quadrilateral domains is supported. In the future we hope to add support
      for three-dimensional transfinite interpolation, as well as transfinite
      formulas for simplices.

## Curving a given mesh

Inti.jl provides some tools to create curved meshes from a given mesh, which
can be useful when the mesh is not conforming to the geometry and the geometry's boundary is
available in parametric form. The following example first creates a flat triangulation of
a disk using splines through Gmsh:

```@example geo-and-meshes
Inti.clear_entities!() # hide
gmsh.initialize()
meshsize = 2π / 4/8
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
# Two kites
f = (s) -> SVector(-1, 0.0) + SVector(cos(2π*s), sin(2π*s))
bnd1 = Inti.gmsh_curve(f, 0, 1; meshsize)
cl = gmsh.model.occ.addCurveLoop([bnd1])
disk = gmsh.model.occ.addPlaneSurface([cl])
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
msh = Inti.import_mesh(; dim = 2)
gmsh.finalize()
Ω = Inti.Domain(Inti.entities(msh)) do ent
      return Inti.geometric_dimension(ent) == 2
end
viz(msh[Ω], showsegments=true)
Ω_quad = Inti.Quadrature(msh[Ω]; qorder = 10)
area = Inti.integrate(x->1.0, Ω_quad)
@assert abs(area - π) > 0.01 # hide
println("Error in area computation using P1 mesh: ", area)
```

As can be seen, despite the large quadrature order employed, the approximation error is
still significant. To improve the accuracy, we can use the `curve_mesh` function
to create a curved mesh based on the boundary of the domain:

```@example geo-and-meshes
gorder = 5
crv_msh = Inti.curve_mesh(msh, f, gorder, 100)
Ω_crv_quad = Inti.Quadrature(crv_msh[Ω]; qorder = 10)
area = Inti.integrate(x->1.0, Ω_crv_quad)
@assert abs(area - π) < 1e-10 # hide
println("Error in area computation using curved mesh: ", area)
```

## Elements of a mesh

 To iterate over the elements of a mesh, use the `elements` function:

```@example geo-and-meshes
filename = joinpath(Inti.PROJECT_ROOT,"docs", "assets", "piece.msh")
msh = Inti.import_mesh(filename)
ents = Inti.entities(msh)
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, ents) 
els = Inti.elements(view(msh, Ω))
centers = map(el -> Inti.center(el), els)
fig = Figure(; size = (800,400))
ax = Axis3(fig[1, 1]; aspect = :data)
scatter!([c[1] for c in centers], [c[2] for c in centers], [c[3] for c in centers], markersize = 5)
fig # hide
```

This example shows how to extract the centers of the tetrahedral elements in the
mesh; and of course we can perform any desired computation on the elements.

!!! tip "Type-stable iteration over elements"
      Since a mesh in Inti.jl can contain elements of various types, the
      `elements` function above is not type-stable. For a type-stable iterator
      approach, one should first iterate over the element types using
      [`element_types`](@ref), and then use `elements(msh, E)` to iterate over a
      specific element type `E`.

Under the hood, each element is simply a functor which maps points `x̂` from a
[`ReferenceShape`](@ref) into the physical space:

```@example geo-and-meshes
el = first(els)
x̂ = SVector(1/3,1/3, 1/3)
el(x̂)
```

Likewise, we can compute the [`jacobian`](@ref) of the element, or its
[`normal`](@ref) at a given parametric coordinate.
