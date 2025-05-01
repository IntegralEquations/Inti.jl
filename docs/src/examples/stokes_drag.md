# Stokes Drag

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this example"
    - Solving a vector-valued problem
    - Usage of curved triangular mesh
    - Post-processing integral quantities

## Problem description

In this example, we solve the classical Stokes drag problem, which models the drag force
experienced by a sphere moving through a viscous fluid. The governing equations are the
Stokes equations:

```math
\begin{align*}
-\nabla p + \mu \Delta \mathbf{u} &= 0, \quad && \text{in } \Omega^c, \\
\nabla \cdot \mathbf{u} &= 0, \quad && \text{in } \Omega^c,
\end{align*}
```

where:

- ``\mathbf{u}`` is the velocity field,
- ``p`` is the pressure,
- ``\mu`` is the dynamic viscosity,
- ``\Omega`` is the sphere, and ``\Omega^c = \mathbb{R}^3 \setminus \overline{\Omega}`` is
  the fluid domain.

The boundary conditions are:

- No-slip condition on the sphere's surface,
- Velocity at infinity is constant.

We compute the drag force on the sphere using integral quantities derived from the solution.
For a sphere of radius ``R`` moving with velocity ``\mathbf{U}``, the [drag force](https://en.wikipedia.org/wiki/Stokes%27_law) is given by:

```math
\mathbf{F}_d = 6\pi\mu R \mathbf{U}.
```

which we will use to validate our numerical solution. We will employ Hebeker's formulation
[hebeker1986efficient](@cite), where we seek the solution ``\mathbf{u}`` in the form of a
combined single- and double-layer potential:

```math
\mathbf{u}(\mathbf{x}) = \mathcal{D}[\boldsymbol{\sigma}](\mathbf{x}) + \eta \mathcal{S}[\boldsymbol{\sigma}](\mathbf{x}),
```

where ``\boldsymbol{\sigma}`` is the unknown density, ``\mathcal{S}`` and ``\mathcal{D}``
are the single- and double-layer potentials, respectively, and ``\eta > 0`` is a (free)
coupling parameter which we set to ``\eta = \mu`` throughout this example. 

As pointed out in [hebeker1986efficient; Theorem 2.4](@cite), the drag force of the body
``\Omega`` is given by:

```math
    \mathbf{F}_d = \eta \int_{\Gamma} \boldsymbol{\sigma} \, d\Gamma,
```

which is the formula we will later use to compute the drag force.

## Discretization

To numerically discretize the boundary ``\Gamma := \partial \Omega``, we employ a second-order triangular mesh created using Gmsh:

```@example stokes_drag
using Inti, Gmsh
meshsize = 0.4
R = 2.0
gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.model.occ.addSphere(0, 0, 0, R)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)
msh = Inti.import_mesh()
gmsh.finalize()
nothing # hide
```

!!! tip "Second-order mesh"
    Using `gmsh.model.mesh.setOrder(2)` creates a second-order mesh, which is crucial for accurately capturing the curved surface of the sphere and significantly enhances the numerical solution's precision. For simple geometries like spheres, an exact (isogeometric) representation can also be achieved using `Inti`'s parametric entities. Refer to the [Geometry and meshes]("Geometry and meshes") tutorial for further details.

Next we extract the `Domain` ``\Gamma`` from the mesh, and create a `Quadrature` on it:

```@example stokes_drag
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, Inti.entities(msh)) # the 3D volume
Γ = Inti.boundary(Ω) # its boundary
Γ_msh = view(msh, Γ)
Γ_quad = Inti.Quadrature(Γ_msh; qorder = 2) # quadrature on the boundary
nothing # hide
```

With the quadrature prepared, we can now define the Stokes operator along with its
associated integral operators. We use the [FMM3D](https://fmm3d.readthedocs.io/en/latest/)
library to accelerate the evaluation of the integral operators:

```@example stokes_drag
using FMM3D
# pick a correction and compression method
correction = (method = :adaptive, )
compression = (method = :fmm, )

# define the Stokes operator
μ = η = 2.0
op = Inti.Stokes(; dim = 3, μ)

# assemble integral operators
S, D = Inti.single_double_layer(;
    op,
    target = Γ_quad,
    source = Γ_quad,
    compression,
    correction,
)
```

## Solution and visualization

We are now ready to set up and solve the problem. First, we define the boundary conditions
(a constant velocity on the sphere):

```@example stokes_drag
using StaticArrays
v = 2.0
U = SVector(2.0,0,0)
f = fill(U, length(Γ_quad))
nothing # hide
```

To solve the linear system, we will use the `gmres` function from `IterativeSolvers`. Since
the function requires scalar types, we need to convert the vector-valued quantities into
scalars and vice versa. We can achieve this by using `reinterpret` to convert the data types:

```@example stokes_drag
using IterativeSolvers, LinearAlgebra, LinearMaps
T = SVector{3, Float64} # vector type
L = I/2 + D + η * S
L_ = LinearMap{Float64}(3 * size(L, 1)) do y, x
    σ = reinterpret(T, x)
    μ = reinterpret(T, y)
    mul!(μ, L, σ)
    return y
end
σ  = zeros(T, length(Γ_quad))
σ_ = reinterpret(Float64, σ)
f_ = reinterpret(Float64, f)
_, hist = gmres!(σ_, L_, f_; reltol = 1e-8, maxiter = 200, restart = 200, log = true)
nothing # hide
```

### Drag force computation

The drag force is computed as:

```@example stokes_drag
drag = μ * sum(eachindex(Γ_quad)) do i
    return σ[i] * Γ_quad[i].weight
end
exact = 6π * μ * R * v
relative_error = (norm(drag) - exact) / exact
```

### Solution

The solution is given by

```@example stokes_drag
𝒮 = Inti.SingleLayerPotential(op, )

```

### Visualization

```@example stokes_drag
using Meshes
using GLMakie
fig = Figure()
ax  = Axis3(fig[1, 1]; title = "Stokes drag", aspect = :equal)
viz!(msh[Γ], showsegments=true)
xx = yy = zz = collect(-2:0.2:2)
quiver!()
current_figure()
```

## Summary

This tutorial demonstrates how to solve the Stokes drag problem using the Inti library. The approach combines boundary integral equations with numerical quadrature and iterative solvers to compute the drag force on a sphere in a viscous fluid.

!!! tip "Extensions"
    - Experiment with different geometries or boundary conditions.
    - Use higher-order quadrature for improved accuracy.
    - Explore the effect of mesh refinement on the solution.
