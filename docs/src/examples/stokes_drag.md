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

- ``\mathbf{u} = \mathbf{U}`` on the sphere's surface, where ``\mathbf{U}`` is the velocity
  of the sphere. This is a no-slip condition.
- ``\mathbf{u} \to \mathbf{0}`` at infinity, which means that the fluid is at rest far away
  from the sphere.
The drag force experienced by the sphere is described by [Stokes' law](https://en.wikipedia.org/wiki/Stokes%27_law):

```math
\mathbf{F}_d = -6\pi\mu R \mathbf{U},
```

where ``R`` is the sphere's radius. This drag force, ``\mathbf{F}_d``, is the primary quantity of interest in this example. We will compute it using Hebeker's formulation [hebeker1986efficient](@cite), which expresses the velocity field ``\mathbf{u}`` as a combination of single- and double-layer potentials:

```math
\mathbf{u}(\mathbf{x}) = \mathcal{D}[\boldsymbol{\sigma}](\mathbf{x}) + \eta \mathcal{S}[\boldsymbol{\sigma}](\mathbf{x}),
```

Here, ``\boldsymbol{\sigma}`` is the unknown density, ``\mathcal{S}`` and ``\mathcal{D}`` denote the single- and double-layer potentials, respectively, and ``\eta > 0`` is a coupling parameter, which we set to ``\eta = \mu`` throughout this example.

## Discretization

To discretize the boundary ``\Gamma := \partial \Omega``, we employ a second-order triangular mesh created using Gmsh:

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
    Using `gmsh.model.mesh.setOrder(2)` creates a second-order mesh, which is crucial for
    accurately capturing the curved surface of the sphere and significantly enhances the
    numerical solution's precision. For simple geometries like spheres, an exact
    (isogeometric) representation can also be achieved using `Inti`'s parametric entities.
    See the [Geometry and meshes](@ref "Geometry and meshes") section for more details.

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

## Solution and drag force computation

We are now ready to set up and solve the problem. First, we define the boundary conditions
(a constant velocity on the sphere):

```@example stokes_drag
using StaticArrays
v = 2.0
U = SVector(v,0,0)
f = fill(U, length(Γ_quad))
nothing # hide
```

To solve the linear system, we will use the `gmres` function from `IterativeSolvers`. Since
the function requires scalar types, we need to convert the vector-valued quantities into
scalars and vice versa. We can achieve this by using `reinterpret` to convert between the
vector of `SVector`s and a vector of `Float64`s types.

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
@assert hist.iters < 10 # hide
hist
```

Note that `gmres` converges in very few iterations, highlighting the favorable spectral
properties of the Hebeker formulation for this problem.

### Drag force computation

Now that we have the density `σ`, we can compute the drag force. As pointed out in
[hebeker1986efficient; Theorem 2.4](@cite), the drag force of the body ``\Omega`` is given
by:

```math
    \mathbf{F}_d = \eta \int_{\Gamma} \boldsymbol{\sigma} \, d\Gamma,
```

which can be approximated using our knowledge of `σ` and the quadrature `Γ_quad`:

```@example stokes_drag
drag = μ * sum(eachindex(Γ_quad)) do i
    return σ[i] * Γ_quad[i].weight
end
```

A quick comparison with the analytical solution indicates a good agreement.

```@example stokes_drag
exact = 6π * μ * R * U
relative_error = norm(drag - exact) / norm(exact)
@assert relative_error < 1e-4 # hide
println("Relative error: ", relative_error)
```

The relative error in this example is less than `1e-4`, indicating that the numerical
solution is very close to the analytical solution.

## Visualization

Finally, to visualize the flow field, we need to evaluate our integral representation at
points off the boundary. The easiest way to achieve this is to use
[`IntegralPotential`](@ref)s, or the convenient [`SingleLayerPotential`](@ref) and
[`DoubleLayerPotential`](@ref) wrappers:

```@example stokes_drag
𝒮 = Inti.SingleLayerPotential(op, Γ_quad)
𝒟 = Inti.DoubleLayerPotential(op, Γ_quad)
u(x) = 𝒟[σ](x) + η*𝒮[σ](x) - U # fluid velocity relative to the sphere
```

In the code above, we have created a function `u` that evaluates the velocity at any point
`x`:

```@example stokes_drag
u(SVector(1,2,3))
```

With `u` defined, we can visualize the flow field around the sphere. For this example we
will simply sample points on a grid in the `xz` plane, and plot the velocity vectors at those
points:

```@example stokes_drag
using Meshes
using GLMakie
L = 5
targets     = [SVector(x, 0, z) for x in -L:meshsize:L, z in -L:meshsize:L] |> vec
filter!(x -> norm(x) > 1.1 * R, targets) # remove points inside or close to the sphere
directions  = u.(targets)
fig = Figure()
ax  = Axis3(fig[1, 1]; title = "Velocity field", aspect = :data, limits = ([-L, L], [-R, R], [-L, L]))
viz!(msh[Γ], showsegments=true)
arrows!(ax, Point3.(targets), Point3.(directions), arrowsize = 0.1, color = :blue)
current_figure()
fig
```

## Summary

This tutorial demonstrates how to solve the Stokes drag problem using the Inti library. The
approach combines boundary integral equations with numerical quadrature and iterative
solvers to compute the drag force on a sphere in a viscous fluid.

!!! tip "Extensions"
    - Experiment with different geometries or boundary conditions.
    - Use higher-order quadrature for improved accuracy.
    - Explore the effect of mesh refinement on the solution.
