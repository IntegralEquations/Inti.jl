# Poisson problem

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this example"
      - Reformulating Poisson-like problems using integral equations
      - Using volume potentials
      - Creating interior meshes using Gmsh

## Problem definition

In this example we will solve the Poisson equation in a domain ``\Omega`` with
Dirichlet boundary conditions on ``\Gamma := \partial \Omega``:

```math
  \begin{align*}
      -\Delta u &= f  \quad \text{in } \quad \Omega\\
      u &= g  \quad \text{on } \quad \Gamma
  \end{align*}
```

where ``f : \Omega \to \mathbb{R}`` and ``g : \Gamma \to \mathbb{R}`` are given
functions.

To solve this problem using integral equations, we split the solution ``u`` into
a particular solution ``u_p`` and a homogeneous solution ``u_h``:

```math
  u = u_p + u_h
```

The function ``u_p`` is given by

```math
u_p(\boldsymbol{r}) = \int_{\Omega} G(\boldsymbol{r}, \boldsymbol{r'}) f(\boldsymbol{r'}) d\boldsymbol{r'}.
```

with ``G`` the fundamental solution of ``-\Delta``.

The function ``u_h`` satisfies the homogeneous problem

```math
  \begin{align*}
      \Delta u_h &= 0  \quad &&\text{in } \quad \Omega \\
      u_h &= g - u_p  \quad &&\text{on }  \quad \Gamma
  \end{align*}
```

which can be solved using the integral equation method. In particular, for this
example, we employ a double-layer formulation:

```math
u_h(\boldsymbol{r}) = \int_{\Gamma} G(\boldsymbol{r}, \boldsymbol{r'}) \sigma(\boldsymbol{r}') d\boldsymbol{r'}.
```

where the density function ``\sigma`` solves the following integral equation:

```math
  -\frac{\sigma(\boldsymbol{x})}{2} + \int_{\Gamma} \partial_{\nu_{\boldsymbol{y}}}G(\boldsymbol{x}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \ \mathrm{d} s_{\boldsymbol{y}} = g(\boldsymbol{x}) - u_p(\boldsymbol{x}).
```

In what follows we will illustrate how to solve the problem above using integral
equations.

## Geometry and mesh

We use the *Gmsh API* to create a jellyfish shaped domain and to generate a
second order mesh of its interior and boundary:

```@example poisson
using Inti, Gmsh
meshsize = 0.1
gmsh.initialize()
jellyfish = Inti.gmsh_curve(0, 2π; meshsize) do s
    r = 1 + 0.3*cos(4*s + 2*sin(s))
    return r*Inti.Point2D(cos(s), sin(s))
end
cl = gmsh.model.occ.addCurveLoop([jellyfish])
surf = gmsh.model.occ.addPlaneSurface([cl])
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)
msh = Inti.import_mesh(; dim = 2)
gmsh.finalize()
```

We can now extract ``\Omega`` and ``\Gamma`` from the mesh:

```@example poisson
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
Γ = Inti.boundary(Ω)
Ω_msh = view(msh, Ω)
Γ_msh = view(msh, Γ)
nothing #hide
```

and visualize them:

```@example poisson
using Meshes, GLMakie
viz(Ω_msh; showsegments=true)
viz!(Γ_msh; color=:red)
Makie.current_figure() #hide
```

To conclude the geometric setup, we need a quadrature for the volume and
boundary:

```@example poisson
Ω_quad = Inti.Quadrature(Ω_msh; qorder = 4)
Γ_quad = Inti.Quadrature(Γ_msh; qorder = 6)
nothing # hide
```

## Integral operators

We can now assemble the required volume potential:

```@example poisson
using FMM2D #to accelerate the maps
pde = Inti.Laplace(; dim = 2)
# Newtonian potential mapping domain to boundary
V_d2b = Inti.volume_potential(;
    pde,
    target = Γ_quad,
    source = Ω_quad,
    compression = (method = :fmm, tol = 1e-12),
    correction = (
        method = :dim,
        maxdist = 5 * meshsize,
        target_location = :on,
    ),
)
```

as well as the boundary integral operators

```@example poisson
# Single and double layer operators on Γ
S_b2b, D_b2b = Inti.single_double_layer(;
    pde,
    target = Γ_quad,
    source = Γ_quad,
    compression = (method = :fmm, tol = 1e-12),
    correction = (method = :dim,),
)
```

!!! note
    In this example we used the Fast Multipole Method (`:fmm`) to accelerate the
    operators, and the Density Interpolation Method (`:dim`) to correct singular
    and nearly-singular integral.

## Solving the linear system

We are now in a position to solve the original Poisson problem, but for that we
need to specify the functions $f$ and $g$. In order to verify that our numerical
approximation is correct, however, we will play a different game and specify
instead a manufactured solution ``u_e`` from which we will derive the functions
``f`` and ``g``:

```@example poisson
# Create a manufactured solution
uₑ = (x) -> cos(2 * x[1]) * sin(2 * x[2])
fₑ  = (x) -> 8 * cos(2 * x[1]) * sin(2 * x[2]) # -Δuₑ
g   = map(q -> uₑ(q.coords), Γ_quad)
f   = map(q -> fₑ(q.coords), Ω_quad)
nothing # hide
```

With these, we can compute the right-hand-side of the integral equation for the
homogeneous part of the solution:

```@example poisson
rhs = g - V_d2b*f
nothing # hide
```

and solve the integral equation for the density function ``σ``:

```@example poisson
using IterativeSolvers, LinearAlgebra
σ = gmres(-I/2 + D_b2b, rhs; abstol = 1e-8, verbose = true, restart = 1000)
nothing # hide
```

With the density function at hand, we can now reconstruct our approximate solution:

```@example poisson
G  = Inti.SingleLayerKernel(pde)
dG = Inti.DoubleLayerKernel(pde)
𝒱 = Inti.IntegralPotential(G, Ω_quad)
𝒟 = Inti.IntegralPotential(dG, Γ_quad)
u = (x) -> 𝒱[f](x) + 𝒟[σ](x)
```

and evaluate it at any point in the domain:

```@example poisson
x = Inti.Point2D(0.1,0.4)
println("error at $x: ", u(x)-uₑ(x))
```

## Solution evaluation and visualization

Although we have "solved" the problem in the previous section, using the
anonymous function `u` to evaluate the field is neither efficient nor accurate
when there are either many points to evaluate, or when they lie close to the
boundary ``\Gamma``. The fundamental reason for this is the usual: our
integral operators are dense matrices, and their evaluation near ``\Gamma``
suffers from the so-called "near-singularity" problem.

To address this issue, we need to assemble *accelerated* and *corrected*
versions of the integral operators. Let us suppose we wish to evaluate the
solution ``u`` at all mesh nodes of ``\Omega``, which will be our `target`:

```@example poisson
target = Inti.nodes(Ω_msh)
```

We now create a volume operator mapping densities from our domain quadrature to our
mesh nodes:

```@example poisson
V_d2d = Inti.volume_potential(;
    pde,
    target = target,
    source = Ω_quad,
    compression = (method = :fmm, tol = 1e-8),
    correction = (method = :dim, target_location = :inside, maxdist = meshsize),
)
```

Likewise, we need operators mapping densities from our boundary quadrature to
our mesh nodes:

```@example poisson
S_b2d, D_b2d = Inti.single_double_layer(;
    pde,
    target = target,
    source = Γ_quad,
    compression = (method = :fmm, tol = 1e-8),
    correction = (method = :dim, maxdist = meshsize, target_location = :inside),
)
```

We now evaluate the solution at all mesh nodes and compare it to the manufactured:

```@example poisson
u_nodes = V_d2d*f + D_b2d*σ
er = u_nodes - map(uₑ, target)
println("maximum error at all mesh nodes:", norm(er, Inf))
```

Lastly, let us visualize the solution:

```@example poisson
colorrange = extrema(u_nodes)
fig = Figure(; size = (800, 500))
ax = Axis(fig[1, 1]; aspect = DataAspect())
viz!(Ω_msh; colorrange, color = u_nodes, interpolate = true)
cb = Colorbar(fig[1, 2]; label = "u", colorrange)
fig # hide
```

as well as the error:

```@example poisson
colorrange = extrema(er)
colormap = :inferno
fig = Figure(; size = (800, 500))
ax = Axis(fig[1, 1]; aspect = DataAspect())
viz!(Ω_msh; colorrange, colormap, color = er, interpolate = true)
cb = Colorbar(fig[1, 2]; label = "u - uₑ", colormap, colorrange)
fig # hide
```
