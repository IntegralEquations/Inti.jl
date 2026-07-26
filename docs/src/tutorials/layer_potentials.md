# Layer potentials

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this tutorial"
    - Nearly singular evaluation of layer potentials
    - Creating a smooth domain with splines using [Gmsh](https://gmsh.info/)'s API
    - Plotting values on a mesh

In this tutorial we focus on **evaluating** the layer potentials given a source
density. This is a common post-processing task in boundary integral equation
methods, and while most of it is straightforward, some subtleties arise when the
target points are close to the boundary (nearly-singular integrals).

## Integral potentials

[`IntegralPotential`](@ref) represent the following mathematical objects:

```math
\mathcal{P}[\sigma](\boldsymbol{r}) = \int_{\Gamma} K(\boldsymbol{r}, \boldsymbol{r'}) \sigma(\boldsymbol{r'}) \, d\boldsymbol{r'}
```

where ``K`` is the kernel of the operator, ``\Gamma`` is the source's boundary,
``\boldsymbol{r} \not \in \Gamma`` is a target point, and ``\sigma`` is the
source density.

Here is a simple example of how to create a kernel representing a Laplace
double-layer potential:

```@example layer_potentials
using Inti, StaticArrays, LinearAlgebra
# define a kernel function
function K(target,source)
    r = Inti.coords(target) - Inti.coords(source)
    ny = Inti.normal(source)
    return 1 / (2π * norm(r)^2) * dot(r, ny)
end
# define a domain
Γ = Inti.parametric_curve(s -> SVector(cos(2π * s), sin(2π * s)), 0, 1) |> Inti.Domain
# and a quadrature of Γ
Q = Inti.Quadrature(Γ; meshsize = 0.1, qorder = 5)
𝒮 = Inti.IntegralPotential(K, Q)
```

If we have a source density ``\sigma``, defined on the quadrature nodes of
``\Gamma``, we can create a function that evaluates the layer potential at an
arbitrary point:

```@example layer_potentials
σ = map(q -> 1.0, Q)
u = 𝒮[σ]
```

`u` is now an anonymous function that evaluates the layer potential at any point:

```@example layer_potentials
r = SVector(0.1, 0.2)
@assert u(r) ≈ -1 # hide
u(r)
```

Although we created the single-layer potential for the Laplace kernel manually,
it is often more convenient to use the `single_layer_potential` when working
with a supported PDE, e.g.:

```@example layer_potentials
op = Inti.Laplace(; dim = 2)
𝒮, 𝒟 = Inti.single_double_layer_potential(; op, source = Q)
```

creates the single and double layer potentials for the Laplace equation in 2D.

## Direct evaluation of layer potentials

We now show how to evaluate the layer potentials of an exact solution on a mesh
created through the Gmsh API. Do to so, let us first define the PDE:g

```@example layer_potentials
using Inti, StaticArrays, LinearAlgebra, GLMakie, Gmsh
# define the PDE
k = 4π
op = Inti.Helmholtz(; dim = 2, k)
```

We will now use the [`gmsh_curve`](@ref) function to create a smooth domain of a
kite using splines:

```@example layer_potentials
gmsh.initialize()
meshsize = 2π / k / 4
kite = Inti.gmsh_curve(0, 1; meshsize) do s
    SVector(0.25, 0.0) + SVector(cos(2π * s) + 0.65 * cos(4π * s[1]) - 0.65, 1.5 * sin(2π * s))
end
cl = gmsh.model.occ.addCurveLoop([kite])
surf = gmsh.model.occ.addPlaneSurface([cl])
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
msh = Inti.import_mesh(; dim = 2)
gmsh.finalize()
```

!!! tip
    The GMSH API is a powerful tool to create complex geometries and meshes
    directly from Julia (the `gmsh_curve` function above is just a simple
    wrapper around some spline functionality). For more information, see the
    [official
    documentation](https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-application-programming-interface).

We can visualize the triangular mesh using:

```@example layer_potentials
using GLMakie
# extract the domain Ω from the mesh entities
ents = Inti.entities(msh)
Ω = Inti.Domain(e->Inti.geometric_dimension(e) == 2, ents)
plot(msh[Ω]; strokewidth = 1, axis = (aspect = DataAspect(), ))
```

For the purpose of testing the accuracy of the layer potential evaluation, we
will construct an exact solution of the Helmholtz equation on the interior
domain and plot it:

```@example layer_potentials
# construct an exact interior solution as a sum of random plane waves
dirs  = [SVector(cos(θ), sin(θ)) for θ in 2π*rand(10)]
coefs = rand(ComplexF64, 10)
u  =  (x)   -> sum(c*exp(im*k*dot(x, d)) for (c,d) in zip(coefs, dirs))
du =  (x,ν) -> sum(c*im*k*dot(d, ν)*exp(im*k*dot(x, d)) for (c,d) in zip(coefs, dirs))
# plot the exact solution
Ω_msh = view(msh, Ω)
target = Inti.nodes(Ω_msh)
plot(Ω_msh; axis = (aspect = DataAspect(), ), color = real(u.(target)))
```

Since `u` satisfies the Helmholtz equation, we know that the following
representation holds:

```math
u(\boldsymbol{r}) = \mathcal{S}[\gamma_1 u](\boldsymbol{r}) - \mathcal{D}[\gamma_0 u](\boldsymbol{r}), \quad \boldsymbol{r} \in \Omega
```

where ``\gamma_0 u`` and ``\gamma_1 u`` are the respective Dirichlet and
Neumann traces of ``u``, and ``\mathcal{S}`` and ``\mathcal{D}`` are the respective
single and double layer potentials over ``\Gamma := \partial \Omega``.

Let's compare next the exact solution with the layer potential evaluation, based
on a quadrature of ``\Gamma``:

```@example layer_potentials
Γ = Inti.boundary(Ω)
Q = Inti.Quadrature(view(msh,Γ); qorder = 5)
# evaluate the layer potentials
𝒮, 𝒟 = Inti.single_double_layer_potential(; op, source = Q)
γ₀u = map(q -> u(q.coords), Q)
γ₁u = map(q -> du(q.coords, q.normal), Q)
uₕ = x -> 𝒮[γ₁u](x) - 𝒟[γ₀u](x)
# plot the error on the target nodes
er_log10 = log10.(abs.(u.(target) - uₕ.(target)))
colorrange = extrema(er_log10)
fig, ax, pl = plot(Ω_msh;
    color = er_log10,
    colormap = :viridis,
    colorrange,
    axis = (aspect = DataAspect(),),
    interpolate=true
)
Colorbar(fig[1, 2]; label = "log₁₀(error)", colorrange)
fig
```

We see a common pattern of potential evaluation: the error is small away from
the boundary, but grows near it. This is due to the nearly-singular nature of
the layer potential integrals, which can be mitigated by using a correction
method that accounts for the singularity of the kernel as ``\boldsymbol{r} \to
\Gamma``.

## Near-field correction of layer potentials

There are two cases where the direct evaluation of layer potentials is not
recommended:

1. When the target point is close to the boundary (nearly-singular integrals).
2. When evaluation at many target points is desired (computationally
   burdensome)and take advantage of an acceleration routine.

In such contexts, it is recommended to use the `single_double_layer` function
(alternately, one can directly assemble an `IntegralOperator`) with a
correction, for the first case, and/or a compression (acceleration) method, for
the latter case, as appropriate. Here is an example of how to use the FMM
acceleration with a near-field correction to evaluate the layer potentials::

```@example layer_potentials
using FMM2D
S, D = Inti.single_double_layer(; op, target, source = Q,
    compression = (method = :fmm, tol = 1e-12),
    correction = (method = :dim, target_location = :inside, maxdist = 0.2)
)
er_log10_cor = log10.(abs.(S*γ₁u - D*γ₀u - u.(target)))
colorrange = extrema(er_log10) # use scale without correction
fig = Figure(resolution = (800, 400))
ax1 = Axis(fig[1, 1], aspect = DataAspect(), title = "Naive evaluation")
plot!(Ω_msh; color = er_log10, colormap = :viridis, colorrange,interpolate=true)
ax2 = Axis(fig[1, 2], aspect = DataAspect(), title = "Nearfield correction")
plot!(Ω_msh; color = er_log10_cor, colormap = :viridis, colorrange, interpolate=true)
Colorbar(fig[1, 3]; label = "log₁₀(error)", colorrange)
fig
```

As can be seen, the near-field correction significantly reduces the error near
the boundary, making if feasible to evaluate the layer potential near ``\Gamma``
if necessary.
