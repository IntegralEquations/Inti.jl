# Getting started

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this tutorial"
      - Create a domain and its accompanying mesh
      - Solve a basic boundary integral equation
      - Visualize the solution

This first tutorial will guide you through the basic steps of setting up a
boundary integral equation and solving it using Inti.jl. We will consider the
classic Helmholtz scattering problem in 2D, and solve it using a *direct*
boundary integral formulation. More precisely, letting ``\Omega \subset
\mathbb{R}^2`` be a bounded domain, and denoting by ``\Gamma = \partial \Omega``
its boundary, we will solve the following Helmholtz problem:

```math
\begin{aligned}
    \Delta u + k^2 u  &= 0 \quad &&\text{in} \quad \mathbb{R}^2 \setminus \overline{\Omega},\\
    \partial_\nu u &= g \quad &&\text{on} \quad \Gamma,\\
    \sqrt{r} \left( \frac{\partial u}{\partial r} - i k u \right) &= o(1) \quad &&\text{as} \quad r = |\boldsymbol{x}| \to \infty,
\end{aligned}
```

where ``g`` is the given boundary datum, ``\nu`` is the outward unit normal to
``\Gamma``, and ``k`` is the constant wavenumber.

!!! info "Sommerfeld radiation condition"
    The last condition is the *Sommerfeld radiation condition*, and is required
    to ensure the uniqueness of the solution; physically, it means that the
    solution sought should radiate energy towards infinity.

The first step is to define the PDE under consideration:

```@example getting_started
using Inti
# PDE
k = 2π
pde = Inti.Helmholtz(; dim = 2, k)
```

Next, we defined the geometry of the problem. For this tutorial, we will
manually create parametric curves representing the boundary of the domain using
the [`parametric_curve`](@ref) function:

```@example getting_started
using StaticArrays # for SVector
# Create the geometry as the union of a kite and a circle
kite = Inti.parametric_curve(0.0, 1.0; labels = ["kite"]) do s
    return SVector(2.5 + cos(2π * s[1]) + 0.65 * cos(4π * s[1]) - 0.65, 1.5 * sin(2π * s[1]))
end
circle = Inti.parametric_curve(0.0, 1.0; labels = ["circle"]) do s
    return SVector(cos(2π * s[1]), sin(2π * s[1]))
end
Γ = kite ∪ circle
```

Inti.jl expects the parametrization of the curve to be a function mapping
scalars to points in space represented by `SVector`s. The `labels` argument is
optional, and can be used to identify the different parts of the boundary. The
`Domain` object `Γ` represents the boundary of the geometry, and can be used to
create a mesh:

```@example getting_started
# Create a mesh for the geometry
msh = Inti.meshgen(Γ; meshsize = 2π / k / 10)
```

Once the mesh is created, we can define a quadrature to be used in the
discretization of the integral operators:

```@example getting_started
# Create a quadrature
Q = Inti.Quadrature(msh; qorder = 5)
nothing # hide
```

A [`Quadrature`](@ref) is simply a collection of [`QuadratureNode`](@ref)
objects:

```@example getting_started
Q[1]
```

To visualize the mesh, we can load
[Meshes.jl](https://github.com/JuliaGeometry/Meshes.jl) and one of
[Makie](https://github.com/MakieOrg/Makie.jl)'s backends:

```@example getting_started
using Meshes, GLMakie
viz(msh; segmentsize = 3, axis = (aspect = DataAspect(), ), figure = (; size = (600,400)))
```

To continue, we need to reformulate the Helmholtz problem as a boundary integral
equation. Among the plethora of options, we will use in this tutorial a simple
*direct* formulation, which uses Green's third identity to relate the values of
``u`` and ``\partial_{\nu} u`` on ``\Gamma``:

```math
    -\frac{u(\boldsymbol{x})}{2} + D[u](\boldsymbol{x}) = S[\partial_\nu u](\boldsymbol{x}), \quad \boldsymbol{x} \in \Gamma.
```

Here ``S`` and ``D`` are the single- and double-layer operators, formally
defined as:

```math
    S[\sigma](\boldsymbol{x}) = \int_\Gamma G(\boldsymbol{x}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \ \mathrm{d}s(\boldsymbol{y}), \quad
    D[\sigma](\boldsymbol{x}) = \int_\Gamma \frac{\partial G}{\partial \nu_{\boldsymbol{y}}}(\boldsymbol{x}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \ \mathrm{d}s(\boldsymbol{y}),
```

where 

```math
G(\boldsymbol{x}, \boldsymbol{y}) = \frac{i}{4} H_0(k|\boldsymbol{x} -
\boldsymbol{y}|)
```

is the fundamental solution of the Helmholtz equation, with ``H_0`` being the
Hankel function of the first kind. Note that ``G`` is singular when
``\boldsymbol{x} = \boldsymbol{y}``, and therefore the numerical discretization
of ``S`` and ``D`` requires special care.

To approximate ``S`` and ``D`` in Inti.jl we can proceed as follows:

```@example getting_started
S, D = Inti.single_double_layer(;
    pde,
    target = Q,
    source = Q,
    compression = (method = :none,),
    correction = (method = :dim,),
)
nothing # hide
```

!!! tip "Fast algorithms"
    Powered by external libraries, Inti.jl supports several acceleration methods
    for matrix-vector multiplication, including so far:
    - **Fast multipole method** (FMM) ``\mapsto`` `correction = (method = :fmm, tol = 1e-8)`
    - **Hierarchical matrix** (H-matrix) ``\mapsto`` `correction = (method = :hmatrix, tol = 1e-8)`
  
    Note that in such cases only the matrix-vector product may not be available, and therefore iterative solvers such as GMRES are required for the solution of the resulting linear systems.

Much of the complexity involved in the numerical computation is hidden in the
function above; later in the tutorials we will discuss in more details the
options available for the *compression* and *correction* methods, as well as how
to define your own kernels and operators. For now, it suffices to know that `S`
and `D` are matrix-like objects that can be used to solve the boundary integral
equation. For that, we need to provide the boundary data ``g``.

We are interested in the scattered field ``u`` produced by an incident plane
wave ``u_i = e^{i k \boldsymbol{d} \cdot \boldsymbol{x}}``, where
``\boldsymbol{d}`` is a unit vector denoting the direction of the plane wave.
Assuming that the total field ``u_t = u_i + u`` satisfies a homogenous Neumann
condition on ``\Gamma``, and that the scattered field ``u`` satisfies the
Sommerfeld radiation condition, we can write the boundary condition as:

```math
    \partial_\nu u = -\partial_\nu u_i, \quad \boldsymbol{x} \in \Gamma.
```

We can thus solve the boundary integral equation to find ``u`` on ``\Gamma``:

```@example getting_started
# define the incident field and compute its normal derivative
θ = 0
d = SVector(cos(θ), sin(θ))
g = map(Q) do q
    # normal derivative of e^{ik*d⃗⋅x}
    x, ν = q.coords, q.normal
    return -im * k * exp(im * k * dot(x, d)) * dot(d, ν)
end ## Neumann trace on boundary
u = (-I / 2 + D) \ (S * g) # Dirichlet trace on boundary
```

Now that we know both the Dirichlet and Neumann data on the boundary, we can use
Green's representation formula, i.e., 

```math
    \mathcal{D}[u](\boldsymbol{r}) - \mathcal{S}[\partial_{\nu} u](\boldsymbol{r}) = \begin{cases}
        u(\boldsymbol{r}) & \text{if } \boldsymbol{r} \in \mathbb{R}^2 \setminus \overline{\Omega},\\
        0 & \text{if } \boldsymbol{r} \in \Omega,
    \end{cases}
```

where ``\mathcal{D}`` and ``\mathcal{S}`` are the double- and single-layer
potentials defined as:

```math
    \mathcal{S}[\sigma](\boldsymbol{r}) = \int_{\Gamma} G(\boldsymbol{r}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \ \mathrm{d}s(\boldsymbol{y}), \quad
    \mathcal{D}[\sigma](\boldsymbol{r}) = \int_{\Gamma} \frac{\partial G}{\partial \nu_{\boldsymbol{y}}}(\boldsymbol{r}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \ \mathrm{d}s(\boldsymbol{y}),
```

to compute the solution ``u`` in the domain:

```@example getting_started
𝒮, 𝒟 = Inti.single_double_layer_potential(; pde, source = Q)
uₛ = x -> 𝒟[u](x) - 𝒮[g](x)
```

To wrap things up, let's visualize the scattered field:

```@example getting_started
xx = yy = range(-5; stop = 5, length = 100)
U = map(uₛ, Iterators.product(xx, yy))
Ui = map(x -> exp(im*k*dot(x, d)), Iterators.product(xx, yy))
Ut = Ui + U
fig, ax, hm = heatmap(
    xx,
    yy,
    real(Ut);
    colormap = :inferno,
    interpolate = true,
    axis = (aspect = DataAspect(), xgridvisible = false, ygridvisible = false),
)
viz!(msh; segmentsize = 2)
Colorbar(fig[1, 2], hm; label = "real(u)")
fig # hide
```

!!! tip "Going further"
    - ...

```@example getting_started
# build an exact solution
G = Inti.SingleLayerKernel(pde)
dG = Inti.DoubleLayerKernel(pde)
xs = map(θ -> 0.5 * rand() * SVector(cos(θ), sin(θ)), 2π * rand(10))
cs = rand(ComplexF64, length(xs))
uₑ  = q -> sum(c * G(x, q) for (x, c) in zip(xs, cs))
∂ₙu = q -> sum(c * dG(x, q) for (x, c) in zip(xs, cs))
g  = map(∂ₙu, Q) 
u = (-I / 2 + D) \ (S * g)
uₛ = x -> 𝒟[u](x) - 𝒮[g](x)
pts = [5*SVector(cos(θ), sin(θ)) for θ in range(0, 2π, length = 100)]
er = norm(uₛ.(pts) - uₑ.(pts), Inf)
println("maximum error on circle of radius 5: $er")
```
