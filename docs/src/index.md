# Inti

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://IntegralEquations.github.io/Inti.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://IntegralEquations.github.io/Inti.jl/dev/)
[![Build Status](https://github.com/IntegralEquations/Inti.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/IntegralEquations/Inti.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/IntegralEquations/Inti.jl/graph/badge.svg?token=2VF6BR8LA0)](https://codecov.io/gh/IntegralEquations/Inti.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[Inti.jl](https://github.com/IntegralEquations/Inti.jl) is a Julia library for
the numerical solution of *boundary* and *volume integral equations*. It
provides routines for *assembling* and *solving* the linear systems of equations
that arise from applying a *Nyström* discretization method to an integral
equation. The package is designed to be flexible and efficient, and supports the
following features:

- Specialized integration routines for computing singular and nearly-singular
  integrals
- Integrated support for acceleration routines such as the Fast Multipole Method (FMM) and
  Hierarchical Matrices by the wrapping of external libraries.
- Predefined kernels for some partial differential equations (PDEs) commonly
  found in mathematical physics such as the Laplace, Helmholtz, and Stokes
  equations.
- Support for complex geometries in 2D and 3D, either through native parametric
  representations or by importing complex meshes using *Gmsh*.
- Efficient construction of complex integral operators from simpler ones through
  lazy composition.

## Installing Julia

Download of Julia from [julialang.org](https://julialang.org/downloads/), or use
[juliaup](https://github.com/JuliaLang/juliaup) installer. We recommend using
the latest stable version of Julia, although `Inti.jl` should work with
`>=v1.6`.

## Installing Inti.jl

Inti.jl is not yet registered in the Julia General registry. You can install it
using by launching a Julia REPL and typing the following command:

```julia
using Pkg; Pkg.add("Inti"; rev = "main")
```

This will download and install the latest version of Inti.jl from the `main`
branch. Change `rev` if you need a different branch or specific commit hash.

## Basic usage

Inti.jl can be used to solve a variety of linear partial differential equations
by recasting them as integral equations. The general workflow for solving a
problem consists of the following steps:

```math
    \underbrace{\fbox{Geometry} \rightarrow \fbox{Mesh}}_{\textbf{pre-processing}} \rightarrow \fbox{\color{red}{Solver}} \rightarrow \underbrace{\fbox{Visualization}}_{\textbf{post-processing}}
```

- **Geometry**: The first step is to define the domain of interest, which can be
  done using a combination of simple shapes (e.g., circles, rectangles) or more
  complex *CAD* models.
- **Mesh**: The mesh is a collection of points and segments that approximate the
  domain. The mesh is used to create a quadrature and discretize the boundary
  integral equation.
- **Solver**: Once a mesh and an accompanying quadrature are available, Inti.jl
  provides routines for assembling and solving the linear system of equations
  arising from the discretization of the integral operators. Once found, the
  solution can be evaluated using an integral representation.
- **Visualization**: Finally, the solution can be visualized using a plotting library, such
  as `Makie.jl`, or exported to a file for further analysis.

As a simple example, consider an interior Laplace problem in two dimensions with
Dirichlet boundary conditions:

```math
\begin{aligned}
\Delta u &= 0 \quad \text{in } \Omega,\\ 
u &= g \quad \text{on } \Gamma,
\end{aligned}
```

where ``\Omega \subset \mathbb{R}^2`` is a sufficiently smooth domain, and
``\Gamma = \partial \Omega`` its boundary. A boundary integral reformulation can
be achieved by e.g. searching for the solution $u$ in the form of a single-layer
potential:

```math
u(\boldsymbol{r}) = \int_\Gamma G(\boldsymbol{r},\boldsymbol{y})\sigma(\boldsymbol{y}) \ \mathrm{d}\Gamma(\boldsymbol{y}),
```

where $\sigma : \Gamma \to \mathbb{R}$ is an unknown density function, and ``G``
is the [fundamental
solution](https://en.wikipedia.org/wiki/Fundamental_solution) of the Laplace
equation. This *ansatz* is, by construction, an exact solution to the PDE.
Imposing the boundary condition on $\Gamma$ leads to the following integral equation:

```math
    \int_\Gamma G(\boldsymbol{x},\boldsymbol{y})\sigma(\boldsymbol{y}) \ \mathrm{d}\Gamma(\boldsymbol{y}) = g(\boldsymbol{x}) \quad \forall \boldsymbol{x} \in \Gamma,
```

Formulating the problem above in Inti.jl looks like this:

```@example lap2d
using Inti, LinearAlgebra, StaticArrays
# create a geometry
geo = Inti.parametric_curve(0, 1) do s
    return SVector(0.25, 0.0) +
            SVector(cos(2π * s) + 0.65 * cos(4π * s[1]) - 0.65, 1.5 * sin(2π * s))
end
Γ = Inti.Domain(geo)
# create a mesh and quadrature
msh = Inti.meshgen(Γ; meshsize = 0.1)
Q = Inti.Quadrature(msh; qorder = 5)
# create the integral operators
pde = Inti.Laplace(;dim=2)
S, _ = Inti.single_double_layer(;
    pde, 
    target = Q,
    source = Q,
    compression = (method = :none,),
    correction = (method = :dim,)
)
# manufacture an exact solution and take its trace on Γ
uₑ = x -> x[1] + x[2] + x[1]*x[2] + x[1]^2 - x[2]^2 + 0.5 * log(norm(x .- SVector(0.5, 1.5))) - 2 * log(norm(x .- SVector(-0.5, -1.5)))
g = map(q -> uₑ(q.coords), Q) # value at quad nodes
# solve for σ
σ = S \ g
# use the single-layer potential to evaluate the solution
𝒮, 𝒟 = Inti.single_double_layer_potential(; pde, source = Q)
uₕ = x -> 𝒮[σ](x)
```

The function `uₕ` is now a numerical approximation of the solution to the
Laplace equation, and can be evaluated at any point in the domain:

```@example lap2d
pt = SVector(0.5, 0.1)
println("Exact value at $pt:   ", uₑ(pt))
println("Approx. value at $pt: ", uₕ(pt))
```

If we care about the solution on the entire domain, we can visualize it using:

```@example lap2d
using Meshes, GLMakie # trigger the loading of some Inti extensions
xx = yy = range(-2, 2, length = 100)
fig = Figure()
inside = x -> Inti.isinside(x, Q) 
opts = (xlabel = "x", ylabel = "y", aspect = DataAspect())
ax1 = Axis(fig[1, 1]; title = "Exact solution", opts...)
h1 = heatmap!(ax1, xx,yy,(x, y) -> inside((x,y)) ? uₑ((x,y)) : NaN)
viz!(msh; segmentsize = 3)
cb = Colorbar(fig[1, 3], h1)
ax2 = Axis(fig[1, 2]; title = "Approx. solution", opts...)
h2 = heatmap!(ax2, xx,yy, (x, y) -> inside((x,y)) ? uₕ((x,y)) : NaN, colorrange = cb.limits[])
viz!(msh; segmentsize = 3)
fig # hide
```

!!! note "Formulation of the problem as an integral equation"
    Given a PDE and boundary conditions, there are often many ways to recast the
    problem as an integral equation, and the choice of formulation plays an
    important role in the unique solvability, efficiency, and accuracy of the
    numerical solution. Inti.jl provides a flexible framework for experimenting
    with different formulations, but it is up to the user to choose the most
    appropriate one for their problem.

While the example above is a simple one, Inti.jl can handle significantly more
complex problems involving multiple domains, heterogeneous coefficients,
vector-valued PDEs, three-dimensional geometries, and possibly requiring the use
of acceleration techniques such as the Fast Multipole Method. The best way to
dive deeper into Inti.jl's capabilities is the [tutorials](@ref "Getting
started") section. You can also find more advanced usage in the [examples](@ref
Examples) section.

## Contributing

## Acknowledgements
