# Helmholtz sound-soft scattering

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this tutorial"
    - Creating a geometry and a mesh using `Gmsh`'s API
    - Assembling integral operators and integral potential
    - Solving a boundary integral equation
    - Visualizing the solution

In this tutorial we will show how to solve an acoustic scattering problem in the
context of Helmholtz equation. We will begin with a simple example involving a
*smooth* sound-soft obstacle, and gradually make the problem more complex
considering a transmission problem. Along the way we will introduce the necessary
techniques used to handle some difficulties encountered. Assuming you
have already installed `Inti`, let us begin by importing the necessary
packages for this tutorial:

```@example helmholtz_scattering_2d
# Load required packages
using Inti
using LinearAlgebra
using StaticArrays
using CairoMakie
using Gmsh
```

We are now ready to begin solving some PDEs!

## Sound-soft scattering

The first example is that of a sound-soft acoustic scattering problem.
Mathematically, this means we will consider an exterior Helmholtz equation
(time-harmonic acoustics) with a Dirichlet boundary condition. More precisely,
let ``\Omega \subset \mathbb{R}^2`` be a bounded domain, and denote by ``\Gamma
= \partial \Omega`` its boundary. Then we wish to solve

```math
    \Delta u + k^2 u = 0 \quad \text{on} \quad \mathbb{R}^2 \setminus \bar{\Omega},
```

subject to Dirichlet boundary conditions on ``\Gamma``:

```math
    u(\boldsymbol{x}) = g(\boldsymbol{x}) \quad \text{for} \quad \boldsymbol{x} \in \Gamma.
```

Here ``g`` is the boundary datum, and ``k`` is a constant (the wavenumber).

For concreteness, we will take ``\Gamma`` to be a disk, and focus on
the *plane-wave scattering* problem. This means we will seek a solution ``u`` of
the form ``u = u_s + u_i``, where ``u_i`` is a known incident field, and ``u_s``
is the scattered field we wish to compute. Using the theory of boundary integral
equations, we can express ``u_s`` as

```math
    u_s(\boldsymbol{r}) = \mathcal{D}[\sigma](\boldsymbol{r}) - i k \mathcal{S}[\sigma](\boldsymbol{r}),
```

where ``\mathcal{S}`` is the so-called single layer potential, ``\mathcal{D}``
is the double-layer potential, and ``\sigma : \Gamma \to \mathbb{C}`` is a
surface density. This is an indirect formulation (because ``\sigma`` is an
*auxiliary* density, not necessarily physical) commonly referred to as a
*combined field formulation*. Taking the limit ``\mathbb{R}^2 \setminus \bar
\Omega \ni x \to \Gamma``, it can be shown that the following equation holds on
``\Gamma``:

```math
    \left( \frac{\mathrm{I}}{2} + \mathrm{D} - i k \mathrm{S} \right)[\sigma] = g,
```

where $\mathrm{I}$ is the identity operator, and $\mathrm{S}$ and $\mathrm{D}$ are the single- and double-layer operators. This is the **combined field integral equation** that we will solve next.

As for the boundary data ``g``, using the sound-soft condition (i.e. ``u=0`` on
the scatterer), it follows that ``u_s = -u_i`` on ``\Gamma``. We are now in a
position to solve the problem! Let us begin by creating the **geometry** and its
corresponding **mesh**:

```@example helmholtz_scattering_2d
h = 0.1
r = 1
center = SVector(0,0)
gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMax", h)
# set gmsh verbosity to 2
gmsh.option.setNumber("General.Verbosity", 2)
gmsh.model.mesh.setOrder(2)
Inti.clear_entities!()
gmsh.model.occ.addDisk(center[1], center[2], 0, r, r)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
Ω = Inti.gmsh_import_domain(; dim=2)
msh = Inti.gmsh_import_mesh(Ω; dim=2)
gmsh.finalize()
```

Note that `msh` contains the mesh of both `Ω` and its boundary `Γ`.

!!! tip "Geometry creation"
    For simple shapes, it is convenient to directly use the `gmsh` API, as
    illustrated above. As things get more complex, however, it is
    preferable to use the `gmsh` UI to create your shapes, save a `.geo` or
    `.msh` file, and then import the file using [`Inti.gmsh_read_msh`](@ref).

Because we will be employing a Nyström method, we must create a quadrature for
$\Gamma$. This is done by calling the [`Quadrature`](@ref)
constructor on a mesh object:

```@example helmholtz_scattering_2d
Γ = Inti.external_boundary(Ω)
Q = Inti.Quadrature(view(msh,Γ); qorder=5)
```

With the `Quadrature` constructed, we now can define the integral operators
``\mathrm{S}`` and ``\mathrm{D}`` associated with the *Helmholtz* equation:

```@example helmholtz_scattering_2d
k = 2π
pde = Inti.Helmholtz(;dim=2,k)
G   = Inti.SingleLayerKernel(pde)
dG  = Inti.DoubleLayerKernel(pde)
Sop = Inti.IntegralOperator(G,Q,Q)
Dop = Inti.IntegralOperator(dG,Q,Q)
```

Both `Sop` and `Dop` are of type [`IntegralOperator`](@ref), which is a subtype
of `AbstractMatrix`. They represent a discrete approximation to linear operators
mapping densities defined on source surface into densities defined on a target
surface. There are two well-known difficulties related to the discretization of
these `IntegralOperator`s:

- The kernel of the integral operator is not smooth, and thus specialized
quadrature rules are required to accurately approximate the matrix entries for
which the target and source point lie *close* (relative to some scale) to each
other.
- The underlying matrix is dense, and thus the storage and computational cost of
the operator is prohibitive for large problems unless acceleration techniques
such as *Fast Multipole Methods* or *Hierarchical Matrices* are employed.  

In this example, we will use a sparse correction method based on the
*density interpolation technique*, and ignore the second issue. More precisely,
because the problem is two-dimensional and simple, we will just assemble a dense
`Matrix` to represent the integral operator.

We first build the *dense* part of the operators:

```@example helmholtz_scattering_2d
S₀ = Matrix(Sop)
D₀ = Matrix(Dop)
```

Next we build the sparse corrections

```@example helmholtz_scattering_2d
δS, δD = Inti.bdim_correction(pde,Q,Q,S₀,D₀)
```

We can now add the corrections to the dense part to obtain the final operators:

```@example helmholtz_scattering_2d
S = S₀ + δS
D = D₀ + δD
```

Finally, we are ready to solve the scattering problem:

```@example helmholtz_scattering_2d
# the linear operator
L = 0.5*I + D - im*k*S
# incident wave
θ  = 0*π/4
d⃗  = (cos(θ),sin(θ))
uᵢ  = x -> exp(im*k*dot(d⃗,x))
rhs = [-uᵢ(q.coords) for q in Q]
σ = L \ rhs
```

We can now reconstruct the solution using the `IntegralPotential` representation:

```@example helmholtz_scattering_2d
𝒮 = Inti.IntegralPotential(G,Q)
𝒟 = Inti.IntegralPotential(dG,Q)
u = x -> 𝒟[σ](x) - im*k*𝒮[σ](x) + uᵢ(x)
```

Note that, albeit not very efficient, the function `u` can be evaluated at any
point. To see what the solution looks like, let's plot it on a grid

```@example helmholtz_scattering_2d
fig = Figure()
ax  = Axis(fig[1,1];xgridvisible=false,ygridvisible=false)
h = 2π/k/10 # 20 pts per wavelength
x = y = -5:h:5
# U = [Inti.isinside((x,y),Q) ? NaN + NaN*im : uᵢ((x,y)) for x in x, y in y]
U = [norm(SVector(x,y) - center) < r ? NaN + NaN*im : u((x,y)) for x in x, y in y]
hm = heatmap!(ax,x,y,real(U),interpolate=true,colorrange=(-1.0,1.0))
cm = Colorbar(fig[1, 2],hm; label="Re(u)")
# plot the geometry
lines!(ax,view(msh,Γ),linewidth=5.0,color=:black)
# some final tweak before displaying the figure
colsize!(fig.layout, 1, Aspect(1, 1.0))
resize_to_layout!(fig)
fig
```
