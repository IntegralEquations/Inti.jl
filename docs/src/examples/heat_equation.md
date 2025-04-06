# Heat equation

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this example"
	- Using Rothe's method to solve a time-dependent PDE
	- Combining volume and boundary integral equations
	- Animating the solution in time using `Makie`

## Problem description

In this example we will solve the heat equation in a domain ``\Omega`` with Dirichlet
boundary conditions on ``\Gamma := \partial \Omega``:

```math
\begin{align*}
	\partial_t u(x,t) - \Delta u(x,t)  &= f(x,t,u), \quad && x \in \Omega, \quad t \in [0,T]\\
	u(x,t) &= g(x,t) \quad && x \in \partial\Gamma, \quad t \in [0,T]\\
	u(x, t = 0) &= u_0(x) \quad && x \in \Omega\\ \tag{Heat equation}
\end{align*}
```

where $f$ is a source term, $g$ is the Dirichlet boundary condition, and $u_0$ is the
initial condition (all assumed to be given).

The heat equation is a parabolic PDE, and we will solve it using Rothe's method, which
reduces the problem to a sequence of elliptic PDEs. The main idea is to discretize the time
variable using an implicit scheme, and then solve the resulting elliptic PDEs using integral
equations. This is the opposite of the methods of lines, where we would discretize the
spatial variable first and then solve the resulting ODEs. Let's get started.

## Time discretization (Rothe's method)

Following these [lecture
notes](https://perso.univ-rennes1.fr/martin.costabel/publis/Co_ECM.pdf) by Costabel, we
first discretize time using an implicit scheme (backward Euler for simplicity):

```math
\begin{align*}
u^{n} - \tau\Delta u^n  &= \tau f^{n-1} + u^{n-1}, \quad && \text{in } \Omega\\
u^n &= g^n, \quad && \text{on } \partial\Omega
\end{align*}
```

where ``\tau`` is the time step and an ``n`` superscript denotes the function value at time
``t = n\tau`` (e.g. ``u^n(x) = u(x, t = n\tau)``). Dividing both sides by ``\tau`` we get a
Yukawa (or modified Helmholtz) equation for the function ``u^n``:

```math
\begin{align*}
-\Delta u^n + \frac{1}{\tau}u^n  &= f^{n-1} + \frac{u^{n-1}}{\tau}, \quad && \text{in } \Omega\\
u^n &= g^n, \quad && \text{on } \partial\Omega \tag{Yukawa equation}
\end{align*}
```

This is now amenable to an integral equation formulation. To do so, we follow the same
process as in the [Poisson problem tutorial](@ref "Poisson Problem"), and split the solution
into a particular solution $u^n_p$ and a homogeneous solution $u^n_h$:

```math
u^n = u^n_p + u^n_h
```

The function ``u^n_p`` is given as a volume potential:

```math
u^n_p(x) = \int_{\Omega} G(x, y) \left( f(y,t^{n-1}, u^{n-1}) + \frac{u^{n-1}(y)}{\tau} \right) \; \mathrm{d}\Omega(y) \tag{Particular solution}
```

As for the homogeneous solution ``u^n_h``, it is easy to see that it satisfies

```math
\begin{align*}
-\Delta u^n_h + \frac{1}{\tau} u^n_h &= 0,  \quad &&\text{in } \quad \Omega, \\
u^n_h &= g^n - u^n_p,  \quad &&\text{on } \; \partial\Omega, \tag{Homogenous problem}
\end{align*}
```

which can be solved using a boundary integral equation formulation. We will use here an
indirect double-layer formulation, where we seek for ``u^n_h`` in the form of

```math
u^n_h(r) = \mathcal{D}[\sigma](r), \quad r \in \Omega,
```

with ``\mathcal{D}`` the double-layer potential associated with the Yukawa equation, and
``\sigma`` the unknown density. Taking the interior Dirichlet trace (for a smooth boundary)
yields

```math
\frac{-\sigma(x)}{2} + D[\sigma](x) = g^n(x) - u^n_p(x), \quad x \in \partial\Omega, \tag{BIE}
```

The steps outlined above reduce the time-dependent heat equation to a sequence forced Yukawa
equations which can be solved using the same techniques as in the [Poisson problem
tutorial](@ref), as illustrated next.

## Spatial discretization

We now proceed to discretize the inhomogeneous Yukawa equation. We will use Gmsh to create a
disk with a few holes in it, and then use the Yukawa operator to solve the problem.

### Mesh

```@example heat_equation
using Inti
using Gmsh
function create_mesh(meshsize, meshorder=2)
	gmsh.initialize()
	gmsh.option.setNumber("General.Verbosity", 2)
	gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
	disk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
	ell1 = gmsh.model.occ.addDisk(.5, 0, 0, 0.2, 0.1)
	ell2 = gmsh.model.occ.addDisk(-.5, 0.3, 0, 0.3, 0.2)
	gmsh.model.occ.rotate([(2,ell2)], 0, 0, 0, 0, 0, 1, -π/3)
	ell3 = gmsh.model.occ.addDisk(-.5, -0.5, 0, 0.2, 0.15)
	gmsh.model.occ.rotate([(2,ell3)], 0, 0, 0, 0, 0, 1, π/3)
	gmsh.model.occ.cut([(2,disk)], [(2,ell1), (2,ell2), (2,ell3)])
	gmsh.model.occ.synchronize()
	gmsh.model.mesh.generate(2)
	gmsh.model.mesh.setOrder(meshorder)
	msh = Inti.import_mesh(; dim = 2)
	gmsh.finalize()
	return msh
end
meshsize = 0.1
meshorder = 2
tau       = 2π / 40
msh = create_mesh(meshsize, meshorder)
```

!!! note "Mesh structure"
	Note that `msh` contains **all** the elements of the generated mesh, including the
	boundary segments and any point entities that Gmsh may have created. To properly index
	into our mesh elements, we must use `Domain`s, which are simply a collection of
	geometric entities.

Since we will need to work with ``\Omega`` and ``\Gamma`` separately, we will extract those
domains and their corresponding (sub)meshes:

```@example heat_equation
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
Γ = Inti.boundary(Ω)
Ω_msh = @views msh[Ω]
Γ_msh = @views msh[Γ]
nothing # hide
```

Lets make sure things look right by plotting the mesh of ``\Omega`` and ``\Gamma``:

```@example heat_equation
using Meshes
using GLMakie
fig = viz(
    Ω_msh;
    segmentsize = 1,
    showsegments = true,
    axis = (aspect = DataAspect(),),
    figure = (; size = (500, 400)),
)
viz!(Γ_msh; color = :red, segmentsize = 4)
fig # hide
```

Ok, things look correct, so let's create the quadrature for ``\Omega`` and ``\Gamma``:

```@example heat_equation
Ω_quad = Inti.Quadrature(Ω_msh; qorder = 3)
Γ_quad = Inti.Quadrature(Γ_msh; qorder = 3)
nothing # hide
```

### Integral Operators

We begin by defining the Yukawa operator:

```@example heat_equation
λ = 1/sqrt(tau)
op = Inti.Yukawa(; dim = 2, λ)
```

Next we will define the various operators that we need. These include:

- ``\mathcal{V}_{\Omega \to \Omega}`` the volume potential mapping values from
  ``\Omega`` to ``\Omega``, used to compute the particular solution at the interior
- ``\mathcal{V}_{\Omega \to \Gamma}`` the volume potential mapping values from
  ``\Omega`` to ``\Gamma``, used to compute the trace of the particular solution at the boundary
- ``\mathcal{D}_{\Gamma \to \Gamma}`` the double-layer operator mapping values from
  ``\Gamma`` to ``\Gamma``, used to solve for the density in the homogeneous problem
- ``\mathcal{D}_{\Gamma \to \Omega}`` the double-layer potential mapping values from
  ``\Gamma`` to ``\Omega``, used to compute the homogeneous solution at the interior

For each operator, we will need to specify the [compression](@ref "Compression methods") and
[correction methods](@ref "Correction methods") (see their documentation for more details). We
will pick some reasonable defaults here:

```@example heat_equation
# using HMatrices
# compression = (method = :hmatrix, atol = 1e-8)
correction = (
    method = :adaptive,
    threads = true,
    maxdist = 3 * meshsize,
    atol = 1e-6,
    maxsubdiv = 10_000,
)
compression = (method = :none,)
nothing # hide
```

Let us create the required operators:

```@example heat_equation
V_d2b = Inti.volume_potential(;
	op,
	target = Γ_quad,
	source = Ω_quad,
	compression,
	correction,
)
_, D_b2b = Inti.single_double_layer(;
	op,
	target = Γ_quad,
	source = Γ_quad,
	compression,
	correction
)
V_d2d = Inti.volume_potential(;
	op,
	target = Ω_quad,
	source = Ω_quad,
	compression,
	correction,
)
_, D_b2d = Inti.single_double_layer(;
	op,
	target = Ω_quad,
	source = Γ_quad,
	compression,
	correction
)
nothing # hide
```

We can now put it all together and evolve our solution in time!

## Solving the problem

Since the domain is fixed, it makes sense to precompute a factorization of our boundary
integral operator. This will save us some time in the loop:

```@example heat_equation
using LinearAlgebra
L = -I/2 + D_b2b
F = lu(L)
nothing # hide
```

We need to decide on our source term ``f``, boundary values ``g``, and initial condition. To
validate the solution, we will first use the method of manufactured solutions, where we
begin with a known solution ``uₑ(x, t)`` and compute the source term ``f`` and boundary values
``g``:

```@example heat_equation
uₑ = (x, t) -> sin(x[1]) * sin(x[2]) * cos(t)
f  = (x, t) -> 2 * uₑ(x, t) - sin(t) * sin(x[1]) * sin(x[2])
g = (x, t) -> uₑ(x, t) # boundary values
nothing # hide
```

We will evolve the solution for a full period of the sine function, i.e. ``T = 2\pi``.

```@example heat_equation
nsteps = round(Int, 2π/tau)
uⁿ⁻¹ = map(q -> uₑ(q.coords, 0), Ω_quad) # initial condition
uⁿ   = zero(uⁿ⁻¹)
t    = Ref(0.0)
for n in 1:nsteps
	fⁿ  = map(q -> f(q.coords, t[] + tau), Ω_quad)
	uₚⁿ = V_d2d*(fⁿ + uⁿ⁻¹/tau)
	gⁿ  = map(q -> g(q.coords, t[] + tau), Γ_quad)
	uₕⁿ = D_b2d * (F \ (gⁿ - V_d2b * (fⁿ + uⁿ⁻¹ / tau)))
	uⁿ .= uₚⁿ + uₕⁿ
	uⁿ⁻¹ .= uⁿ
	t[] += tau
end
```

We can now check the error between the computed solution and the exact solution:

```@example heat_equation
uref = map(q -> uₑ(q.coords, t[]), Ω_quad) # reference solution
er = norm(uⁿ - uref, Inf) / norm(uref, Inf)
@assert er < 1e-2 # hide
er
```

It seems like we are getting a reasonable error, so our implementation is (probably)
correct! Let's move on to a more interesting example next, and actually visualize the
solution.

!!! note "Convergence"
	Testing that the error is "small", as done above, is a rather rudimentary and
	qualitative way to check the correctness of our implementation. For a more careful
	analysis, we should check the *convergence order* as we refine the spatial and/or the
	temporal discretization. If we do so for this simple example, we should find that error
	is mostly dominated by the temporal discretization, which is first order due to the use
	of the backward Euler scheme. You can check that e.g. by halving the time step the error
	above should be roughly halved!

For a more interesting application, let us consider the problem where the temperature given
by ``0`` (in some unspecified units) at the outer boundary, and by ``sin(t)^2`` at the boundary of
the inclusions. We will take a ``f=0`` and ``u_0 = 0`` for simplicity:

```@example heat_equation
f  = (x, t) -> 0.0
g  = (x, t) -> norm(x) > 0.9 ? 0.0 : (sin(t))^2
u⁰ = zeros(length(Ω_quad))
nothing # hide
```

Here is what the solution looks like over time:

```@example heat_equation
fig = Figure()
ax = Axis(fig[1, 1]; aspect = DataAspect())
colorrange = (0.0, 1.0)
record(fig, joinpath(@__DIR__,"heat.gif")) do io
	nsteps = round(Int, 2π/tau)
	uⁿ⁻¹ = u⁰
	uⁿ   = zero(uⁿ⁻¹)
	t    = 0.0
	for n in 1:nsteps
		u_nodes   = Inti.quadrature_to_node_vals(Ω_quad, uⁿ⁻¹)
		ax.title = "t = $(round(t, digits = 2))"
		viz!(Ω_quad.mesh; showsegments = true, color = u_nodes, colorrange)
		viz!(Γ_msh; color = :black, segmentsize = 4)
		Colorbar(fig[1, 2]; colorrange = colorrange)
		recordframe!(io)
		fⁿ  = map(q -> f(q.coords, t + tau), Ω_quad)
		uₚⁿ = V_d2d*(fⁿ + uⁿ⁻¹/tau)
		gⁿ  = map(q -> g(q.coords, t + tau), Γ_quad)
		uₕⁿ = D_b2d * (F \ (gⁿ - V_d2b * (fⁿ + uⁿ⁻¹ / tau)))
		uⁿ = uₚⁿ + uₕⁿ
		uⁿ⁻¹ = uⁿ
		t += tau
	end
end
nothing # hide
```

![Heat equation](heat.gif)

!!! tip "Going further"
	- Second order in time scheme for improved accuracy
	- Use of compression method such as `HMatrix`
	- More complex domains
	- Nonlinear source term
