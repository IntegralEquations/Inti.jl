# Solving an (interior/exterior) heat equation with Inti.jl

```@meta
CurrentModule = Inti
```

## Mathematical formulation

In this example we will solve the heat equation in a domain ``\Omega`` with Dirichlet
boundary conditions on ``\Gamma := \partial \Omega``:

```math
\begin{align}
	\partial_t u(x,t) - \Delta u(x,t)  &= f, \quad && x \in \Omega, \quad t \in [0,T]\\
	u(x,t) &= g(x,t) \quad && x \in \partial\Gamma, \quad t \in [0,T]\\
	u(x, t = 0) &= h \quad && x \in \Omega\\
\end{align}
```

Following these [lecture
notes](https://perso.univ-rennes1.fr/martin.costabel/publis/Co_ECM.pdf) by Costabel, we
first discretize time using an implicit scheme (backward Euler for simplicity):

```math
\begin{align}
u^{n} - \tau\Delta u^n  &= \tau f^n + u^{n-1}, \quad && \text{in } \Omega\\
u^n &= g^n, \quad && \text{on } \partial\Omega
\end{align}
```

where ``\tau`` is the time step and an ``n`` superscript denotes the function value at time
``t = n\tau`` (e.g. ``u^n(x) = u(x, t = n\tau)``). Dividing both sides by ``\tau`` we get a
Yukawa (or modified Helmholtz) equation for the function ``u^n``:

```math
\begin{align}
-\Delta u^n + \frac{1}{\tau}u^n  &= f^{n} + \frac{u^{n-1}}{\tau}, \quad && \text{in } \Omega\\
u^n &= g^n, \quad && \text{on } \partial\Omega
\end{align}\tag{3}
```

This is now amenable to an integral equation formulation. To do so, we follow the same
process as in the [Poisson problem tutorial](@ref), and split the solution into a particular
solution $u^n_p$ and a homogeneous solution $u^n_h$:

```math
u^n = u^n_p + u^n_h
```

The function ``u^n_p`` is given as a volume potential:

```math
u^n_p(x) = \int_{\Omega} G(x, y) \left( f^n(y) - \frac{u^{n-1}(y)}{\tau} \right) \;\mathrm{d}y \; \tag{5}.
```

whereas ``u^n_h`` satisfies the homogeneous problem

```math
\begin{align}
-\Delta u^n_h + \frac{1}{\tau} u^n_h &= 0,  \quad &&\text{in } \quad \Omega, \\
u^n_h &= g^n - u^n_p,  \quad &&\text{on } \; \partial\Omega,
\end{align}
```

which can be solved using a boundary integral equation formulation. For simplicity we use an
indirect double-layer formulation, where we seek for ``u^n_h`` in the form of

```math
u^n_h(r) = \mathcal{D}[\sigma](r), \quad r \in \Omega,
```

with ``\mathcal{D}`` the double-layer potential and ``\sigma`` the unknown density. Taking
the interior Dirichlet trace we get

```math
\frac{\sigma(x)}{2} + D[\sigma](x) = g^n(x) - u^n_p(x), \quad x \in \partial\Omega,
```

Putting it all together we get that the approximation solution ``u^n`` is given by

```math
\begin{align}
u^n(r) &= \int_{\Omega} G(r, y) \left( f^n(y) - \frac{u^{n-1}(y)}{\tau} \right) \;\mathrm{d}y + \mathcal{D}[\sigma](r), \quad r \in \Omega\\
\end{align}
```

## Discretization

We now proceed to discretize the problem.

### Mesh

We will use gmsh to generate the mesh:

```@example heat_equation
using Inti
using Gmsh
meshsize = 0.1
gmsh.initialize()
gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)
msh = Inti.import_mesh(; dim = 2)
gmsh.finalize()
```

Note that `msh` contains **all** the elements of the generated mesh, including the boundary
segments and any point entities that Gmsh may have created:

```@example heat_equation
msh
```

Since we will need to work with ``\Omega`` and ``\Gamma`` separately, we will extract those
domains and their corresponding (sub)meshes:

```@example heat_equation
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
Γ = Inti.boundary(Ω)
Ω_msh = msh[Ω]
Γ_msh = msh[Γ]
nothing # hide
```

Lets make sure things look right by plotting the mesh:

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

Ok, things look correct, so lets create the quadrature for ``\Omega`` and ``\Gamma``:

```@example heat_equation
Ω_quad = Inti.Quadrature(Ω_msh; qorder = 4)
Γ_quad = Inti.Quadrature(Γ_msh; qorder = 4)
@assert abs(area - π) < 1e-4 #hide
@assert abs(perimeter - 2π) < 1e-4 #hide
nothing # hide
```

### Integral Operators

We begin by defining the Yukawa operator:

```@example heat_equation
τ = meshsize
λ = 1/√(τ) 
op = Inti.Yukawa(; dim = 2, λ)
```

Next, we create a volume potential operator mapping values from ``\Omega`` to ``\Gamma``:

```@example heat_equation
correction = (method = :local, threads=false)
compression = (method = :none, )
V_d2b = Inti.volume_potential(;
	op,
	target = Ω_quad,
	source = Ω_quad,
	compression,
	correction,
)
```

This will help us compute the particular solution at the boundary. We also need to create
the double-layer operator mapping values from ``\Gamma`` to ``\Gamma``:

```@example heat_equation
_, D_b2b = Inti.single_double_layer(;
	op,
	target = Γ_quad,
	source = Γ_quad,
	compression,
	correction
)
D_b2b
```

Finally, to compute ``u^n``, we will need a volume potential mapping ``\Omega \to \Omega``
and the double-layer potential mapping ``\Gamma \to \Omega``:

```@example heat_equation
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
```

## 3. Plots and results

Timestep: $(if allow_run @bind t_plot PlutoUI.Slider(1:length(tsol.t),default=1,show_value=true) else "tick the checkbox to run the solver :)" end)

```@setup heat_equation.jl
if allow_run
	nodes = Inti.nodes(Ω_msh)
	u_quad = tsol[t_plot]
	u_nodes = Inti.quadrature_to_node_vals(Ω_quad, u_quad)

	colorrange = (0.0, 1.0) # extrema(u_nodes)
	fig_sol = Figure(; size = (800, 300))
	ax = Axis(fig_sol[1, 1]; aspect = DataAspect())
	viz!(Ω_msh; colorrange, color=u_nodes, interpolate = true, colormap = :jet)
	cb = Colorbar(fig_sol[1, 2]; label = "u", colorrange, colormap = :jet)
	fig_sol

	#= if t_plot == 1 
		colorrange = (0.0, extrema(u_nodes)[2])
		fig_sol = Figure(; size = (800, 300))
		ax = Axis(fig_sol[1, 1]; aspect = DataAspect())
		viz!(Ω_msh; colorrange, color=u_nodes, interpolate = true)
		cb = Colorbar(fig_sol[1, 2]; label = "u", colorrange)
		fig_sol
	else
		colorrange = extrema(u_nodes)
		fig_sol = Figure(; size = (800, 300))
		ax = Axis(fig_sol[1, 1]; aspect = DataAspect())
		viz!(Ω_msh; colorrange, color=u_nodes, interpolate = true)
		cb = Colorbar(fig_sol[1, 2]; label = "u", colorrange)
		fig_sol
	end =#
end
```

## 4. References

[a] _Time-dependent problems with the boundary integral equation method_, Martin Costabel, see [here](https://perso.univ-rennes1.fr/martin.costabel/publis/Co_ECM.pdf).

