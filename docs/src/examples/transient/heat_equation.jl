### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f4a0859a-6b96-11ef-0c3b-7b7ba5c458ce
begin
	using Pkg, Revise

	Pkg.activate(joinpath(@__DIR__, "Project.toml"))

	using Inti # dev version as dependency
	using Meshes, Gmsh, CairoMakie, PlutoUI, LinearAlgebra, RecursiveArrayTools
end

# ╔═╡ 95e65f76-f1ec-434d-8c1c-c314b2eca7c3
using HypertextLiteral

# ╔═╡ c0df4cc3-55cf-4f0c-a5a9-7e912724bb11
md"""
# Solving an (interior/exterior) heat equation with Inti.jl
"""

# ╔═╡ 9db5ce89-9a3b-41b7-bf46-2c7327dc0f2a
md"""
## 1. Model: Mathematical formulation
"""

# ╔═╡ 64fe3119-aa5a-4b82-8562-872de48d11b6
md"""$\begin{equation} \begin{cases}
\partial_t u(t,x) - \Delta u(t,x)  &= 0 \quad &\text{in } \Omega\\[0.3cm]
\hspace{3.6cm} u &= g &\text{on } \partial\Omega\\[0.3cm]
\hspace{2.7cm} u(0,x) &= 0 \quad & x \in \Omega\\
\end{cases}\tag{1}\end{equation}$ 
"""

# ╔═╡ 9730531f-8b7b-4bdb-af50-3a76ff781e69
md"""
Following [a], the system $(1)$ can be reformulated as an elliptic BVP which can be solve __one step at time__, we have for $n= 1, \dots, N$

$\begin{equation} \begin{cases}
u^n - \tau\Delta u^n  &= u^{n-1} \quad &\text{in } \Omega\\[0.3cm]
\hspace{1.6cm} u^n &= g^n &\text{on } \partial\Omega
\end{cases}\tag{2}\end{equation}$ 
"""

# ╔═╡ fc0e5d68-309e-471d-a99c-3966f6789803
md"""
We can rewrite $(2)$ to resemble the Helmholtz equation and obtain

$\begin{equation} \begin{cases}
\Delta u + k^2u  &= f \quad &\text{in } \Omega\\[0.3cm]
\hspace{1.6cm} u &= g &\text{on } \partial\Omega
\end{cases}\tag{3}\end{equation}$
where $k = \frac{i}{\sqrt\tau}$, $u = u^n$, $f = -\frac{1}{\tau}u^{n-1}$.
"""

# ╔═╡ 7344c3a8-52b3-4a4d-baf8-9d1c58d3e3a2
md"""
To deal with the inhomogeneous PDE $(3)$, we follow the same process as in the Poisson problem tutorial. We split the solution into a particular solution $u_p$ and a homogeneous solution $u_h$:
```math
u = u_p + u_h \;\tag{4}.
```
The function ``u_p`` is given as a volume potential:
```math
u_p(x) = \int_{\Omega} G(x, y) f(y) \;\mathrm{d}y \; \tag{5}.
```
"""

# ╔═╡ a20158e0-ad00-4af0-826f-0dc64073b993
md"""
The function ``u_h`` satisfies the homogeneous problem

```math
\begin{align}
	\Delta u_h + k^2u_h &= 0,  \quad &&\text{in } \quad \Omega, \\
	u_h &= g - u_p,  \quad &&\text{on } \; \partial\Omega,
\end{align}\tag{6}
```
which can be solved using BIE method.
"""

# ╔═╡ deffe780-72ed-4c84-b8e5-6f59316a16d3
md"""
One possible formulation is the following:

```math
\left(\frac{I}{2} + D - ikS\right)[\sigma] = g - u_p \;\; \tag{7},
```
where $S$ and $D$ are the single- and double-layer operators and $\sigma$ the density function.

Or look at one of the formulations from Costabel [a].
"""

# ╔═╡ 89d4ca3b-e3c7-4519-917e-706a0fd6aea0
md"""
## 2. Discretization
"""

# ╔═╡ 6029fb91-1263-4c7c-8d13-fd220a00a90c
md"""
### 2.1 Geometry and mesh
"""

# ╔═╡ 9e57c112-6328-4179-ab0b-5953978f4bd4
begin
	meshsize = 0.1
	gmsh.initialize()
	circle = Inti.gmsh_curve(0, 2π; meshsize) do s
	    return Inti.Point2D(cos(s), sin(s))
	end
	cl = gmsh.model.occ.addCurveLoop([circle])
	surf = gmsh.model.occ.addPlaneSurface([cl])
	gmsh.model.occ.synchronize()
	gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
	gmsh.model.mesh.generate(2)
	gmsh.model.mesh.setOrder(2)
	msh = Inti.import_mesh(; dim = 2)
	gmsh.finalize()
end

# ╔═╡ 6e8f8c46-6902-4a88-8423-ae2046467ea6
begin
	Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
	Γ = Inti.boundary(Ω)
	Ω_msh = view(msh, Ω)
	Γ_msh = view(msh, Γ)
end

# ╔═╡ 9ab4251b-a061-44f3-b8d1-1ec9133832ee
begin
	fig = viz(Ω_msh ; segmentsize=1, showsegments=true, axis=(aspect=DataAspect(),), figure=(;size=(500,400)))
	viz!(Γ_msh; color=:red)
	fig
end

# ╔═╡ c5811230-dcbf-403b-bb94-9f056cdb92bb
md"""
### 2.2 Integral Operators
"""

# ╔═╡ baefc33b-9f0c-4caf-b77d-a0dc118e911e
begin
	Ω_quad = Inti.Quadrature(Ω_msh; qorder=4)
	Γ_quad = Inti.Quadrature(Γ_msh; qorder=4)
end

# ╔═╡ 1dd4afd5-4383-438b-8a4a-5b2b5cbbea75


# ╔═╡ 218e2978-5388-4aec-a48f-73bc450145a3
begin
	# Initial Condition
	u₀(x) = 1.0
	f = map(q -> u₀(q.coords), Ω_quad)
	# Boundary Conditions
	bc(x) = 0.0 # consider making it time dependent for future tests
	g = map(q -> bc(q.coords), Γ_quad)
end

# ╔═╡ 5529e3aa-cabb-4714-93e1-138b0e7431f9
function main(Ω_quad, Γ_quad, f, g ; t_start=0.0, t_end=1.0, τ=0.05)

	k = im/τ
	pde = Inti.Helmholtz(; dim = 2, k=k)
	
	## Initialize Volume Potential, single and double layer operators for solving the problem and then for solution evaluation
	
	# Integral operator to build the LSE 
	V_d2b = Inti.volume_potential(;
		pde,
		target = Γ_quad,
		source = Ω_quad,
		compression = (method = :none,),
		correction = ( method = :none,),
	)
	S, D = Inti.single_double_layer(;
		pde,
		target = Γ_quad,
		source = Γ_quad,
		compression = (method = :none,),
		correction = (method = :dim,),
	)
	# For the solution evaluation
	V_d2d = Inti.volume_potential(;
	    pde,
	    target = Ω_quad,
	    source = Ω_quad,
	    compression = (method = :none,),
	    correction = (method = :dim,),
	)
	S_b2d, D_b2d = Inti.single_double_layer(;
	    pde,
	    target = Ω_quad,
	    source = Γ_quad,
	    compression = (method = :none,),
	    correction = (method = :none,),
	)

	times, sol_u = [t_start], [f]

	for i in 1:floor(Int64,(t_end - t_start)/τ)
		
		L = I / 2 + D - im * k * S	
		rhs = g - V_d2b*f
	
		σ = L \ rhs

		# Solution Evaluation (to improve with compression and correction methods)
		f = D_b2d*σ - im*k*S_b2d*σ + V_d2d*f # f = u^{n} -> used for the next iteration as source term of the PDE
		
		push!(sol_u, real(f)) # Taking the real part of the solution (im(f)≈0)
		push!(times, τ*i)
	end

	DiffEqArray(sol_u, times)	
end

# ╔═╡ 0f5436a7-4eac-4bd9-aca4-15b1f9d811b8
md"""
 __Run next cell__: $(@bind allow_run PlutoUI.CheckBox(default=false))
"""

# ╔═╡ 79527886-824d-4cb7-bee7-00041a6151b6
if allow_run
	tsol = main(Ω_quad, Γ_quad, f, g)
end

# ╔═╡ 825a4196-8cfe-4a71-bb4a-34d9f445afd7
md"""
## 3. Plots and results
"""

# ╔═╡ 7f2aaacb-e159-4e24-8ed3-a9237287dd6b
md"""
Timestep: $(if allow_run @bind t_plot PlutoUI.Slider(1:length(tsol.t),default=1,show_value=true) else "tick the checkbox to run the solver :)" end)
"""

# ╔═╡ 5ae1b44d-98cd-497b-b187-638ed42bad26
if allow_run
	nodes = Inti.nodes(Ω_msh)
	u_quad = tsol[t_plot]
	u_nodes = Inti.quadrature_to_node_vals(Ω_quad, u_quad)

	if t_plot == 1 
		colorrange = (0.0, f[1])
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
	end
end

# ╔═╡ c39c9560-b050-4fe4-9cb0-376fe1ffc532
md"""
## 4. References
"""

# ╔═╡ b6ec9ab6-ab4f-4d26-b2b0-9256621e472d
md"""
[a] _Time-dependent problems with the boundary integral equation method_, Martin Costabel, see [here](https://perso.univ-rennes1.fr/martin.costabel/publis/Co_ECM.pdf).
"""

# ╔═╡ 8182287b-f56b-40bd-a24d-3fb69409d9a7
begin
    highlight(mdstring,color)= htl"""<blockquote style="padding: 10px; background-color: $(color);">$(mdstring)</blockquote>"""
	
	macro important_str(s)	:(highlight(Markdown.parse($s),"#ffcccc")) end
	macro definition_str(s)	:(highlight(Markdown.parse($s),"#ccccff")) end
	macro statement_str(s)	:(highlight(Markdown.parse($s),"#ccffcc")) end
		
		
    html"""
    <style>
     h1{background-color:#dddddd;  padding: 10px;}
     h2{background-color:#e7e7e7;  padding: 10px;}
     h3{background-color:#eeeeee;  padding: 10px;}
     h4{background-color:#f7f7f7;  padding: 10px;}
    </style>
"""
end

# ╔═╡ cec3125d-5b90-42b5-8485-7926dbbac91c
TableOfContents()

# ╔═╡ Cell order:
# ╠═f4a0859a-6b96-11ef-0c3b-7b7ba5c458ce
# ╟─c0df4cc3-55cf-4f0c-a5a9-7e912724bb11
# ╟─9db5ce89-9a3b-41b7-bf46-2c7327dc0f2a
# ╟─64fe3119-aa5a-4b82-8562-872de48d11b6
# ╟─9730531f-8b7b-4bdb-af50-3a76ff781e69
# ╟─fc0e5d68-309e-471d-a99c-3966f6789803
# ╟─7344c3a8-52b3-4a4d-baf8-9d1c58d3e3a2
# ╟─a20158e0-ad00-4af0-826f-0dc64073b993
# ╟─deffe780-72ed-4c84-b8e5-6f59316a16d3
# ╟─89d4ca3b-e3c7-4519-917e-706a0fd6aea0
# ╟─6029fb91-1263-4c7c-8d13-fd220a00a90c
# ╟─9e57c112-6328-4179-ab0b-5953978f4bd4
# ╟─6e8f8c46-6902-4a88-8423-ae2046467ea6
# ╠═9ab4251b-a061-44f3-b8d1-1ec9133832ee
# ╟─c5811230-dcbf-403b-bb94-9f056cdb92bb
# ╠═baefc33b-9f0c-4caf-b77d-a0dc118e911e
# ╟─1dd4afd5-4383-438b-8a4a-5b2b5cbbea75
# ╠═218e2978-5388-4aec-a48f-73bc450145a3
# ╟─5529e3aa-cabb-4714-93e1-138b0e7431f9
# ╟─0f5436a7-4eac-4bd9-aca4-15b1f9d811b8
# ╠═79527886-824d-4cb7-bee7-00041a6151b6
# ╟─825a4196-8cfe-4a71-bb4a-34d9f445afd7
# ╟─7f2aaacb-e159-4e24-8ed3-a9237287dd6b
# ╟─5ae1b44d-98cd-497b-b187-638ed42bad26
# ╟─c39c9560-b050-4fe4-9cb0-376fe1ffc532
# ╟─b6ec9ab6-ab4f-4d26-b2b0-9256621e472d
# ╟─95e65f76-f1ec-434d-8c1c-c314b2eca7c3
# ╟─8182287b-f56b-40bd-a24d-3fb69409d9a7
# ╟─cec3125d-5b90-42b5-8485-7926dbbac91c
