using Pkg
Pkg.activate(@__DIR__)

using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using GLMakie

function create_disk_shaped_crack(meshsize, radius_x, radius_y)
	gmsh.initialize()
	gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
	gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
	# gmsh.option.setNumber("Mesh.Algorithm", 11)
	gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius_x, radius_y)
	gmsh.model.occ.synchronize()
	gmsh.model.mesh.generate(2)
	gmsh.model.mesh.recombine()
	msh = Inti.import_mesh(; dim = 3)
	# gmsh.fltk.run()
	gmsh.finalize()
	return msh
end

function _displacement_jump_function(x)
	r = sqrt(x[1]^2 + x[2]^2)
	ν = λ / (2 * (λ + μ))
	E = μ * (3 * λ + 2 * μ) / (λ + μ)
	G = E / (2 * (1 + ν))
	return SVector(0, 0, 4 * (1 - ν) / (π * G) * sqrt(abs(1 - r^2)))
end

function display_displacement_jump(Γ_quad, φ)
	X = [node.coords[1] for node in Γ_quad]
	Y = [node.coords[2] for node in Γ_quad]
	global_mesh_size = abs(maximum(X) - minimum(X))
	Z₁⁺ = φ
	fig = Figure()
	ax = Axis3(fig[1, 1])
	meshscatter!(ax, X, Y, Z₁⁺; markersize = 0.01)
	# meshscatter!(ax, X, Y, Z₁⁻; markersize = 0.01)
	GLMakie.xlims!(ax, -global_mesh_size, global_mesh_size)
	GLMakie.ylims!(ax, -global_mesh_size, global_mesh_size)
	# GLMakie.zlims!(ax, -global_mesh_size, global_mesh_size)

	return display(fig)
end

# custom kernel for a disk shaped crack defined as K * sqrt(d)

function _kernel_correction_term(target, source)::Float64
	x, y = Inti.coords(target), Inti.coords(source)
	correction = sqrt(abs(radius_x - sqrt((y[1]^2 + y[2]^2))))
	return correction
end

function _custom_kernel(target, source)::SMatrix{3, 3, Float64, 9}
	x, y = Inti.coords(target), Inti.coords(source)
	correction = _kernel_correction_term(target, source)
	return correction * T(target, source)
end

### STRESS INTENSITY FACTOR POST-PROCESSING

function get_nearest_boundary_nodes(Γ_quad, ∂Γ_msh)
	global_mesh_size = abs(maximum([node.coords[1] for node in Γ_quad]) - minimum([node.coords[1] for node in Γ_quad]))
	X = [node.coords for node in Γ_quad]
	tol_d = global_mesh_size
	d = Dict{DataType, Vector{Int64}}()
	for E in Inti.element_types(∂Γ_msh)
		nearlist_E = Inti.near_interaction_list(X, ∂Γ_msh; tol = tol_d)[E]
		nearlist_E_new = Int[]
		for (el_id, element) in enumerate(Inti.elements(∂Γ_msh, E))
			nearlist = nearlist_E[el_id]
			while length(nearlist) > 1 && length(Inti.near_interaction_list(X, ∂Γ_msh; tol = 0.5 * tol_d)[E][el_id]) > 0
				tol_d = 0.5 * tol_d
				temp = Inti.near_interaction_list(X, ∂Γ_msh; tol = tol_d)[E]
				nearlist = temp[el_id]
			end
			nearest = nearlist[1]
			push!(nearlist_E_new, nearest)
			tol_d = global_mesh_size
		end
		d[E] = nearlist_E_new
	end
	return d
end

function stress_intensity_factor(Γ_quad, ∂Γ_msh, ψ)
	ν = λ / (2(λ + μ))
	# ψ_block_array = Inti.BlockArray{SVector{3, Float64}}(ψ)
	d = get_nearest_boundary_nodes(Γ_quad, ∂Γ_msh)
	Kₛ = Float64[]
	X = collect(Inti.nodes(∂Γ_msh))
	points_on_boundary = collect(Inti.nodes(∂Γ_msh))
	for E in Inti.element_types(∂Γ_msh)
		nearlist = d[E]
		for node in nearlist
			K_I = μ / (4 * (1 - ν)) * sqrt(2 * π) * ψ[node][3]
			push!(Kₛ, K_I)
		end
	end
	return Kₛ
end

function stress_intensity_factor(θ)
	a = radius_x
	b = radius_y
	E_k = π / 2
	tempA = σ_inf * sqrt(π) / (E_k) * sqrt(b / a)
	tempB = (a^4 * sin(θ)^2 + b^4 * cos(θ)^2) / (a^2 * sin(θ)^2 + b^2 * cos(θ)^2)
	return tempA * tempB^(1 / 4)
end

function compute_analytical_stress_intensity_factor(∂Γ_msh)
	Kₛ = Float64[]
	for node in Inti.nodes(∂Γ_msh)
		θ = atan(node[2] / node[1])
		K_I = stress_intensity_factor(θ)
		push!(Kₛ, K_I)
	end
	return Kₛ
end

function display_stress_intensity_factor(Kₛ, fig, ax, labelK::String = "")
	GLMakie.lines!(ax, Kₛ; label = labelK)
	# GLMakie.ylims!(ax, 0, 2 * maximum(Kₛ))
end

function show_convergence_SIF(Kₛ_list, Kₛ_th, N_points_list, fig, ax)
	normalize_abscissa = x -> x / maximum(N_points_list)
	for (n, Kₛ) in enumerate(Kₛ_list)
		X = 0.0:1/(length(Kₛ)-1):1.0
		GLMakie.lines!(ax, X, Kₛ; label = "N = $(N_points_list[n])")
	end
	X = 0.0:1/(length(Kₛ_th)-1):1.0
	GLMakie.lines!(ax, X, Kₛ_th; label = "Analytical")
	GLMakie.ylims!(ax, 0, 2 * maximum(Kₛ_th))
end

function display_convergence(radius_x, radius_y)
	λ = 1.0
	μ = 1.0
	op = Inti.Elastostatic(; dim = 3, λ, μ)
	T = Inti.HyperSingularKernel(op)

	meshsize_list = [1.0, 0.5, 0.4]
	msh_list = [create_disk_shaped_crack(meshsize, radius_x, radius_y) for meshsize in meshsize_list]
	Γ_list = [Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh) for msh in msh_list]
	∂Γ_list = [Inti.boundary(Γ) for Γ in Γ_list]
	Γ_msh_list = [msh[Γ] for (msh, Γ) in zip(msh_list, Γ_list)]
	∂Γ_msh_list = [msh[∂Γ] for (msh, ∂Γ) in zip(msh_list, ∂Γ_list)]
	Γ_quad_list = [Inti.Quadrature(Γ_msh; qorder = 3) for Γ_msh in Γ_msh_list]

	T_op_sing_list = [Inti.IntegralOperator(_custom_kernel, Γ_quad, Γ_quad) for Γ_quad in Γ_quad_list]

	T₀_sing_list = [Inti.assemble_matrix(T_op) for T_op in T_op_sing_list]

	δT_list = [Inti.guiggiani_correction(
		T_op;
		nearfield_distance = 3 * meshsize,
		nearfield_qorder = 40,
	) for (T_op, meshsize) in zip(T_op_sing_list, meshsize_list)]

	Tnew_list = [T₀ + δT for (T₀, δT) in zip(T₀_sing_list, δT_list)]

	nodes_list = [[node.coords for node in Γ_quad] for Γ_quad in Γ_quad_list]

	σ_inf = 1.0

	T⁺_x_list = [map(node -> 0.0, nodes) for nodes in nodes_list]
	T⁺_y_list = [map(node -> 0.0, nodes) for nodes in nodes_list]
	T⁺_z_list = [map(node -> σ_inf, nodes) for nodes in nodes_list]

	T⁺_list = [collect(Iterators.flatten((x, y, z) for (x, y, z) in zip(T⁺_x_list[n], T⁺_y_list[n], T⁺_z_list[n]))) for n in 1:length(meshsize_list)]

	f_list = [[SVector(0, 0, σ_inf) for node in Γ_quad] for Γ_quad in Γ_quad_list]
	_f_list = [reinterpret(Float64, f) for f in f_list]

	_ψ_list = [(-Tnew_sing.data) \ _f for (Tnew_sing, _f) in zip(Tnew_list, _f_list)]
	ψ_list = [reinterpret(SVector{3, Float64}, _ψ) for _ψ in _ψ_list]

	ω = [[_kernel_correction_term(node, node) for node in Γ_quad] for Γ_quad in Γ_quad_list]

	φ_list = [ψ .* ω for (ψ, ω) in zip(ψ_list, ω)]

	N_points_list = [length(Inti.nodes(∂Γ_msh)) for ∂Γ_msh in ∂Γ_msh_list]

	Kₛ_num_list = [stress_intensity_factor(Γ_quad, ∂Γ_msh, ψ) for (Γ_quad, ∂Γ_msh, ψ) in zip(Γ_quad_list, ∂Γ_msh_list, ψ_list)]
	Kₛ_th = compute_analytical_stress_intensity_factor(∂Γ_msh_list[end])

	fig = Figure()
	ax = Axis(fig[1, 1])
	show_convergence_SIF(Kₛ_num_list, Kₛ_th, N_points_list, fig, ax)
	axislegend(ax)
	display(fig)
end

function _kernel_correction_term(target, source)::Float64
	x, y = Inti.coords(target), Inti.coords(source)
	correction = sqrt(abs(radius_x - sqrt((y[1]^2 + y[2]^2))))
	return correction
end

λ = 1.0
μ = 1.0
op = Inti.Elastostatic(; dim = 3, λ, μ)
T = Inti.HyperSingularKernel(op)

function _custom_kernel(target, source)::SMatrix{3, 3, Float64, 9}
	x, y = Inti.coords(target), Inti.coords(source)
	correction = _kernel_correction_term(target, source)
	return correction * T(target, source)
end

function show_FIC(meshsize)
	radius_x = 1.0
	radius_y = 1.0
	msh = create_disk_shaped_crack(meshsize, radius_x, radius_y)

	Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
	∂Γ = Inti.boundary(Γ)
	Γ_msh = msh[Γ]
	∂Γ_msh = msh[∂Γ]

	Γ_quad = Inti.Quadrature(Γ_msh; qorder = 3)

	T_op_sing = Inti.IntegralOperator(_custom_kernel, Γ_quad, Γ_quad)
	T₀_sing = Inti.assemble_matrix(T_op_sing)
	δT_sing = Inti.guiggiani_correction(
		T_op_sing;
		nearfield_distance = 5 * meshsize,
		nearfield_qorder = 80,
	)
	Tnew_sing = T₀_sing + δT_sing

	σ_inf = 1.0
	f = [SVector(0, 0, σ_inf) for node in Γ_quad]

	_f = reinterpret(Float64, f)

	_ψ = (-Tnew_sing.data) \ _f
	ψ = reinterpret(SVector{3, Float64}, _ψ)

	Kₛ_num = stress_intensity_factor(Γ_quad, ∂Γ_msh, ψ)
	Kₛ_th = compute_analytical_stress_intensity_factor(∂Γ_msh)

	ϵ_K = norm(Kₛ_num - Kₛ_th) / norm(Kₛ_th)
	@show ϵ_K
end

# display_convergence(1.0, 1.0)
meshsize = 0.15
show_FIC(meshsize)
