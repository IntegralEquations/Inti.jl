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

meshsize = 1.0
radius_x = 1.0
radius_y = 1.0
msh = create_disk_shaped_crack(meshsize, radius_x, radius_y)

##

Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
Γ_msh = msh[Γ]
Γ_quad = Inti.Quadrature(Γ_msh; qorder = 3)

λ = 1.0
μ = 1.0
op = Inti.Elastostatic(; dim = 3, λ, μ)
T = Inti.HyperSingularKernel(op)

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

@time begin
	T_op_sing = Inti.IntegralOperator(_custom_kernel, Γ_quad, Γ_quad)
	T₀_sing = Inti.assemble_matrix(T_op_sing)
	δT_sing = Inti.guiggiani_correction(
		T_op_sing;
		nearfield_distance = 5 * meshsize,
		nearfield_qorder = 80,
	)
	Tnew_sing = T₀_sing + δT_sing

	φ_th = [_displacement_jump_function(node.coords) for node in Γ_quad]
	σ_inf = 1.0
	f = [SVector(0, 0, σ_inf) for node in Γ_quad]

	_f = reinterpret(Float64, f)

	_ψ = (-Tnew_sing.data) \ _f
	ψ = reinterpret(SVector{3, Float64}, _ψ)

	ω = [_kernel_correction_term(node, node) for node in Γ_quad]

	φ = ψ .* ω
end
# er = [norm(φ[i] - φ_th[i]) for i in 1:length(φ)]
# display_displacement_jump(Γ_quad, [x[3] for x in tmp])
# display_displacement_jump(Γ_quad, [x[3] for x in φ_th])
# display_displacement_jump(Γ_quad, [x[3] for x in φ])
# display_displacement_jump(Γ_quad, er)

@show norm(φ - φ_th, Inf)
@show norm(φ - φ_th, Inf) / norm(φ_th, Inf)

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
	ν = λ / (2(λ + μ))
	E_temp = μ * (3λ + 2μ) / (λ + μ)
	a = radius_x
	b = radius_y
	k = sqrt(1 - b^2 / a^2)
	E_k = π / 2
	tempA = σ_inf * sqrt(π) / (E_temp * E_k) * sqrt(b / a)
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

filter_Γ = e -> Inti.geometric_dimension(e) == 2

Γ = Inti.Domain(filter_Γ, msh)
∂Γ = Inti.boundary(Γ)
Γ_msh = view(msh, Γ)
∂Γ_msh = view(msh, ∂Γ)

Kₛ_num = stress_intensity_factor(Γ_quad, ∂Γ_msh, ψ)
Kₛ_analytical = compute_analytical_stress_intensity_factor(∂Γ_msh)
fig = Figure()
ax = Axis(fig[1, 1])
display_stress_intensity_factor(Kₛ_num, fig, ax, "Numerical")
display_stress_intensity_factor(Kₛ_analytical, fig, ax, "Analytical")
axislegend(ax)
display(fig)
