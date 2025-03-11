using Test
using Revise
using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using GLMakie
using Plots

function create_disk_shaped_crack(meshsize, radius_x, radius_y)
	gmsh.initialize()
	gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
	gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
	gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius_x, radius_y)
	gmsh.model.occ.synchronize()
	gmsh.model.mesh.generate(2)
	gmsh.model.mesh.recombine()
	msh = Inti.import_mesh(; dim = 3)
	# gmsh.fltk.run()
	gmsh.finalize()

	# filter_Γ = e -> Inti.geometric_dimension(e) == 2

	# Γ = Inti.Domain(filter_Γ, msh)
	# ∂Γ = Inti.boundary(Γ)
	# Γ_msh = view(msh, Γ)
	# ∂Γ_msh = view(msh, ∂Γ)

	return msh
end

meshsize = 0.5
radius_x = 1.0
radius_y = 1.0
msh = create_disk_shaped_crack(meshsize, radius_x, radius_y)
Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
Γ_msh = msh[Γ]
Γ_quad = Inti.Quadrature(Γ_msh; qorder = 3)

op = Inti.Laplace(; dim = 3)
K′ = Inti.AdjointDoubleLayerKernel(op)
T = Inti.HyperSingularKernel(op)

K′_op = Inti.IntegralOperator(K′, Γ_quad, Γ_quad)
T_op = Inti.IntegralOperator(T, Γ_quad, Γ_quad)

K′₀ = Inti.assemble_matrix(K′_op)
T₀ = Inti.assemble_matrix(T_op)

δK′ = Inti.guiggiani_correction(
	K′_op;
	nearfield_distance = 3 * meshsize,
	nearfield_qorder = 40,
)
K′new = K′₀ + δK′

δT = Inti.guiggiani_correction(
	T_op;
	nearfield_distance = 3 * meshsize,
	nearfield_qorder = 20,
)
Tnew = T₀ + δT

function _displacement_jump_function(x)
	r = sqrt(x[1]^2 + x[2]^2)
	return 4 / π * sqrt(1 - r^2)
end

φ_th = [_displacement_jump_function(node.coords) for node in Γ_quad]

nodes = [node.coords for node in Γ_quad]

# T⁺ = [i / length(Γ_quad) for (i, node) in enumerate(Γ_quad)]
# T⁺ = map(node -> 1.0, nodes)
# T⁺ = map(node -> abs(node[1] * node[2]), nodes)

T = -Tnew * φ_th

fig = Figure()
ax = Axis(fig[1, 1])
X = 0.0:1/(length(T)-1):1.0
GLMakie.lines!(ax, X, T)
GLMakie.ylims!(ax, 0, maximum(T))
display(fig)

# φ = -Tnew \ T⁺

# rhs = -Tnew * φ_th
# @test norm(rhs - lhs, Inf) < 1e-6

function display_displacement_jump(Γ_quad, φ)
	X = [node.coords[1] for node in Γ_quad]
	Y = [node.coords[2] for node in Γ_quad]
	global_mesh_size = abs(maximum(X) - minimum(X))
	Z₁⁺ = φ / 2
	Z₁⁻ = -φ / 2
	fig = Figure()
	ax = Axis3(fig[1, 1])
	meshscatter!(ax, X, Y, Z₁⁺, markersize = 0.01)
	meshscatter!(ax, X, Y, Z₁⁻, markersize = 0.01)
	GLMakie.xlims!(ax, -global_mesh_size, global_mesh_size)
	GLMakie.ylims!(ax, -global_mesh_size, global_mesh_size)
	GLMakie.zlims!(ax, -global_mesh_size, global_mesh_size)

	display(fig)
end

display_displacement_jump(Γ_quad, φ)
