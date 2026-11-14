# In this script we will test the accuracy of the solution for
# the Lippmann-Schwinger equation using Müller's formulation.

# The problem will be two-dimensional.
# The domain will be the unit disk.
# The exterior domain will be points outside the unit disk but
#   inside the square [-3, 3] x [-3, 3].
# The incident field will be a plane wave propagating in the
#   direction [1, 0].
# The exterior wave number will be 8π.
# The interior wave number will be 4π.


# This example is constructed by taking a look at:
# - lippman_schwinger_problem.jl
# - Poisson problem from the Examples in Inti's documentation
# - Section 7.3 of the paper on "Fast, High-order Accurate Volume Potentials"
# - Müller's formulation for the transmission problem from Inti's documentation

# Let's start by adding the necessary imports
using Inti
using Gmsh
using GLMakie
using HMatrices
using FMM2D
using IterativeSolvers
using Meshes
using LinearAlgebra
using LinearMaps

# Let's set the physical parameters and the approximation orders
k₁ = 12.0
k₂ = 24.0
λ₁ = 2π / k₁
λ₂ = 2π / k₂
Ω_qorder = 4
Γ_qorder = 6
gorder = 2

# Define the function to create the domain, boundary and exterior
function gmsh_disk(; meshsize, order = gorder, radius = 1, visualize = false, name)
    gmsh.initialize()
    gmsh.model.add("disk_lippmann_schwinger")
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    disk_tag = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    rectangle_tag = gmsh.model.occ.addRectangle(-3, -3, 0, 6, 6)
    outDimTags, _ =
        gmsh.model.occ.cut([(2, rectangle_tag)], [(2, disk_tag)], -1, true, false)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [disk_tag], -1, "omega")
    gmsh.model.addPhysicalGroup(2, [dt[2] for dt in outDimTags], -1, "sigma")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    visualize && gmsh.fltk.run()
    gmsh.option.setNumber("Mesh.SaveAll", 1) # otherwise only the physical groups are saved
    gmsh.write(name)
    return gmsh.finalize()
end

name = joinpath(@__DIR__, "disk_and_exterior.msh")
meshsize = min(λ₁, λ₂) / 8
gmsh_disk(; meshsize, order = gorder, visualize = true, name);
Inti.clear_entities!() # empty the entity cache
msh = Inti.import_mesh(name; dim = 2);

# Extra components of the mesh
Ω = Inti.Domain(e -> "omega" ∈ Inti.labels(e), msh)
Γ = Inti.boundary(Ω)
Σ = Inti.Domain(e -> "sigma" ∈ Inti.labels(e), msh)

Ω_msh = view(msh, Ω)
Γ_msh = view(msh, Γ)
Σ_msh = view(msh, Σ)

# Visualize the mesh components
viz(Ω_msh; showsegments = true)
viz!(Γ_msh; color = :red)
viz!(Σ_msh; showsegments = true, color = :green)

# Quadrature for volume and boundary
Ω_quad = Inti.Quadrature(Ω_msh; qorder = Ω_qorder)
Γ_quad = Inti.Quadrature(Γ_msh; qorder = Γ_qorder)

# Extract nodes for the exterior domain
Σ_nodes = Inti.nodes(Σ_msh)

# Create external and internal PDEs
op₁ = Inti.Helmholtz(; k = k₁, dim = 2)
op₂ = Inti.Helmholtz(; k = k₂, dim = 2)

# Create the boundary integral operators for the transmission problem solution
S₁, D₁ = Inti.single_double_layer(;
    op = op₁,
    target = Γ_quad,
    source = Γ_quad,
    compression = (method = :fmm, tol = :1.0e-8),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

K₁, N₁ = Inti.adj_double_layer_hypersingular(;
    op = op₁,
    target = Γ_quad,
    source = Γ_quad,
    compression = (method = :fmm, tol = :1.0e-8),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

S₂, D₂ = Inti.single_double_layer(;
    op = op₂,
    target = Γ_quad,
    source = Γ_quad,
    compression = (method = :fmm, tol = :1.0e-8),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

K₂, N₂ = Inti.adj_double_layer_hypersingular(;
    op = op₂,
    target = Γ_quad,
    source = Γ_quad,
    compression = (method = :fmm, tol = :1.0e-8),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

# Construct the block operator of the BIE system
L = [
    I + LinearMap(D₂) - LinearMap(D₁) -LinearMap(S₂) + LinearMap(S₁)
    LinearMap(N₂) - LinearMap(N₁) I - LinearMap(K₂) + LinearMap(K₁)
]

# Let's define the planewave incident field
uᵢ = x -> exp(im * k₁ * dot(x, [0, 1]))
∇uᵢ = x -> im * k₁ * uᵢ(x) * [0, 1]

# Right hand side of the linear system
rhs₁_dir = map(Γ_quad) do q
    x = q.coords
    return uᵢ(x)
end

rhs₂_dir = map(Γ_quad) do q
    x = q.coords
    n = q.normal
    return dot(n, ∇uᵢ(x))
end

rhs_dir = [rhs₁_dir; rhs₂_dir]

# Use GMRES to solve the linear system
sol_dir, hist_dir =
    gmres(L, rhs_dir; log = true, abstol = 1.0e-8, verbose = true, restart = 1000, maxiter = 1000)
@show hist_dir

# Reshape and define the traces
sol_dir_temp = reshape(sol_dir, size(Γ_quad, 1), 2)
γ⁻u₂, ∂ₙ⁻u₂ = sol_dir_temp[:, 1], sol_dir_temp[:, 2]
γ⁺u₁ = γ⁻u₂ - rhs₁_dir
∂ₙ⁺u₁ = ∂ₙ⁻u₂ - rhs₂_dir

# Build the potentials for the representation
𝒮₁, 𝒟₁ = Inti.single_double_layer(;
    op = op₁,
    target = Σ_nodes,
    source = Γ_quad,
    compression = (method = :fmm, tol = :1.0e-8),
    correction = (method = :dim, maxdist = 5 * meshsize, target_location = :outside),
)

𝒮₂, 𝒟₂ = Inti.single_double_layer(;
    op = op₂,
    target = Ω_quad,
    source = Γ_quad,
    compression = (method = :fmm, tol = :1.0e-8),
    correction = (method = :dim, maxdist = 5 * meshsize, target_location = :inside),
)
pt = Inti.Point2D([0.6145833333333334, -1.0989583333333333])
𝒮P₁, 𝒟P₁ = Inti.single_double_layer_potential(; op = op₁, source = Γ_quad)
𝒮P₂, 𝒟P₂ = Inti.single_double_layer_potential(; op = op₂, source = Γ_quad)
w₂pt = 𝒟P₁[γ⁺u₁](pt) - 𝒮P₁[∂ₙ⁺u₁](pt)

# Obtain the approximate solution of the scattered and transmitted fields
w₁ = 𝒟₁ * γ⁺u₁ - 𝒮₁ * ∂ₙ⁺u₁
w₂ = -𝒟₂ * γ⁻u₂ + 𝒮₂ * ∂ₙ⁻u₂

# Evaluate the incident field in the exterior nodes
Σ_uᵢ = map(Σ_nodes) do x
    return uᵢ(x)
end

# Visualize the transmission problem solution
Ω_nodes = Inti.nodes(Ω_msh)
u_Ω_nodes = Inti.quadrature_to_node_vals(Ω_quad, w₂)
colorrange = (-2, 2)
colormap = :RdBu
fig = Figure(; size = (800, 300))
ax = Axis(fig[1, 1]; aspect = DataAspect())
viz!(Ω_msh; colorrange, colormap = colormap, color = real(u_Ω_nodes), interpolate = true)
viz!(Σ_msh; colorrange, colormap = colormap, color = real(w₁ + Σ_uᵢ), interpolate = true)
cb = Colorbar(fig[1, 2]; label = "u", colormap = colormap, colorrange)
ax.title = "Transmission Problem Solution"

# Now let's solve the same problem using the Lippmann-Schwinger equation
#-----------------------------------------------------------------------
# Let's first assemble the volume operator

V_k₁ = Inti.volume_potential(;
    op = op₁,
    target = Ω_quad,
    source = Ω_quad,
    compression = (method = :fmm, tol = 1.0e-8),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

# Now let's solve the system using GMRES

rhs_lippmann = map(Ω_quad) do q
    x = q.coords
    return uᵢ(x)
end

sol_lippmann, hist_lippmann =
    gmres(I + (k₁^2 - k₂^2) * V_k₁, rhs_lippmann; log = true, abstol = 1.0e-8, verbose = true, restart = 1000, maxiter = 1000)
@show hist_lippmann

# Build the solution in the exterior using the representation formula

V_k₁_Σ = Inti.volume_potential(;
    op = op₁,
    target = Σ_nodes,
    source = Ω_quad,
    compression = (method = :fmm, tol = 1.0e-8),
    correction = (method = :dim, maxdist = 5 * meshsize, target_location = :outside),
)

ext_lipmann = Σ_uᵢ + (k₂^2 - k₁^2) * V_k₁_Σ * sol_lippmann

# Visualize the Lippmann-Schwinger solution

u_Ω_nodes_lippmann = Inti.quadrature_to_node_vals(Ω_quad, sol_lippmann)
ax_2 = Axis(fig[1, 3]; aspect = DataAspect())
viz!(Ω_msh; colorrange, colormap = colormap, color = real(u_Ω_nodes_lippmann), interpolate = true)
viz!(Σ_msh; colorrange, colormap = colormap, color = real(ext_lipmann), interpolate = true)
cb = Colorbar(fig[1, 4]; label = "u", colormap = colormap, colorrange)
ax_2.title = "Lippmann-Schwinger Solution"

# Now let's compare both solutions and visualize the error

Ω_error = abs.(u_Ω_nodes - u_Ω_nodes_lippmann)
Σ_error = abs.(w₁ + Σ_uᵢ - ext_lipmann)
ax_3 = Axis(fig[1, 5]; aspect = DataAspect())
colormap = :inferno
colorrange = (-8, 0)
viz!(Ω_msh; colorrange, colormap = colormap, color = log10.(Ω_error), interpolate = true)
viz!(Σ_msh; colorrange, colormap = colormap, color = log10.(Σ_error), interpolate = true)
cb = Colorbar(fig[1, 6]; label = "log10(error)", colormap = colormap, colorrange)
ax_3.title = "Pointwise absolute error"

# Save the figure

name_fig = joinpath(@__DIR__, "circle_lippmann_schwinger_vs_muellers_transmission.png")
save(name_fig, fig, px_per_unit = 10)
