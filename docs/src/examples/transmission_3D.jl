# Specify the packages to be used

using Inti
using LinearAlgebra
using StaticArrays
using Gmsh
using Meshes
using GLMakie
using SpecialFunctions
using GSL
using IterativeSolvers
using LinearMaps

# Define the global parameters for the 3D Helmholz transmission problem

k₁       = 3π
k₂       = 2π
λ₁       = 2π / k₁
λ₂       = 2π / k₂	
qorder   = 4 # quadrature order
gorder   = 3 # order of geometrical approximation
nothing #hide

# Let's create the geometry of the scatterer

function gmsh_sphere(; meshsize, order = gorder, radius = 1, visualize = false, name)
    gmsh.initialize()
    gmsh.model.add("sphere-scattering")
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    sphere_tag = gmsh.model.occ.addSphere(0, 0, 0, radius)
    rectangle_xy_tag = gmsh.model.occ.addRectangle(-2, -2, 0, 4, 4)
    rectangle_xz_tag = gmsh.model.occ.addRectangle(-2, -2, 0, 4, 4)
    gmsh.model.occ.rotate([(2, rectangle_xz_tag)], 0, 0, 0, 1, 0, 0, π / 2)
    rectangle_yz_tag = gmsh.model.occ.addRectangle(-2, -2, 0, 4, 4)
    gmsh.model.occ.rotate([(2, rectangle_yz_tag)], 0, 0, 0, 0, 1, 0, π / 2)
    disk_xy_tag = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    disk_xz_tag = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    gmsh.model.occ.rotate([(2, disk_xz_tag)], 0, 0, 0, 1, 0, 0, π / 2)
    outDimTags, _ =
        gmsh.model.occ.cut([(2, rectangle_xy_tag), (2, rectangle_xz_tag)], [(3, sphere_tag)], -1, true, false)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [sphere_tag], -1, "omega")
    gmsh.model.addPhysicalGroup(2, [dt[2] for dt in outDimTags], -1, "sigma_out")
    gmsh.model.addPhysicalGroup(2, [disk_xy_tag, disk_xz_tag], -1, "sigma_in")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    visualize && gmsh.fltk.run()
    gmsh.option.setNumber("Mesh.SaveAll", 1) # otherwise only the physical groups are saved
    gmsh.write(name)
    return gmsh.finalize()
end
nothing #hide

name = joinpath(@__DIR__, "sphere.msh")
meshsize = min(λ₁, λ₂) / 10
gmsh_sphere(; meshsize, order = gorder, visualize = false, name)
Inti.clear_entities!() # empty the entity cache
msh = Inti.import_mesh(name; dim = 3)

# Define the domain and the boundary of the scatterer

Ω = Inti.Domain(e -> "omega" ∈ Inti.labels(e), Inti.entities(msh))
Γ = Inti.boundary(Ω)

# Define the quadrature rule on the boundary

Γ_msh = view(msh, Γ)
Q = Inti.Quadrature(Γ_msh; qorder)

# Define the external and internal PDEs

pde₁ = Inti.Helmholtz(; k = k₁, dim = 3)
pde₂ = Inti.Helmholtz(; k = k₂, dim = 3)

# Define the integral operators

using HMatrices
S₁, D₁ = Inti.single_double_layer(;
    pde = pde₁,
    target = Q,
    source = Q,
    compression = (method = :hmatrix, tol = :1e-6),
    correction = (method = :dim, maxdist = 5 * meshsize),
    )

K₁, N₁ = Inti.adj_double_layer_hypersingular(;
    pde = pde₁,
    target = Q,
    source = Q,
    compression = (method = :hmatrix, tol = :1e-6),
    correction = (method = :dim, maxdist = 5 * meshsize),
    )

S₂, D₂ = Inti.single_double_layer(;
    pde = pde₂,
    target = Q,
    source = Q,
    compression = (method = :hmatrix, tol = :1e-6),
    correction = (method = :dim, maxdist = 5 * meshsize),
    )

K₂, N₂ = Inti.adj_double_layer_hypersingular(;
    pde = pde₂,
    target = Q,
    source = Q,
    compression = (method = :hmatrix, tol = :1e-6),
    correction = (method = :dim, maxdist = 5 * meshsize),
    )
nothing

# Taking Müller's indirect formulation for the transmission problem we can assemble the following system of equations

L = [
    	I+LinearMap(D₁)-LinearMap(D₂) -LinearMap(S₁)+LinearMap(S₂)
    	LinearMap(N₁)-LinearMap(N₂) I-LinearMap(K₁)+LinearMap(K₂)
	]
nothing

# For validation purposes, let's manufacture outer and inner solutions and set the boundary conditions

# point source for the exterior solution with center at the origin and its normal gradient
u₁ = x -> exp(im * k₁ * sqrt(dot(x, x))) / sqrt(dot(x, x))
∇u₁ = x -> (im * k₁ - 1/sqrt(dot(x, x)))  * exp(im * k₁ * sqrt(dot(x, x))) / sqrt(dot(x, x))^2 * x
# plane wave for the interior solution
𝐝 = [1, 0, 0];
u₂ = x -> exp(im * k₂ * dot(𝐝, x))
∇u₂ = x -> im * k₂ * 𝐝 * exp(im * k₂ * dot(𝐝, x))

# Define the right-hand side of the system of equations

rhs₁ = map(Q) do q
    x = q.coords
    return u₁(x) + u₂(x)
end

rhs₂ = map(Q) do q
    x = q.coords
    n = q.normal
    return dot(n, ∇u₁(x) + ∇u₂(x))
end

rhs = [rhs₁; rhs₂]

# Solve the system of equations using GMRES

sol, hist =
    gmres(L, rhs; log = true, abstol = 1e-8, verbose = false, restart = 400, maxiter = 400)
@show hist

nQ = size(Q, 1)
sol_temp = reshape(sol, nQ, 2)
φ, ψ = sol_temp[:, 1], sol_temp[:, 2]

# Define the solution fields for the exterior and interior solutions

𝒮₁, 𝒟₁ = Inti.single_double_layer_potential(; pde = pde₁, source = Q)
𝒮₂, 𝒟₂ = Inti.single_double_layer_potential(; pde = pde₂, source = Q)

# Obtain the approximate solution of the scattered and transmitted fields
v₁ = x -> 𝒟₁[φ](x) - 𝒮₁[ψ](x)
v₂ = x -> -𝒟₂[φ](x) + 𝒮₂[ψ](x)

# To check the accuracy of the numerical approximation, we can compute the error
# over points in spheres centered at the origin outside and inside the scatterer

er₁ = maximum(1:100) do _
    x̂ = rand(Inti.Point3D)|> normalize # an SVector of unit norm
    x = 2 * x̂
    return norm(v₁(x)-u₁(x),Inf)/norm(u₁(x),Inf)
end
@assert er₁ < 1e-3 #hide
@info "error with correction = $er₁"

er₂ = maximum(1:100) do _
    x̂ = rand(Inti.Point3D)|> normalize # an SVector of unit norm
    x = 0.25 * x̂
    return norm(v₂(x)-u₂(x),Inf)#/norm(u₂(x),Inf)
end
@assert er₂ < 1e-3 #hide
@info "error with correction = $er₂"

# Visualize the solution

Σ⁺ = Inti.Domain(e -> "sigma_out" ∈ Inti.labels(e), Inti.entities(msh))
Σ⁻ = Inti.Domain(e -> "sigma_in" ∈ Inti.labels(e), Inti.entities(msh))

Σ⁺_msh = view(msh, Σ⁺)
Σ⁻_msh = view(msh, Σ⁻)

target_out = Inti.nodes(Σ⁺_msh)
target_in = Inti.nodes(Σ⁻_msh)

# Define an incident field in the form of a plane wave and its derivative

u_inc = x -> exp(im * k₁ * dot(𝐝, x))
∇u_inc = x -> im * k₁ * 𝐝 * exp(im * k₁ * dot(𝐝, x))

# Define the right-hand-side of the system of equations

rhs₁_inc = map(Q) do q
    x = q.coords
    return -u_inc(x)
end

rhs₂_inc = map(Q) do q
    x = q.coords
    n = q.normal
    return dot(n, -∇u_inc(x))
end

rhs_inc = [rhs₁_inc; rhs₂_inc]

# Solve the system of equations using GMRES

sol_inc, hist_inc =
    gmres(L, rhs_inc; log = true, abstol = 1e-8, verbose = false, restart = 400, maxiter = 400)
@show hist

nQ = size(Q, 1)
sol_temp_inc = reshape(sol_inc, nQ, 2)
φ_inc, ψ_inc = sol_temp_inc[:, 1], sol_temp_inc[:, 2]

S⁺, D⁺ = Inti.single_double_layer(;
    pde = pde₁,
    target = target_out,
    source = Q,
    compression = (method = :hmatrix, tol = :1e-6),
    correction = (method = :dim, maxdist = 5 * meshsize, target_location = :outside),
    )

S⁻, D⁻ = Inti.single_double_layer(;
    pde = pde₂,
    target = target_in,
    source = Q,
    compression = (method = :hmatrix, tol = :1e-6),
    correction = (method = :dim, maxdist = 5 * meshsize, target_location = :inside),
    )

w₁ = D⁺ * φ_inc - S⁺ * ψ_inc
w₂ = D⁻ * φ_inc - S⁻ * ψ_inc

uᵢ = u_inc.(target_out)

nv = length(Inti.nodes(Γ_msh))
colorrange_out = extrema(real(w₁)+real(uᵢ))
colorrange_int = extrema(real(w₂))
colorrange = (minimum([colorrange_out[1], colorrange_int[1]]), maximum([colorrange_out[2], colorrange_int[2]]))
colormap = :inferno
fig = Figure(resolution = (800, 800))
ax = Axis3(fig[1, 1]; aspect = :data)
viz!(Σ⁺_msh; colorrange, colormap, color = real(w₁)+real(uᵢ))
viz!(Σ⁻_msh; colorrange, colormap, color = real(w₂))
viz!(Γ_msh; colorrange, colormap = :bone, color = zeros(nv), interpolate = true, alpha = 0.4)
cb = Colorbar(fig[1, 2]; label = "Re(u)", colormap, colorrange)
fig