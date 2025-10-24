using Inti
using Gmsh
using SpecialFunctions
using StaticArrays
using HMatrices
using IterativeSolvers
using FastGaussQuadrature
using LinearAlgebra
using LinearMaps
using Meshes
# using GLMakie

# 1. PARAMETER AND GEOMETRY SETUP
println("Setting up test parameters and geometry...")

# PDE and Discretization Parameters
λ = 30.0 * π
op = Inti.Yukawa(; dim=2, λ)
meshsize = 0.1
n_quad_pts = 16
hmatrix_tol = 1e-12
gmres_tol = 1e-14

# Helper function for boundary inverse
angle_mod = x -> mod(angle(x), 2π)

# Define Starfish Geometry
r₀ = 1.0 # mean radius
a = 0.3 # amplitude of wobble
ω = 5 # frequency of wobble
θ₀ = 0.2 # rotation angle for the starfish shape
starfish_rad = (θ) -> r₀ + a * cos(ω * (2π * θ - θ₀)) # radius function for starfish shape
starfish_rad_p = (θ) -> -a * 2π * ω * sin(ω * (2π * θ - θ₀)) # derivative of the radius function
starfish_v = (θ) -> (starfish_rad_p(θ) + im * 2π * starfish_rad(θ)) * exp(im * 2π * θ) # velocity function
starfish_s = (θ) -> norm(starfish_v(θ)) # speed function
starfish_v_x = (θ) -> real(starfish_v(θ)) # x-component of the velocity function
starfish_v_y = (θ) -> imag(starfish_v(θ)) # y-component of the velocity function

starfish_rad_pp = (θ) -> -a * (4π^2) * ω^2 * cos(ω * (2π * θ - θ₀)) # second derivative of the radius function
starfish_vp =  # derivative of the velocity function
    (θ) ->
        (starfish_rad_pp(θ) + 2 * im * 2π * starfish_rad_p(θ) - (4π^2) * starfish_rad(θ)) *
        exp(im * 2π * θ)
starfish_vp_x = (θ) -> real(starfish_vp(θ)) # x-component of the derivative of velocity
starfish_vp_y = (θ) -> imag(starfish_vp(θ)) # y-component of the derivative of velocity

function starfish_κ(θ)
    return (starfish_v_x(θ) * starfish_vp_y(θ) - starfish_v_y(θ) * starfish_vp_x(θ)) /
           (starfish_s(θ)^3)
end # curvature function

# set the generic function name
v = θ -> starfish_v(θ)
s = θ -> starfish_s(θ)
κ = θ -> starfish_κ(θ)
boundary_inv = (θ) -> angle_mod(θ[1] + im * θ[2]) / (2π)
function starfish_coor(θ)
    r = starfish_rad(θ)
    return SVector(r * cos(2π * θ), r * sin(2π * θ))
end

# Generate and Curve Mesh
gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)

bnd1 = Inti.gmsh_curve(starfish_coor, 0, 1; meshsize)
cl = gmsh.model.occ.addCurveLoop([bnd1])
disk = gmsh.model.occ.addPlaneSurface([cl])
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
msh = Inti.import_mesh(; dim=2)
gmsh.finalize()
Ω = Inti.Domain(Inti.entities(msh)) do ent
    return Inti.geometric_dimension(ent) == 2
end

θ_smooth = 5 # smoothness order of curved elements
crvmsh = Inti.curve_mesh(msh, starfish_coor, θ_smooth)
# viz(crvmsh[Ω], showsegments=true, figure=(; size=(425, 400),)) # Optional visualization

# Define Quadratures
Ω_crv_quad = Inti.Quadrature(crvmsh[Ω]; qorder=10)
Γ = Inti.external_boundary(Ω)
Γ_crv_el = collect(Inti.elements(crvmsh[Γ]))
Γ_crv_quad = Inti.Quadrature(crvmsh[Γ], Inti.GaussLegendre(n_quad_pts))
Γ_crv_quad_connectivity = Inti.etype2qtags(Γ_crv_quad, first(Inti.element_types(crvmsh[Γ])))
n_el = length(Γ_crv_el)
@info "Geometry setup complete. n_elements = $n_el, n_quad_pts = $n_quad_pts"

# Define Exact Solution
uₑₓ = (x) -> 1e12 * besselk(0, λ * norm((x .- (1.5, 0)), 2))
g = uₑₓ

g_vec = map(q -> g(q.coords), Γ_crv_quad)
u_sol_ex = map(q -> uₑₓ(Inti.coords(q)), Ω_crv_quad)

# 2. TEST 1: STANDARD KERNEL SPLIT
println("\n" * "="^30)
println("Running Test 1: Standard Kernel Split")
println("="^30)

# B2B Operator (Standard KSplit)
ksplit_b2b_settings = (
    method=:ksplit,
    connectivity=Γ_crv_quad_connectivity,
    elements=Γ_crv_el,
    velocity_fn=v,
    curvature_fn=κ,
    boundary_inv=boundary_inv,
    PARAMETRIC_LENGTH=1.0,
    n_panel_corr=3,
)

@time S_b2b_ks, D_b2b_ks = Inti.single_double_layer(;
    op=op,
    target=Γ_crv_quad,
    source=Γ_crv_quad,
    compression=(method=:hmatrix, tol=hmatrix_tol),
    correction=ksplit_b2b_settings,
)

# B2D Operator (Standard KSplit)
ksplit_b2d_settings = (
    method=:ksplit,
    connectivity=Γ_crv_quad_connectivity,
    elements=Γ_crv_el,
    velocity_fn=v,
    curvature_fn=κ,
    boundary_inv=boundary_inv,
    PARAMETRIC_LENGTH=1.0,
    maxdist=1.1 * meshsize,
    target_location=:inside,
    # affine_preimage=false, # default is true
)

@time S_b2d_ks, D_b2d_ks = Inti.single_double_layer(;
    op=op,
    target=Ω_crv_quad,
    source=Γ_crv_quad,
    compression=(method=:hmatrix, tol=hmatrix_tol),
    correction=ksplit_b2d_settings,
)

# Solve and Check Error (Standard KSplit)
L_b2b_ks = -I / 2 + D_b2b_ks
L_b2d_ks = D_b2d_ks

println("Solving system (Standard Kernel Split)...")
σ_ks = gmres(L_b2b_ks, g_vec; reltol=gmres_tol, abstol=gmres_tol, verbose=true, restart=1000)
u_sol_ks = L_b2d_ks * σ_ks

er_ks = u_sol_ks - u_sol_ex
er_norm_ks = abs.(er_ks / maximum(abs.(u_sol_ex)))
println("KSplit Max Relative Error: ", norm(er_norm_ks, Inf))

# 3. TEST 2: ADAPTIVE KERNEL SPLIT
println("\n" * "="^30)
println("Running Test 2: Adaptive Kernel Split")
println("="^30)

# B2B Operator (Adaptive KSplit)
ksplit_adaptive_b2b_settings = (
    method=:ksplit_adaptive,
    connectivity=Γ_crv_quad_connectivity,
    elements=Γ_crv_el,
    velocity_fn=v,
    curvature_fn=κ,
    boundary_inv=boundary_inv,
    PARAMETRIC_LENGTH=1.0,
    n_panel_corr=3,
    Cε=3.5,
    Rε=3.7,
    target_location=:inside,
    # affine_preimage=false, # default is true
)

@time S_b2b_aks, D_b2b_aks = Inti.single_double_layer(;
    op=op,
    target=Γ_crv_quad,
    source=Γ_crv_quad,
    compression=(method=:hmatrix, tol=hmatrix_tol),
    correction=ksplit_adaptive_b2b_settings,
)

# B2D Operator (Adaptive KSplit)
ksplit_adaptive_b2d_settings = (
    method=:ksplit_adaptive,
    connectivity=Γ_crv_quad_connectivity,
    elements=Γ_crv_el,
    velocity_fn=v,
    curvature_fn=κ,
    boundary_inv=boundary_inv,
    PARAMETRIC_LENGTH=1.0,
    maxdist=1.1 * meshsize,
    Cε=3.5,
    Rε=3.7,
    target_location=:inside,
    # affine_preimage=false, # default is true
)

@time S_b2d_aks, D_b2d_aks = Inti.single_double_layer(;
    op=op,
    target=Ω_crv_quad,
    source=Γ_crv_quad,
    compression=(method=:hmatrix, tol=hmatrix_tol),
    correction=ksplit_adaptive_b2d_settings,
)

# Solve and Check Error (Adaptive KSplit)
L_b2b_aks = -I / 2 + D_b2b_aks
L_b2d_aks = D_b2d_aks

println("Solving system (Adaptive Kernel Split)...")
σ_aks = gmres(L_b2b_aks, g_vec; reltol=gmres_tol, abstol=gmres_tol, verbose=true, restart=1000)
u_sol_aks = L_b2d_aks * σ_aks

er_aks = u_sol_aks - u_sol_ex
er_norm_aks = abs.(er_aks / maximum(abs.(u_sol_ex)))
println("Adaptive KSplit Max Relative Error: ", norm(er_norm_aks, Inf))