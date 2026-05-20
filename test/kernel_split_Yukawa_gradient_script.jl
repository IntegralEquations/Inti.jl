using Inti
using Gmsh
using HMatrices
using IterativeSolvers
using LinearAlgebra
using SpecialFunctions
using StaticArrays

# 1. PARAMETER AND GEOMETRY SETUP
println("Setting up Yukawa gradient script parameters and geometry...")

λ = 30.0 * π
meshsize = 0.1
qorder = 12
n_quad_pts = 16
smoothness = 5
maxdist_factor = 1.1
relerr_tol = 5.0e-6
source_scale = 1.0e12
hmatrix_tol = 1.0e-12
gmres_tol = 1.0e-14

op = Inti.Yukawa(; dim = 2, λ)
source_point = SVector(1.5, 0.0)

angle_mod = x -> mod(angle(x), 2π)

r₀ = 1.0
a = 0.3
ω = 5
θ₀ = 0.2
starfish_rad = θ -> r₀ + a * cos(ω * (2π * θ - θ₀))
starfish_rad_p = θ -> -a * 2π * ω * sin(ω * (2π * θ - θ₀))
starfish_v = θ -> (starfish_rad_p(θ) + im * 2π * starfish_rad(θ)) * exp(im * 2π * θ)
starfish_s = θ -> norm(starfish_v(θ))
starfish_v_x = θ -> real(starfish_v(θ))
starfish_v_y = θ -> imag(starfish_v(θ))

starfish_rad_pp = θ -> -a * (4π^2) * ω^2 * cos(ω * (2π * θ - θ₀))
starfish_vp =
    θ ->
(starfish_rad_pp(θ) + 2 * im * 2π * starfish_rad_p(θ) - (4π^2) * starfish_rad(θ)) *
    exp(im * 2π * θ)
starfish_vp_x = θ -> real(starfish_vp(θ))
starfish_vp_y = θ -> imag(starfish_vp(θ))

function starfish_κ(θ)
    return (starfish_v_x(θ) * starfish_vp_y(θ) - starfish_v_y(θ) * starfish_vp_x(θ)) /
        (starfish_s(θ)^3)
end

v = θ -> starfish_v(θ)
κ = θ -> starfish_κ(θ)
boundary_inv = θ -> angle_mod(θ[1] + im * θ[2]) / (2π)
function starfish_coor(θ)
    r = starfish_rad(θ)
    return SVector(r * cos(2π * θ), r * sin(2π * θ))
end

u_exact = x -> source_scale * besselk(0, λ * norm(x - source_point))
grad_u_exact = x -> begin
    r = x - source_point
    d = norm(r)
    return -source_scale * λ * besselk(1, λ * d) * r / d
end

Inti.clear_entities!()
gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)

bnd1 = Inti.gmsh_curve(starfish_coor, 0, 1; meshsize)
cl = gmsh.model.occ.addCurveLoop([bnd1])
gmsh.model.occ.addPlaneSurface([cl])
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
msh = Inti.import_mesh(; dim = 2)
gmsh.finalize()

Ω = Inti.Domain(Inti.entities(msh)) do ent
    return Inti.geometric_dimension(ent) == 2
end
Γ = Inti.external_boundary(Ω)

crvmsh = Inti.curve_mesh(msh, starfish_coor, smoothness)
Ω_crv_quad = Inti.Quadrature(crvmsh[Ω]; qorder)
Γ_crv_el = collect(Inti.elements(crvmsh[Γ]))
Γ_crv_quad = Inti.Quadrature(crvmsh[Γ], Inti.GaussLegendre(n_quad_pts))
Γ_crv_quad_connectivity = Inti.etype2qtags(Γ_crv_quad, first(Inti.element_types(crvmsh[Γ])))
n_el = length(Γ_crv_el)
area = Inti.integrate(x -> 1.0, Ω_crv_quad)
@info "Geometry setup complete." n_el n_quad_pts area

g_vec = map(q -> u_exact(q.coords), Γ_crv_quad)
u_exact_vals = map(q -> u_exact(Inti.coords(q)), Ω_crv_quad)
gradient_exact = map(q -> grad_u_exact(Inti.coords(q)), Ω_crv_quad)

println("\n" * "="^30)
println("Running Yukawa kernel-split gradient script")
println("="^30)

ksplit_b2b_settings = (
    method = :ksplit,
    connectivity = Γ_crv_quad_connectivity,
    elements = Γ_crv_el,
    velocity_fn = v,
    curvature_fn = κ,
    boundary_inv = boundary_inv,
    parametric_length = 1.0,
    n_panel_corr = 3,
)

@time _, D_b2b = Inti.single_double_layer(;
    op = op,
    target = Γ_crv_quad,
    source = Γ_crv_quad,
    compression = (method = :hmatrix, tol = hmatrix_tol),
    correction = ksplit_b2b_settings,
)

σ = gmres(-I / 2 + D_b2b, g_vec; reltol = gmres_tol, abstol = gmres_tol, restart = 1000)

ksplit_b2d_settings = (
    method = :ksplit,
    connectivity = Γ_crv_quad_connectivity,
    elements = Γ_crv_el,
    velocity_fn = v,
    curvature_fn = κ,
    boundary_inv = boundary_inv,
    parametric_length = 1.0,
    maxdist = maxdist_factor * meshsize,
    target_location = :inside,
)

@time _, D_b2d = Inti.single_double_layer(;
    op = op,
    target = Ω_crv_quad,
    source = Γ_crv_quad,
    compression = (method = :hmatrix, tol = hmatrix_tol),
    correction = ksplit_b2d_settings,
)

u_num = D_b2d * σ
solution_abs_err = maximum(abs.(u_num .- u_exact_vals))
solution_relerr = solution_abs_err / maximum(abs.(u_exact_vals))

@time Dx_b2d, Dy_b2d = Inti.double_layer_gradient(;
    op = op,
    target = Ω_crv_quad,
    source = Γ_crv_quad,
    compression = (method = :hmatrix, tol = hmatrix_tol),
    correction = ksplit_b2d_settings,
)

gradient_x = Dx_b2d * σ
gradient_y = Dy_b2d * σ
gradient_num = [SVector(gradient_x[i], gradient_y[i]) for i in eachindex(Ω_crv_quad)]

abs_err = maximum(norm.(gradient_num .- gradient_exact))
relerr = abs_err / maximum(norm.(gradient_exact))

println("Yukawa solution max absolute error: ", solution_abs_err)
println("Yukawa solution max relative error: ", solution_relerr)
println("Yukawa gradient max absolute error: ", abs_err)
println("Yukawa gradient max relative error: ", relerr)
println("Requested relative tolerance: ", relerr_tol)
