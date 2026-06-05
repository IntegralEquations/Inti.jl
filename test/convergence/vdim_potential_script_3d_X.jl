# # High-order convergence of vdim

using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using HMatrices
using FMM3D
using CairoMakie
using DataStructures

include("../test_utils.jl")
compression = (method = :fmm, tol = 1.0e-13, ndiv = 1600)
#compression = (method = :none,)

meshsize = 0.05
meshsize_bdry = meshsize
Inti.clear_entities!()
tmsh = @elapsed begin
    Ω, msh =
        gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = meshsize_bdry)
    Γ = Inti.external_boundary(Ω)

    Ωₕ = view(msh, Ω)
    Γₕ = view(msh, Γ)
end
meshsize_bdry = meshsize
Ωₕ_coarse = Ωₕ

#Inti.clear_entities!()
#tmsh = @elapsed begin
#    Ω_coarse, msh_coarse =
#        gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = meshsize)
#    Γ_coarse = Inti.external_boundary(Ω_coarse)
#
#    Ωₕ_coarse = view(msh_coarse, Ω_coarse)
#    Γₕ_coarse = view(msh_coarse, Γ_coarse)
#end
#@info "Mesh generation time: $tmsh"

interpolation_order = 4
VR_qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(interpolation_order)
bdry_qorder = 10 #min(2 * VR_qorder, 7)

tquad = @elapsed begin
    # Use VDIM with the Vioreanu-Rokhlin quadrature rule for Ωₕ
    Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = VR_qorder)
    dict = OrderedDict(E => Q for E in Inti.element_types(Ωₕ_coarse))
    Ωₕ_quad = Inti.Quadrature(Ωₕ_coarse, dict)
    # Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorders[1])
    #Qbdry = Inti.Gauss(; domain = :triangle, order = bdry_qorder)
    Qbdry = Inti.VioreanuRokhlin(; domain = :triangle, order = bdry_qorder)
    dictbdry = OrderedDict(E => Qbdry for E in Inti.element_types(Γₕ))
    Γₕ_quad = Inti.Quadrature(Γₕ, dictbdry)
end
@info "Quadrature generation time: $tquad"

k0 = π
#k0 = 3π
#k0 = 1
#k = 2π # helmholtz
k = 0 # laplace
θ = SVector(sin(π / 3) * cos(π / 3), sin(π / 3) * sin(π / 3), cos(π / 3))
#u  = (x) -> exp(im * k0 * dot(x, θ))
#du = (x,n) -> im * k0 * dot(θ, n) * exp(im * k0 * dot(x, θ))
u_exact = x -> cos(k0 * dot(x, θ))
grad_u_exact = x -> -k0 * θ * sin(k0 * dot(x, θ))
normal_deriv = (x, n) -> dot(grad_u_exact(x), n)
#du = (x, n) -> -k0 * dot(θ, n) * sin(k0 * dot(x, θ))
f_val = (x) -> (k0^2 - k^2) * u_exact(x)

f_d_nonpoly = [f_val(q.coords) for q in Ωₕ_quad]
u_b_nonpoly = [u_exact(q.coords) for q in Γₕ_quad]
du_b_nonpoly = [normal_deriv(q.coords, q.normal) for q in Γₕ_quad]

op = k == 0 ? Inti.Laplace(; dim = 3) : Inti.Helmholtz(; dim = 3, k)

## Boundary operators
tbnd = @elapsed begin
    S_b2d_std, D_b2d_std = Inti.single_double_layer(;
        op,
        target = Ωₕ_quad,
        source = Γₕ_quad,
        compression,
        correction = (method = :dim, maxdist = 5 * meshsize_bdry, target_location = :inside),
        kernel_variant = :default,
    )
end
@info "(Standard) Boundary operators time: $tbnd"
tbnd = @elapsed begin
    S_b2d_grad, D_b2d_grad = Inti.single_double_layer(;
        op,
        target = Ωₕ_quad,
        source = Γₕ_quad,
        compression,
        correction = (method = :dim, maxdist = 5 * meshsize_bdry, target_location = :inside),
        kernel_variant = :gradient,
    )
end
@info "(Gradient) Boundary operators time: $tbnd"

## Volume potentials
#tvol = @elapsed begin
#    V_d2d_std = Inti.volume_potential(;
#        op,
#        target = Ωₕ_quad,
#        source = Ωₕ_quad,
#        compression,
#        correction = (
#            method = :dim,
#            interpolation_order,
#            maxdist = 5 * meshsize_bdry,
#            boundary = Γₕ_quad,
#            S_b2d = S_b2d_std,
#            D_b2d = D_b2d_std,
#        ),
#        kernel_variant = :default,
#    )
#end
#@info "(Standard) Volume potential time: $tvol"
#tvol = @elapsed begin
#    V_d2d_grad = Inti.volume_potential(;
#        op,
#        target = Ωₕ_quad,
#        source = Ωₕ_quad,
#        compression,
#        correction = (
#            method = :dim,
#            interpolation_order,
#            maxdist = 5 * meshsize_bdry,
#            boundary = Γₕ_quad,
#            S_b2d = S_b2d_grad,
#            D_b2d = D_b2d_grad,
#        ),
#        kernel_variant = :gradient,
#    )
#end
#@info "(Gradient) Volume potential time: $tvol"
tvol = @elapsed begin
    X_d2d = Inti.volume_potential(;
        op,
        target = Ωₕ_quad,
        source = Ωₕ_quad,
        compression,
        correction = (
            method = :dim,
            interpolation_order,
            maxdist = 5 * meshsize_bdry,
            boundary = Γₕ_quad,
            S_b2d = S_b2d_std,
            D_b2d = D_b2d_std,
        ),
        kernel_variant = :hessian_source,
        #kernel_variant = :gradient_source,
    )
end
@info "(X) Volume potential time: $tvol"

# Standard plane wave test
#u_d_nonpoly_std = [u_exact(q.coords) for q in Ωₕ_quad]
#vref_nonpoly_std = u_d_nonpoly_std + D_b2d_std * u_b_nonpoly - S_b2d_std * du_b_nonpoly
#vapprox_nonpoly_std = V_d2d_std * f_d_nonpoly
#err_nonpoly_std = norm(vref_nonpoly_std - vapprox_nonpoly_std, Inf)
#println("  Standard plane wave test: Max Error = ", err_nonpoly_std)
#
## Gradient plane wave test
#u_d_nonpoly_grad = [grad_u_exact(q.coords) for q in Ωₕ_quad]
#vref_nonpoly_grad = u_d_nonpoly_grad + D_b2d_grad * u_b_nonpoly - S_b2d_grad * du_b_nonpoly
#vapprox_nonpoly_grad = V_d2d_grad * f_d_nonpoly
#err_nonpoly_grad = norm(vref_nonpoly_grad - vapprox_nonpoly_grad, Inf)
#println("  Gradient plane wave test: Max Error = ", err_nonpoly_grad)
#println("  NDOFS: ", length(Ωₕ_quad))

# ------------------------------------------------------------------------------
# Non-polynomial convergence test:  g = ∇u with u harmonic  ⇒  div g = 0,
# so  W[g] = -S[∂_ν u]  (closed form, exercising the volume residual).
# ------------------------------------------------------------------------------
#uₑ(x)  = exp(sqrt(2) * x[1]) * cos(x[2]) * cos(x[3])                 # harmonic: Δu = 0
#gradu(x)    = SVector(sqrt(2) * exp(sqrt(2) * x[1]) * cos(x[2]) * cos(x[3]),
#                    -exp(sqrt(2) * x[1]) * sin(x[2]) * cos(x[3]),
#                    -exp(sqrt(2) * x[1]) * cos(x[2]) * sin(x[3]))
#g_d   = [gradu(q.coords) for q in Ωₕ_quad]
#du_b  = [dot(gradu(q.coords), q.normal) for q in Γₕ_quad]   # ∂_ν u = g⋅ν
#w_ref = -(S_b2d_std * du_b)                                     # W[∇u] = -S[∂_ν u]
#w_app = W_d2d * g_d

#Ψ(x) = exp(x[1] + x[2]) * cos(x[3])
#gradΨ(x) = SVector(exp(x[1] + x[2]) * cos(x[3]),
#                   exp(x[1] + x[2]) * cos(x[3]),
#                   -exp(x[1] + x[2]) * sin(x[3]))
#g(x) = -1*SVector(1/3 * exp(x[1] + x[2]) * cos(x[3]),
#               1/3 * exp(x[1] + x[2]) * cos(x[3]),
#               1/3 * exp(x[1] + x[2]) * sin(x[3]) )

α = π; β = α; γ = α;
Ψ(x) = cos(α * x[1]) * sin(β * x[2]) * cos(γ * x[3])
gradΨ(x) = SVector(-α*sin(α * x[1]) * sin(β * x[2]) * cos(γ * x[3]),
                   β*cos(α * x[1]) * cos(β * x[2]) * cos(γ * x[3]),
                   -γ*cos(α * x[1]) * sin(β * x[2]) * sin(γ * x[3]))
g(x) = SVector(α * sin(α * x[1]) * sin(β * x[2]) * cos(γ * x[3]), 
               -β * cos(α * x[1]) * cos(β * x[2]) * cos(γ * x[3]),
               γ * cos(α * x[1]) * sin(β * x[2]) * sin(γ * x[3]))
g_d = [g(q.coords) for q in Ωₕ_quad]
Ψ_b = [Ψ(q.coords) for q in Γₕ_quad]
Ψ_d = [Ψ(q.coords) for q in Ωₕ_quad]
gradΨ_d = [gradΨ(q.coords) for q in Ωₕ_quad]
BvΨ_plus_gnu = [dot(gradΨ(q.coords), q.normal) for q in Γₕ_quad] + [dot(g(q.coords), q.normal) for q in Γₕ_quad]
w_ref = gradΨ_d + D_b2d_grad * Ψ_b - S_b2d_grad * BvΨ_plus_gnu
#Id = [1 0 0; 0 1 0; 0 0 1]
#Sdotg = similar(w_ref)
#for i in 1:length(w_ref)
#    Sdotg[i] = -1/3 * Id * g_d[i]
#end
#w_ref = Ψ_d + D_b2d_std * Ψ_b - S_b2d_std * BvΨ_plus_gnu
w_app = X_d2d * g_d
err = norm(w_app - w_ref, Inf)
ndofs = length(w_app)
@show ndofs, meshsize, err
