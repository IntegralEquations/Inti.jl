# # High-order convergence of vdim

using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using HMatrices
using FMM3D
using CairoMakie

include("../test_utils.jl")

meshsize = 0.1
r1 = 1.0
tmsh = @elapsed begin
    r2 = 0.5
    Ω, msh =
        gmsh_torus(; center = [0.0, 0.0, 0.0], r1 = r1, r2 = r2, meshsize = meshsize)
    Γ = Inti.external_boundary(Ω)

    function face_element_on_torus(nodelist, R, r)
        return all(
            [
                (sqrt(node[1]^2 + node[2]^2) - R^2)^2 + node[3]^2 ≈ r^2 for node in nodelist
            ]
        )
    end
    face_element_on_curved_surface =
        (nodelist) -> face_element_on_torus(nodelist, r1, r2)

    ψ =
        (v) ->
    [(r1 + r2 * sin(v[1])) * cos(v[2]), (r1 + r2 * sin(v[1])) * sin(v[2]), r2 * cos(v[1])]
    θ = 5 # smoothness order of curved elements
    crvmsh = Inti.curve_mesh(
        msh,
        ψ,
        θ;
        face_element_on_curved_surface = face_element_on_curved_surface,
    )

    Ωₕ = view(crvmsh, Ω)
    Γₕ = view(crvmsh, Γ)
end
@info "Mesh generation time: $tmsh"

interpolation_order = 2
VR_qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(interpolation_order)
bdry_qorder = 2 * VR_qorder

tquad = @elapsed begin
    # Use VDIM with the Vioreanu-Rokhlin quadrature rule for Ωₕ
    Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = VR_qorder)
    dict = Dict(E => Q for E in Inti.element_types(Ωₕ))
    Ωₕ_quad = Inti.Quadrature(Ωₕ, dict)
    # Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorders[1])
    Qbdry = Inti.Gauss(; domain = :triangle, order = bdry_qorder)
    dictbdry = Dict(E => Qbdry for E in Inti.element_types(Γₕ))
    Γₕ_quad = Inti.Quadrature(Γₕ, dictbdry)
end
@info "Quadrature generation time: $tquad"

k0 = π
k = 0
θ = (sin(π / 3) * cos(π / 3), sin(π / 3) * sin(π / 3), cos(π / 3))
#u  = (x) -> exp(im * k0 * dot(x, θ))
#du = (x,n) -> im * k0 * dot(θ, n) * exp(im * k0 * dot(x, θ))
u = (x) -> cos(k0 * dot(x, θ))
du = (x, n) -> -k0 * dot(θ, n) * sin(k0 * dot(x, θ))
f = (x) -> (k^2 - k0^2) * u(x)

u_d = map(q -> u(q.coords), Ωₕ_quad)
u_b = map(q -> u(q.coords), Γₕ_quad)
du_b = map(q -> du(q.coords, q.normal), Γₕ_quad)
f_d = map(q -> f(q.coords), Ωₕ_quad)

op = k == 0 ? Inti.Laplace(; dim = 3) : Inti.Helmholtz(; dim = 3, k)

## Boundary operators
tbnd = @elapsed begin
    S_b2d, D_b2d = Inti.single_double_layer(;
        op,
        target = Ωₕ_quad,
        source = Γₕ_quad,
        compression = (method = :fmm, tol = 1.0e-8),
        correction = (method = :dim, maxdist = 5 * meshsize, target_location = :inside),
    )
end
@info "Boundary operators time: $tbnd"

## Volume potentials
tvol = @elapsed begin
    V_d2d = Inti.volume_potential(;
        op,
        target = Ωₕ_quad,
        source = Ωₕ_quad,
        compression = (method = :fmm, tol = 1.0e-8),
        correction = (
            method = :dim,
            interpolation_order,
            maxdist = 5 * meshsize,
            boundary = Γₕ_quad,
            S_b2d = S_b2d,
            D_b2d = D_b2d,
        ),
    )
end
@info "Volume potential time: $tvol"

vref = -u_d - D_b2d * u_b + S_b2d * du_b
vapprox = V_d2d * f_d
er = vref - vapprox

ndofs = length(er)

@show ndofs, meshsize, norm(er, Inf)
