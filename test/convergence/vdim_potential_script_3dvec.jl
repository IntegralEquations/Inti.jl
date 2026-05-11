# # High-order convergence of vdim

using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using HMatrices
using FMM3D
using CairoMakie

include(joinpath(@__DIR__, "../test_utils.jl"))
#compression = (method = :hmatrix, tol = 1.0e-8)
compression = (method = :fmm, tol = 1.0e-13)

meshsize = 0.05
r1 = 1.0
tmsh = @elapsed begin
    r2 = 0.5
    Ω, msh =
        gmsh_ball(; center = [0.0, 0.0, 0.0], radius = r1, meshsize = meshsize)
    Γ = Inti.external_boundary(Ω)
    Ωₕ = view(msh, Ω)
    Γₕ = view(msh, Γ)
end
@info "Mesh generation time: $tmsh"

interpolation_order = 3
VR_qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(interpolation_order)
bdry_qorder = 2 * VR_qorder

tquad = @elapsed begin
    # Use VDIM with the Vioreanu-Rokhlin quadrature rule for Ωₕ
    Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = VR_qorder)
    Ωₕ_quad = Inti.Quadrature(Ωₕ, Q)
    # Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorders[1])
    Qbdry = Inti.Gauss(; domain = :triangle, order = bdry_qorder)
    Γₕ_quad = Inti.Quadrature(Γₕ, Qbdry)
end
@info "Quadrature generation time: $tquad"

μ = 1.0
op = Inti.Stokes(; dim = 3, μ)

## Boundary operators
tbnd = @elapsed begin
    S_b2d, D_b2d = Inti.single_double_layer(;
        op,
        target = Ωₕ_quad,
        source = Γₕ_quad,
        compression,
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
        compression,
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

freq = 5.0
u = (x) -> SVector{3, Float64}(cos(freq * x[2]), sin(freq * x[3]), cos(freq * x[1]))
p = (x) -> sin(x[3]) * cos(x[2]) * cos(x[1])
gradu = (x) -> SMatrix{3, 3, Float64}([0 0 -freq * sin(freq * x[1]); -freq * sin(freq * x[2]) 0 0; 0 freq * cos(freq * x[3]) 0])
gradu_plustranspose = (x) -> gradu(x) + gradu(x)'
du = (x, n) -> -p(x) * n + gradu_plustranspose(x) * n
f = (x) -> μ * freq^2 * u(x) + SVector{3, Float64}(-sin(x[3]) * cos(x[2]) * sin(x[1]), -sin(x[3]) * sin(x[2]) * cos(x[1]), cos(x[3]) * cos(x[2]) * cos(x[1]))

u_d = map(q -> u(q.coords), Ωₕ_quad)
u_b = map(q -> u(q.coords), Γₕ_quad)
du_b = map(q -> du(q.coords, q.normal), Γₕ_quad)
f_d = map(q -> f(q.coords), Ωₕ_quad)

vref = u_d + D_b2d * u_b - S_b2d * du_b
vapprox = V_d2d * f_d
er = vref - vapprox

ndofs = length(er)

@show ndofs, meshsize, norm(er, Inf)

# verifies VDIM on polynomials -- Working
#basis = Inti.polynomial_solutions_vdim(op, interpolation_order)
#idx = 3
#N = Inti.ambient_dimension(Ωₕ)
#c = SVector(ntuple(i -> rand(), N)...)
#f = (q) -> basis[idx].source(q) * c
#u = (q) -> basis[idx].solution(q) * c
#t = (q) -> basis[idx].neumann_trace(q) * c
#u_d = map(q -> u(q), Ωₕ_quad)
#u_b = map(q -> u(q), Γₕ_quad)
#du_b = map(q -> t(q), Γₕ_quad)
#f_d = map(q -> f(q), Ωₕ_quad)
#
## Compute reference solution: -u - D*u_b + S*du_b
## This comes from Green's representation: u = S*t - D*u + V*f
## So V*f = u - S*t + D*u, and we test -V*f = S*t - D*u - u
#vref = -u_d - D_b2d * u_b + S_b2d * du_b
#
## Compute uncorrected approximation
#vapprox = V_d2d * f_d
#er = vref - vapprox
