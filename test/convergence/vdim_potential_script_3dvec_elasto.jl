# # High-order convergence of vdim

using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using HMatrices
using FMM3D
using CairoMakie

include(joinpath(@__DIR__, "../test_utils.jl"))
#compression = (method = :fmm, tol = 1.0e-12)
compression = (method = :hmatrix, tol = 1.0e-8)
#compression = (method = :none,)

meshsize = 0.2
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

interpolation_order = 2
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
λ = 2.0
#op = Inti.Stokes(; dim = 3, μ)
op = Inti.Elastostatic(; dim = 3, μ, λ)

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

# Stokes
#u = (x) -> SVector{3, Float64}(cos(x[2]), sin(x[3]), cos(x[1]))
#p = (x) -> sin(x[3]) * cos(x[2]) * cos(x[1])
#gradu = (x) -> SMatrix{3, 3, Float64}([0 0 -sin(x[1]); -sin(x[2]) 0 0; 0 cos(x[3]) 0])
#gradu_plustranspose = (x) -> gradu(x) + gradu(x)'
#du = (x, n) -> -p(x) * n + gradu_plustranspose(x) * n
#f = (x) -> μ * u(x) + SVector{3, Float64}(-sin(x[3]) * cos(x[2]) * sin(x[1]), -sin(x[3]) * sin(x[2]) * cos(x[1]), cos(x[3]) * cos(x[2]) * cos(x[1]))

# Elastostatics
u = (x) -> SVector{3, Float64}(cos(x[1]) * sin(x[2]), sin(x[3]) * cos(x[2]), cos(x[3]) * sin(x[1]))
graddivu = (x) -> -SVector{3, Float64}(cos(x[1]) * sin(x[2]) + sin(x[3]) * cos(x[1]), sin(x[1]) * cos(x[2]) + sin(x[3]) * cos(x[2]), cos(x[3]) * sin(x[2]) + cos(x[3]) * sin(x[1]))
gradu = (x) -> SMatrix{3, 3, Float64}(-sin(x[1]) * sin(x[2]), 0, cos(x[3]) * cos(x[1]), cos(x[1]) * cos(x[2]), -sin(x[3]) * sin(x[2]), 0, 0, cos(x[3]) * cos(x[2]), -sin(x[3]) * sin(x[1]))
f = (x) -> 2 * μ * u(x) - (μ + λ) * graddivu(x)
divu = (x) -> -sin(x[1]) * sin(x[2]) - sin(x[3]) * sin(x[2]) - sin(x[3]) * sin(x[1])
curlv = (x) -> SVector{3, Float64}(-cos(x[3]) * cos(x[2]), -cos(x[3]) * cos(x[1]), -cos(x[1]) * cos(x[2]))
du = (x, n) -> λ * divu(x) * n + 2 * μ * gradu(x) * n + μ * cross(n, curlv(x))

u_d = map(q -> u(q.coords), Ωₕ_quad)
u_b = map(q -> u(q.coords), Γₕ_quad)
du_b = map(q -> du(q.coords, q.normal), Γₕ_quad)
f_d = map(q -> f(q.coords), Ωₕ_quad)

vref = u_d + D_b2d * u_b - S_b2d * du_b
vapprox = V_d2d * f_d
er = vref - vapprox

ndofs = length(er)

@show ndofs, meshsize, norm(er, Inf)
