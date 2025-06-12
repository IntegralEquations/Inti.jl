using Test
using Inti
using FMM3D
using Gmsh
using LinearAlgebra
using Random
using StaticArrays
using NearestNeighbors
using ForwardDiff
using NonlinearSolve

include("test_utils.jl")

# create a boundary and area meshes and quadrature only once
meshsize = 0.1
r1 = 1.0
r2 = 0.5
Ω, msh = gmsh_torus(; center = [0.0, 0.0, 0.0], r1 = r1, r2 = r2, meshsize = meshsize)
Γ = Inti.external_boundary(Ω)
Γ_msh = view(msh, Γ)

function face_element_on_torus(nodelist, R, r)
    return all([
        (sqrt(node[1]^2 + node[2]^2) - R^2)^2 + node[3]^2 ≈ r^2 for node in nodelist
    ])
end
face_element_on_curved_surface = (nodelist) -> face_element_on_torus(nodelist, r1, r2)

ψ = (v) -> [(r1 + r2*sin(v[1]))*cos(v[2]), (r1 + r2*sin(v[1]))*sin(v[2]), r2*cos(v[1])]
θ = 5 # smoothness order of curved elements
crvmsh = Inti.curve_mesh(
    msh,
    ψ,
    θ,
    50*Int(1/meshsize);
    face_element_on_curved_surface = face_element_on_curved_surface,
)

Γₕ = crvmsh[Γ]
Ωₕ = crvmsh[Ω]

truevol = 2 * π^2 * r2^2 * r1
truesfcarea = 4 * π^2 * r1 * r2

qorder = 2
Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)
@assert isapprox(Inti.integrate(x -> 1, Ωₕ_quad), truevol, rtol = 1e-6)
@assert isapprox(Inti.integrate(x -> 1, Γₕ_quad), truesfcarea, rtol = 1e-6)

qorder = 5
Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)
@assert isapprox(Inti.integrate(x -> 1, Ωₕ_quad), truevol, rtol = 1e-11)
@assert isapprox(Inti.integrate(x -> 1, Γₕ_quad), truesfcarea, rtol = 1e-11)

qorder = 8
Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)
@assert isapprox(Inti.integrate(x -> 1, Ωₕ_quad), truevol, rtol = 1e-12)
@assert isapprox(Inti.integrate(x -> 1, Γₕ_quad), truesfcarea, rtol = 1e-14)

divF = (x) -> x[3] + x[3]^2 + x[2]^3
F = (x) -> [x[1]*x[3], x[2]*x[3]^2, x[2]^3*x[3]]
#divF = (x) -> 1.0
#F = (x) -> 1/3*[x[1], x[2], x[3]]
divtest_vol = Inti.integrate(q -> divF(q.coords), Ωₕ_quad)
divtest_sfc = Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad)
@assert isapprox(divtest_vol, divtest_sfc, rtol = 1e-12)
