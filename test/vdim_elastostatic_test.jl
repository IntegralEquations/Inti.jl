# # High-order convergence of vdim

using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using HMatrices
using CairoMakie

meshsize = 0.2
meshorder = 1
bdry_qorder = 8
interpolation_order = 2

Inti.clear_entities!()
gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(meshorder)
msh = Inti.import_mesh(; dim = 2)
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
gmsh.finalize()

Γ = Inti.external_boundary(Ω)
Ωₕ = view(msh, Ω)
Γₕ = view(msh, Γ)
##

VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(interpolation_order)

# Use VDIM with the Vioreanu-Rokhlin quadrature rule for Ωₕ
Q = Inti.VioreanuRokhlin(; domain = :triangle, order = VR_qorder)
dict = Dict(E => Q for E in Inti.element_types(Ωₕ))
Ωₕ_quad = Inti.Quadrature(Ωₕ, dict)
# Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorders[1])
Γₕ_quad = Inti.Quadrature(Γₕ; qorder = bdry_qorder)

##

# build exact solution
μ = 1.0
λ = 1.0

u  = (x) -> SVector(1.0, 1.0)
du = (x, n) -> SVector(0.0, 0.0)
f  = (x) -> SVector(0.0, 0.0)

u_d = map(q -> u(q.coords), Ωₕ_quad)
u_b = map(q -> u(q.coords), Γₕ_quad)
du_b = map(q -> du(q.coords, q.normal), Γₕ_quad)
f_d = map(q -> f(q.coords), Ωₕ_quad)

op = Inti.Elastostatic(; λ = λ, μ = μ, dim = 2)

# m, d, n = Inti._polynomial_solutions_vec(op, 1)
# x = @SVector rand(2)
# m[1](x)
# d[1](x)
# n[1]((; coords = x, normal = x))

## Boundary operators
S_b2d, D_b2d = Inti.single_double_layer(;
    op,
    target = Ωₕ_quad,
    source = Γₕ_quad,
    compression = (method = :none, tol = 1e-14),
    correction = (method = :dim, maxdist = 5 * meshsize, target_location = :inside),
)

## Volume potentials
V_d2d = Inti.volume_potential(;
    op,
    target = Ωₕ_quad,
    source = Ωₕ_quad,
    compression = (method = :none, tol = 1e-14),
    correction = (method = :dim, interpolation_order),
)

vref    = -u_d - D_b2d * u_b + S_b2d * du_b
vapprox = V_d2d * f_d
er      = vref - vapprox

ndofs = length(er)
@show ndofs, meshsize, norm(er, Inf), t
