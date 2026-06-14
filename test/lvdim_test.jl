# # Testing local vdim

using DynamicPolynomials
using FixedPolynomials
using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using HMatrices
using FMMLIB2D
using Meshes
using DataStructures

#meshsize = 0.001/8
#meshsize = 0.125/8
meshsize = 0.125 / 2 /2
interpolation_order = 2
VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(4)
#VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(interpolation_order)
bdry_qorder = 2 * VR_qorder

function gmsh_disk(; name, meshsize, order = 1, center = (0, 0), paxis = (2, 1))
    return try
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("circle-mesh")
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addDisk(center[1], center[2], 0, paxis[1], paxis[2])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        gmsh.write(name)
    finally
        gmsh.finalize()
    end
end

name = joinpath(@__DIR__, "disk.msh")
gmsh_disk(; meshsize, order = 1, name, paxis = (1, 1))
#gmsh_disk(; meshsize, order = 1, name, paxis = (meshsize * 20, meshsize * 10))

Inti.clear_entities!() # empty the entity cache
msh = Inti.import_mesh(name; dim = 2)
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
Γ = Inti.boundary(Ω)

Ωₕ = msh[Ω]
Γₕ = msh[Γ]
Ωₕ_Sub = view(msh, Ω)
Γₕ_Sub = view(msh, Γ)

tquad = @elapsed begin
    # Use VDIM with the Vioreanu-Rokhlin quadrature rule for Ωₕ
    Q = Inti.VioreanuRokhlin(; domain = :triangle, order = VR_qorder)
    dict = OrderedDict(E => Q for E in Inti.element_types(Ωₕ))
    Ωₕ_quad = Inti.Quadrature(Ωₕ, dict)
    Ωₕ_Sub_quad = Inti.Quadrature(Ωₕ_Sub, dict)
    # Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorders[1])
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = bdry_qorder)
    Γₕ_Sub_quad = Inti.Quadrature(Γₕ_Sub; qorder = bdry_qorder)
end
@info "Quadrature generation time: $tquad"

#k = 0.1 / meshsize
#k = 1.0
k = 0
op = k == 0 ? Inti.Laplace(; dim = 2) : Inti.Helmholtz(; dim = 2, k)

## Boundary operators
tbnd = @elapsed begin
    S_b2d, D_b2d = Inti.single_double_layer(;
        op,
        target = Ωₕ_quad,
        source = Γₕ_quad,
        compression = (method = :fmm, tol = 1.0e-14),
        correction = (method = :dim, maxdist = 5 * meshsize, target_location = :inside),
    )
end
@info "Boundary operators time: $tbnd"

## Volume potentials
#tvol = @elapsed begin
V_d2d = Inti.volume_potential(;
    op,
    target = Ωₕ_quad,
    source = Ωₕ_quad,
    compression = (method = :fmm, tol = 1.0e-14),
    correction = (
        method = :ldim,
        mesh = Ωₕ,
        interpolation_order,
        quadrature_order = VR_qorder,
        bdry_nodes = Γₕ.nodes,
        maxdist = 5 * meshsize,
        meshsize = meshsize,
        boundary = Γₕ_quad,
    ),
)
#end
#@info "Volume potential time: $tvol"

using ElementaryPDESolutions
import ElementaryPDESolutions: Polynomial

k0 = 1.1
θ = (cos(π / 3), sin(π / 3))
#u = (x) -> exp(im * k0 * dot(x, θ))
#du = (x, n) -> im * k0 * dot(θ, n) * exp(im * k0 * dot(x, θ))
u = (x) -> cos(k0 * dot(x, θ))
du = (x, n) -> -k0 * dot(θ, n) * sin(k0 * dot(x, θ))
f = (x) -> -1 * (k^2 - k0^2) * u(x)

#I = (2, 0)
# `L` must match the major semi-axis passed to `gmsh_disk` so that f = (x/L)²
# is O(1) on the domain; otherwise norm(er, Inf) is inflated by the constant
# 1/L² while the relative error is unaffected.
#L = 1.0
#f = Polynomial(I => 1 / L^2)
#u = if k == 0
#    convert_coefs(ElementaryPDESolutions.solve_laplace(-f), Float64)
#else
#    convert_coefs(ElementaryPDESolutions.solve_helmholtz(-f; k), ComplexF64)
#end
#gradu = ElementaryPDESolutions.gradient(u)
#du = (x, n) -> map(zip(gradu, 1:length(n))) do v
#
#    p = v[1](x)
#    normal = n[v[2]]
#    return p[1]*normal[1] + p[2]*normal[2]
#
#end
#function du(x, n)
#    accum = 0.0
#    for i in 1:length(n)
#        accum+= gradu[i](x) * n[i]
#    end
#    return accum
#end


#s  = 4
#u  = (x) -> 1 / (k^2 - k0^2) * exp(im * k0 * dot(x, θ)) + 1 / (k^2 - 4 * s) * exp(-s * norm(x)^2)
#du = (x, n) -> im * k0 * dot(θ, n) / (k^2 - k0^2) * exp(im * k0 * dot(x, θ)) - 2 * s / (k^2 - 4 * s) * dot(x, n) * exp(-s * norm(x)^2)
#f  = (x) -> exp(im * k0 * dot(x, θ)) + 1 / (k^2 - 4 * s) * (4 * s^2 * norm(x)^2 - 4 * s + k^2) * exp(-s * norm(x)^2)

u_d = map(q -> u(q.coords), Ωₕ_quad)
u_b = map(q -> u(q.coords), Γₕ_quad)
du_b = map(q -> du(q.coords, q.normal), Γₕ_quad)
f_d = map(q -> f(q.coords), Ωₕ_quad)

vref = u_d + D_b2d * u_b - S_b2d * du_b
#vref = -u_d - D_b2d * u_b + S_b2d * du_b
vapprox = V_d2d * f_d
er = vref - vapprox

ndofs = length(er)

@show ndofs, meshsize, k, norm(er, Inf), norm(er, Inf) / norm(vref, Inf)

## ---- exterior evaluation just outside Γ ----
δoff = 0.5 * meshsize
ext_pts = [q.coords + δoff * q.normal for q in Γₕ_quad]

S_b2e, D_b2e = Inti.single_double_layer(;
    op,
    target = ext_pts,
    source = Γₕ_quad,
    compression = (method = :fmm, tol = 1.0e-14),
    correction = (method = :dim, maxdist = 5 * meshsize, target_location = :outside),
)

V_d2e = Inti.volume_potential(;
    op,
    target = ext_pts,
    source = Ωₕ_quad,
    compression = (method = :fmm, tol = 1.0e-14),
    correction = (
        method = :ldim,
        mesh = Ωₕ,
        interpolation_order,
        quadrature_order = VR_qorder,
        bdry_nodes = Γₕ.nodes,
        maxdist = 5 * meshsize,
        meshsize = meshsize,
        boundary = Γₕ_quad,
        target_location = :outside,
    ),
)

# Green identity for exterior targets: μ = 0, so the u term drops out
vref_ext = D_b2e * u_b - S_b2e * du_b
vapprox_ext = V_d2e * f_d
er_ext = vref_ext - vapprox_ext
@show norm(er_ext, Inf), norm(er_ext, Inf) / norm(vref_ext, Inf)
