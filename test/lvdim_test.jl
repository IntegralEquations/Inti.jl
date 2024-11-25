# # Testing local vdim

using DynamicPolynomials
using FixedPolynomials
using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using HMatrices
using FMMLIB2D
using GLMakie
using Meshes

#meshsize = 0.001/8
meshsize = 0.000125
interpolation_order = 4
VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(4)
bdry_qorder = 2 * VR_qorder

function gmsh_disk(; name, meshsize, order = 1, center = (0, 0), paxis = (2, 1))
    try
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
gmsh_disk(; meshsize, order = 2, name, paxis = (meshsize * 20, meshsize * 10))

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
    dict = Dict(E => Q for E in Inti.element_types(Ωₕ))
    Ωₕ_quad = Inti.Quadrature(Ωₕ, dict)
    Ωₕ_Sub_quad = Inti.Quadrature(Ωₕ_Sub, dict)
    # Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorders[1])
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = bdry_qorder)
    Γₕ_Sub_quad = Inti.Quadrature(Γₕ_Sub; qorder = bdry_qorder)
end
@info "Quadrature generation time: $tquad"

k0 = 1
k  = 0
θ  = (cos(π / 3), sin(π / 3))
#u  = (x) -> exp(im * k0 * dot(x, θ))
#du = (x,n) -> im * k0 * dot(θ, n) * exp(im * k0 * dot(x, θ))
u  = (x) -> cos(k0 * dot(x, θ))
du = (x, n) -> -k0 * dot(θ, n) * sin(k0 * dot(x, θ))
f  = (x) -> (k^2 - k0^2) * u(x)

#s  = 4
#u  = (x) -> 1 / (k^2 - k0^2) * exp(im * k0 * dot(x, θ)) + 1 / (k^2 - 4 * s) * exp(-s * norm(x)^2)
#du = (x, n) -> im * k0 * dot(θ, n) / (k^2 - k0^2) * exp(im * k0 * dot(x, θ)) - 2 * s / (k^2 - 4 * s) * dot(x, n) * exp(-s * norm(x)^2)
#f  = (x) -> exp(im * k0 * dot(x, θ)) + 1 / (k^2 - 4 * s) * (4 * s^2 * norm(x)^2 - 4 * s + k^2) * exp(-s * norm(x)^2)

u_d = map(q -> u(q.coords), Ωₕ_quad)
u_b = map(q -> u(q.coords), Γₕ_quad)
du_b = map(q -> du(q.coords, q.normal), Γₕ_quad)
f_d = map(q -> f(q.coords), Ωₕ_quad)

op = k == 0 ? Inti.Laplace(; dim = 2) : Inti.Helmholtz(; dim = 2, k)

## Boundary operators
tbnd = @elapsed begin
    S_b2d, D_b2d = Inti.single_double_layer(;
        op,
        target = Ωₕ_quad,
        source = Γₕ_quad,
        compression = (method = :fmm, tol = 1e-14),
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
    compression = (method = :fmm, tol = 1e-14),
    correction = (
        method = :ldim,
        mesh = Ωₕ,
        interpolation_order,
        quadrature_order = VR_qorder,
        bdry_nodes = Γₕ.nodes,
        maxdist = 5 * meshsize,
        meshsize = meshsize,
    ),
)
#end
#@info "Volume potential time: $tvol"

vref    = -u_d - D_b2d * u_b + S_b2d * du_b
vapprox = V_d2d * f_d
er      = vref - vapprox

ndofs = length(er)

@show ndofs, meshsize, norm(er, Inf)
