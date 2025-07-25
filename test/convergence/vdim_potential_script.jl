# # High-order convergence of vdim

using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using HMatrices
using FMMLIB2D
using CairoMakie

function domain_and_mesh(; meshsize, meshorder = 1)
    Inti.clear_entities!()
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(meshorder)
    msh = Inti.import_mesh(; dim = 2)
    Ω = Inti.Domain(Inti.entities(msh)) do ent
        return Inti.geometric_dimension(ent) == 2
    end
    gmsh.finalize()
    return Ω, msh
end

meshsize = 0.05

interpolation_order = 4
VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(interpolation_order)
bdry_qorder = 2 * VR_qorder

tmesh = @elapsed begin
    Ω, msh = domain_and_mesh(; meshsize)
end
@info "Mesh generation time: $tmesh"

Γ = Inti.external_boundary(Ω)
#Γₕ = msh[Γ]
#Ωₕ = msh[Ω]

ψ = (t) -> [cos(2*π*t), sin(2*π*t)]
θ = 3 # smoothness order of curved elements
crvmsh = Inti.curve_mesh(msh, ψ, θ, 500*Int(1/meshsize))
Ωₕ = view(crvmsh, Ω)
Γₕ = view(crvmsh, Γ)
#
#Γₕ = crvmsh[Γ]
#Ωₕ = crvmsh[Ω]

tquad = @elapsed begin
    # Use VDIM with the Vioreanu-Rokhlin quadrature rule for Ωₕ
    Q = Inti.VioreanuRokhlin(; domain = :triangle, order = VR_qorder)
    dict = Dict(E => Q for E in Inti.element_types(Ωₕ))
    Ωₕ_quad = Inti.Quadrature(Ωₕ, dict)
    # Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorders[1])
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = bdry_qorder)
end
@info "Quadrature generation time: $tquad"

k0 = π
k  = 0
θ = (cos(π / 3), sin(π / 3))
#u  = (x) -> exp(im * k0 * dot(x, θ))
#du = (x,n) -> im * k0 * dot(θ, n) * exp(im * k0 * dot(x, θ))
u  = (x) -> cos(k0 * dot(x, θ))
du = (x, n) -> -k0 * dot(θ, n) * sin(k0 * dot(x, θ))
f  = (x) -> (k^2 - k0^2) * u(x)

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
tvol = @elapsed begin
    V_d2d = Inti.volume_potential(;
        op,
        target = Ωₕ_quad,
        source = Ωₕ_quad,
        compression = (method = :fmm, tol = 1e-14),
        correction = (
            method = :dim,
            interpolation_order,
            maxdist = 5 * meshsize,
            S_b2d = S_b2d,
            D_b2d = D_b2d,
        ),
    )
end
@info "Volume potential time: $tvol"

vref    = -u_d - D_b2d * u_b + S_b2d * du_b
vapprox = V_d2d * f_d
er      = vref - vapprox

ndofs = length(er)

@show ndofs, meshsize, norm(er, Inf)
