using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using HMatrices
using FMM3D
using CairoMakie
using DataStructures

tinit = time() # hide

# ## Problem definition

# In this example we will solve the Poisson equation in a domain $\Omega$ with
# Dirichlet boundary conditions on $\Gamma := \partial \Omega$:
# ```math
#   \begin{align}
#       \Delta u &= f  \quad \text{in } \Omega \\
#       u &= g  \quad \text{on } \partial \Gamma
#   \end{align}
# ```
# where $f : \Omega \to \mathbb{R}$ and $g : \Gamma \to \mathbb{R}$ are given
# functions.
#
# Seeking for a solution $u$ of the form ...

include("../test_utils.jl")
compression = (method = :fmm, tol = 1.0e-13)

meshsize = 0.24
# `n` in the VDIM paper
interpolation_order = 4
qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(interpolation_order)
nothing #hide

r1 = 1.0
r2 = 0.5
# first build boundary meshes for use with BDIM - at higher orders we want to saturate the accuracy in VDIM, so we overresolve the boundaries. Unfortunately BDIM is not stable for large bdry_qorder
bdry_qorder = min(2 * qorder, 7)
meshsize_bdry = meshsize / 4
tmsh = @elapsed begin
    Ω, msh =
        gmsh_torus(; center = [0.0, 0.0, 0.0], r1 = r1, r2 = r2, meshsize = meshsize_bdry)
    #Ω, msh =
    #    gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = meshsize)
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
    θ = min(qorder + 1, interpolation_order + 3) - 1 # smoothness order of curved elements
    crvmsh = Inti.curve_mesh(
        msh,
        ψ,
        θ;
        face_element_on_curved_surface = face_element_on_curved_surface,
    )

    Ωₕ = view(crvmsh, Ω)
    Γₕ = view(crvmsh, Γ)
end
Qgauss = Inti.Gauss(; domain = :triangle, order = bdry_qorder);
Γdict = OrderedDict(E => Qgauss for E in Inti.element_types(Γₕ))
Γₕ_quad = Inti.Quadrature(Γₕ, Γdict)
@info "Mesh generation time: $tmsh"

#cleanup
Inti.clear_entities!()

# now build the volume mesh
tmsh = @elapsed begin
    Ω_coarse, msh_coarse =
        gmsh_torus(; center = [0.0, 0.0, 0.0], r1 = r1, r2 = r2, meshsize = meshsize)
    #Ω, msh =
    #    gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = meshsize)
    Γ_coarse = Inti.external_boundary(Ω_coarse)

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
    θ = min(qorder + 1, interpolation_order + 3) - 1 # smoothness order of curved elements
    crvmsh_coarse = Inti.curve_mesh(
        msh_coarse,
        ψ,
        θ;
        face_element_on_curved_surface = face_element_on_curved_surface,
    )

    Ωₕ_coarse = view(crvmsh_coarse, Ω_coarse)
    Γₕ_coarse = view(crvmsh_coarse, Γ_coarse)
end
@info "Mesh generation time: $tmsh"

Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, Inti.entities(msh))
Γ = Inti.boundary(Ω)

# Use VDIM with the Vioreanu-Rokhlin quadrature rule
Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = qorder);
dict = OrderedDict(E => Q for E in Inti.element_types(Ωₕ_coarse))
Ωₕ_quad = Inti.Quadrature(Ωₕ_coarse, dict)
#Qgauss = Inti.Gauss(; domain = :triangle, order = bdry_qorder);
#Γdict = OrderedDict(E => Qgauss for E in Inti.element_types(Γₕ))
#Γₕ_quad = Inti.Quadrature(Γₕ, Γdict)
vol_err = Inti.integrate(x -> 1, Ωₕ_quad) - 2 * π^2 * r2^2 * r1
@info "Volume error: $vol_err"


# ## Manufactured solution

# For the purpose of comparing our numerical results to an exact solution, we
# will use the method of manufactured solutions. For simplicity, we will take as
# an exact solution

uₑ = (x) -> cos(10 * x[1]) * sin(10 * x[2])

# which yields

fₑ = (x) -> -200 * uₑ(x)

# Here is what the solution looks like:
# qvals = map(Ωₕ_quad) do q
#     return uₑ(q.coords)
# end

# ivals = Inti.quadrature_to_node_vals(Ωₕ_quad, qvals)

# er = ivals - uₑ.(Ωₕ_quad.mesh.nodes)
# norm(er,Inf)
# Inti.write_gmsh_view(Ωₕ, uₑ.(Ωₕ.nodes))

# ## Boundary and integral operators
op = Inti.Laplace(; dim = 3)

## Boundary operators
S_b2b, D_b2b = Inti.single_double_layer(;
    op,
    target = Γₕ_quad,
    source = Γₕ_quad,
    compression,
    correction = (method = :dim,),
)
S_b2d, D_b2d = Inti.single_double_layer(;
    op,
    target = Ωₕ_quad,
    source = Γₕ_quad,
    compression,
    correction = (method = :dim, maxdist = 5 * meshsize, target_location = :inside),
)

## Volume potentials
V_d2d = Inti.volume_potential(;
    op,
    target = Ωₕ_quad,
    source = Ωₕ_quad,
    compression,
    correction = (method = :dim, interpolation_order, S_b2d = S_b2d, D_b2d = D_b2d, boundary = Γₕ_quad),
)
V_d2b = Inti.volume_potential(;
    op,
    target = Γₕ_quad,
    source = Ωₕ_quad,
    compression,
    correction = (
        method = :dim,
        maxdist = 5 * meshsize,
        interpolation_order,
        target_location = :on,
        S_b2d = S_b2b,
        D_b2d = D_b2b,
        boundary = Γₕ_quad,
    ),
)

# We can now solve a BIE for the unknown density $\sigma$:
f = map(Ωₕ_quad) do q
    return fₑ(q.coords)
end
g = map(Γₕ_quad) do q
    return uₑ(q.coords)
end
rhs = V_d2b * f + g

using LinearAlgebra
L = -I / 2 + D_b2b

# If `compression=none` or `compresion=hmatrix` is used above for constructing `D_b2b`, we could alternately use dense linear algebra:
#F = lu(L)
#σ = F \ rhs

using IterativeSolvers
σ, hist =
    gmres(L, rhs; log = true, abstol = 1.0e-14, verbose = false, restart = 100, maxiter = 100)
@show hist

# To check the solution, lets evaluate it at the nodes $\Omega$
uₕ_quad = -(V_d2d * f) + D_b2d * σ
uₑ_quad = map(q -> uₑ(q.coords), Ωₕ_quad)
er = abs.(uₕ_quad - uₑ_quad)
@show norm(er, Inf)

# ## Visualize the solution error using Gmsh

tend = time() # hide
@info "Example completed in $(tend - tinit) seconds" # hide
ndofs = length(Ωₕ_quad)
@info "ndofs: $ndofs"
