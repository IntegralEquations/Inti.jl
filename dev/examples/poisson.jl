using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

# # [Poisson solver](@id poisson)

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](poisson.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/poisson.ipynb)

# !!! note "Important points covered in this example"
#       - Solving a volumetric problem
#       - Using the method of manufactured solutions
#       - ...

using Inti

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

meshsize = 0.05
# `n` in the VDIM paper
interpolation_order = 3
qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(interpolation_order)

# ## Meshing

# We now create the required meshes and quadratures for both $\Omega$ and $\Gamma$:

using Gmsh # this will trigger the loading of Inti's Gmsh extension

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
gmsh_disk(; meshsize, order = 2, name)

Ω, msh = Inti.import_mesh_from_gmsh_file(name; dim = 2)
Γ = Inti.boundary(Ω)

Ωₕ = view(msh, Ω)
Γₕ = view(msh, Γ)
# Use VDIM with the Vioreanu-Rokhlin quadrature rule
Q = Inti.VioreanuRokhlin(; domain = :triangle, order = qorder);
dict = Dict(E => Q for E in Inti.element_types(Ωₕ))
Ωₕ_quad = Inti.Quadrature(Ωₕ, dict)
Γₕ_quad = Inti.Quadrature(Γₕ; qorder)

# ## Manufactured solution

# For the purpose of comparing our numerical results to an exact solution, we
# will use the method of manufactured solutions. For simplicity, we will take as
# an exact solution

uₑ = (x) -> cos(2*x[1]) * sin(2*x[2])

# which yields

fₑ = (x) -> -8 * uₑ(x)

# Here is what the solution looks like:
# qvals = map(Ωₕ_quad) do q
#     return uₑ(q.coords)
# end

# ivals = Inti.quadrature_to_node_vals(Ωₕ_quad, qvals)

# er = ivals - uₑ.(Ωₕ_quad.mesh.nodes)
# norm(er,Inf)
# Inti.write_gmsh_view(Ωₕ, uₑ.(Ωₕ.nodes))

# ## Boundary and integral operators
using FMM2D

pde = Inti.Laplace(; dim = 2)

## Boundary operators
S_b2b, D_b2b = Inti.single_double_layer(;
    pde,
    target = Γₕ_quad,
    source = Γₕ_quad,
    compression = (method = :fmm, tol = 1e-12),
    correction = (method = :dim,),
)
S_b2d, D_b2d = Inti.single_double_layer(;
    pde,
    target = Ωₕ_quad,
    source = Γₕ_quad,
    compression = (method = :fmm, tol = 1e-12),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

## Volume potentials
V_d2d = Inti.volume_potential(;
    pde,
    target = Ωₕ_quad,
    source = Ωₕ_quad,
    compression = (method = :fmm, tol = 1e-12),
    correction = (method = :dim, interpolation_order)
)
V_d2b = Inti.volume_potential(;
    pde,
    target = Γₕ_quad,
    source = Ωₕ_quad,
    compression = (method = :fmm, tol = 1e-12),
    correction = (method = :dim, maxdist = 5 * meshsize, interpolation_order),
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
    gmres(L, rhs; log = true, abstol = 1e-10, verbose = false, restart = 100, maxiter = 100)
@show hist

# To check the solution, lets evaluate it at the nodes $\Omega$
uₕ_quad = -(V_d2d * f) + D_b2d * σ
uₑ_quad = map(q -> uₑ(q.coords), Ωₕ_quad)
er = abs.(uₕ_quad - uₑ_quad)
@show norm(er, Inf)

# ## Visualize the solution error using Gmsh
# er_nodes = Inti.quadrature_to_node_vals(Ωₕ_quad, er)
sol_nodes = uₑ.(Inti.nodes(Ωₕ))
solₕ_nodes = Inti.quadrature_to_node_vals(Ωₕ_quad, uₑ_quad)
er_nodes = abs.(sol_nodes - solₕ_nodes)

gmsh.initialize()
Inti.write_gmsh_model(msh)
Inti.write_gmsh_view!(Ωₕ, er_nodes; name="error")
# Inti.write_gmsh_view!(Ωₕ, sol_nodes; name="solution")
"-nopopup" in ARGS || gmsh.fltk.run()
gmsh.finalize()
