using Markdown                       #src
import Pkg                           #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                #src

# # [Poisson solver](@id poisson)

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](poisson.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/poisson.ipynb)

# !!! note "Important points covered in this example"
#       - Solving a volumetric problem
#       - Using the method of manufactured solutions

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

meshsize = 0.1
qorder = 2

# ## Meshing

# We now create the required meshes and quadratures for both $\Omega$ and $\Gamma$:

using Gmsh # this will trigger the loading of Inti's Gmsh extension

function gmsh_disk(; name, meshsize, order = 1, center = (0, 0), paxis = (2, 1))
    try
        gmsh.initialize()
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
gmsh_disk(; meshsize, order = 1, name)

Ω, msh = Inti.gmsh_read_msh(name; dim = 2)
@show Ω
#-
Γ = Inti.boundary(Ω)
@show Γ
#-
@show msh

Ωₕ = view(msh, Ω)
Γₕ = view(msh, Γ)
Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder)
Γₕ_quad = Inti.Quadrature(Γₕ; qorder)

# ## Manufactured solution

# For the purpose of comparing our numerical results to an exact solution, we
# will use the method of manufactured solutions. For simplicity, we will take as
# an exact solution

uₑ = (x) -> cos(x[1]) * sin(x[2])

# which yields

fₑ = (x) -> -2 * uₑ(x)

# ## Boundary and integral operators
using HMatrices

pde = Inti.Laplace(; dim = 2)

## Boundary operators
S_b2b, D_b2b = Inti.single_double_layer(;
    pde,
    target = Γₕ_quad,
    source = Γₕ_quad,
    compression = (method = :none,),
    correction = (method = :dim,),
)
S_b2d, D_b2d = Inti.single_double_layer(;
    pde,
    target = Ωₕ_quad,
    source = Γₕ_quad,
    compression = (method = :none,),
    correction = (method = :dim, maxdist = 5*meshsize),
)

## Volume potentials
V_d2d = Inti.volume_potential(;
    pde,
    target = Ωₕ_quad,
    source = Ωₕ_quad,
    compression = (method = :none,),
    correction = (method = :dim,),
)
V_d2b = Inti.volume_potential(;
    pde,
    target = Γₕ_quad,
    source = Ωₕ_quad,
    compression = (method = :none,),
    correction = (method = :dim, maxdist = 5*meshsize),
)

# We can now solve a BIE for the unknown density $\sigma$:
f = map(Ωₕ_quad) do q
    return fₑ(q.coords)
end
g = map(Γₕ_quad) do q
    return uₑ(q.coords)
end
g̃ = V_d2b * f + g

using LinearAlgebra
L = -I / 2 + D_b2b
σ = L \ g̃

# To check the solution, lets evaluate it at the quadrature nodes of $\Omega$
# and
uₕ_quad = -V_d2d * f + D_b2d * σ
uₑ_quad = map(q->uₑ(q.coords),Ωₕ_quad)
er = uₕ_quad - uₑ_quad
@show norm(er, Inf)

# ## Visualize the solution error using Gmsh
gmsh.initialize()
gmsh.open(name)
v1 = gmsh.view.add("Solution error")
# gmsh.view.addModelData(v1, 0, "disk", ElementData, tags)
gmsh.finalize()
