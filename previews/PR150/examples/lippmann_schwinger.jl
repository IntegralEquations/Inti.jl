using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

# # [Lippmann Schwinger Solver](@id lippmann_schwinger)

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](lippmann_schwinger.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/lippmann_schwinger.ipynb)

# !!! note "Important points covered in this example"
#       - Using VDIM for volume integral operators
#       - Solving a volume integral equation

using Inti

# ## Problem definition

# In this example we will solve the Lippmann Schwinger Volume Integral Equation
# in a domain $\Omega$:
# ```math
#   \begin{align}
#       u + k^2 \mathcal{V}_k[(1 - \eta) u] &= u^{\textit{inc}}  \quad \text{in } \Omega \\
#   \end{align}
# ```
# where $u^{\textit{inc}} : \Omega \to \mathbb{C}$ is a given free-space Helmholtz
# solution.
#

interpolation_order = 2 # `interpolation_order` corresponds to `n` in the VDIM paper
qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(interpolation_order)

k₁ = 4π
k₂ = 2π
λ₁ = 2π / k₁
λ₂ = 2π / k₂
meshsize = min(λ₁, λ₂) / 7
nothing # hide

# !!! note "Refraction Index Perturbation"
#       A VIE with piecewise-constant refraction index perturbation ``η`` can be
#       verified against a BIE formulation.  Generally, we will want to use a
#       VIE formulation for variable media e.g. `η = (x) -> 1 +
#       .7*exp(-40*(x[1]^2 + x[2]^2))`.
η = (x) -> (k₂ / k₁)^2
nothing # hide

# ## Meshing

# We now create the required meshes and quadratures for both $\Omega$ and $\Gamma$:

using Gmsh # this will trigger the loading of Inti's Gmsh extension

function gmsh_disk(; name, meshsize, order = 1, center = (0, 0), paxis = (1, 1))
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
gmsh_disk(; meshsize, order = 2, name)

msh = Inti.import_mesh(name; dim = 2)
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
Γ = Inti.boundary(Ω)

using OrderedCollections
Ωₕ = view(msh, Ω)
Γₕ = view(msh, Γ)
# Use VDIM with the Vioreanu-Rokhlin quadrature rule
Q = Inti.VioreanuRokhlin(; domain = :triangle, order = qorder);
dict = OrderedDict(E => Q for E in Inti.element_types(Ωₕ))
Γₕ_quad = Inti.Quadrature(Γₕ; qorder)
Ωₕ_quad = Inti.Quadrature(Ωₕ, dict)

# ## Volume Integral Operators and Volume Integral Equations
using FMMLIB2D

op = Inti.Helmholtz(; dim = 2, k = k₁)

# With quadratures constructed on the volume, we can define a discrete approximation
# to the volume integral operator ``\mathcal{V}`` using VDIM.
V_d2d = Inti.volume_potential(;
    op,
    target = Ωₕ_quad,
    source = Ωₕ_quad,
    compression = (method = :fmm, tol = 1.0e-7),
    correction = (method = :dim, interpolation_order),
)

using LinearAlgebra
using LinearMaps

using SpecialFunctions
uⁱ = (x) -> exp(im * k₁ * x[2]) # plane-wave incident field
rhs = map(Ωₕ_quad) do q
    x = q.coords
    return uⁱ(x)
end

# The full VIO incorporates scalar point multiplication using the contrast
# function η, implemented as a composition of `LinearMap`
refr_map_d = map(Ωₕ_quad) do q
    x = q.coords
    return 1 - η(x)
end
apply_refr!(y, x) = y .= refr_map_d .* x
Lη = LinearMap{ComplexF64}(apply_refr!, length(refr_map_d), length(refr_map_d))
L = I + k₁^2 * V_d2d * Lη

# The unknown volumetric field $u$:
using IterativeSolvers
u, hist =
    gmres(L, rhs; log = true, abstol = 1.0e-7, verbose = false, restart = 200, maxiter = 200)
@show hist

𝒱 = Inti.IntegralPotential(Inti.SingleLayerKernel(op), Ωₕ_quad)

# The representation formula gives the solution in $\R^2 \setminus \Omega$:
uˢ = (x) -> uⁱ(x) - k₁^2 * 𝒱[refr_map_d .* u](x)
nothing # hide

# To visualize the solution using Gmsh, let's query it at the triangle vertices  in $\Omega$

solₕ_nodes = Inti.quadrature_to_node_vals(Ωₕ_quad, real(-u))

gmsh.initialize()
Inti.write_gmsh_model(msh)
Inti.write_gmsh_view!(Ωₕ, solₕ_nodes; name = "LS solution")
isinteractive() && gmsh.fltk.run()
gmsh.finalize()
nothing # hide
