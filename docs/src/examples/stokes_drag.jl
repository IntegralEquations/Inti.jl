using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

#nb ## Environment setup
#nb const DEPENDENCIES = ["Gmsh", "LinearAlgebra", "StaticArrays"];
#nb ## __NOTEBOOK_SETUP__

# # Stokes drag

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](stokes_drag.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/stokes_drag.ipynb)

#=

!!! note "Important points covered in this example"
    - Solving a vector-value problem
    - Usage of curved triangular mesh
    - Post-processing integral quantities

=#

tinit = time() # hide

using Inti
using StaticArrays
using LinearAlgebra
using Gmsh

# parameters
μ = 5.0
R = 4.0
v = 1.0

# create a sphere using gmsh
msh_file = joinpath(tempdir(), "stokes-drag.msh")
gmsh.initialize()
gmsh.model.add("stokes-drag")
# set verbosity level to 0
gmsh.option.setNumber("General.Verbosity", 2)
# set max and min meshsize to meshsize
meshsize = 1.0
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.model.occ.addSphere(0, 0, 0, R)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)
gmsh.write(msh_file)
gmsh.finalize()

## import the geometry and mesh
Inti.clear_entities!()
msh = Inti.import_mesh(msh_file)
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, Inti.entities(msh))
Γ = Inti.boundary(Ω)

## create a quadrature
Γ_msh = view(msh, Γ)
Q = Inti.Quadrature(Γ_msh; qorder = 2)

## check error in surface area
@show length(Q)
@show abs(Inti.integrate(x -> 1, Q) - 4π * R^2)

## the pde and its integral kernels
pde = Inti.Stokes(; dim = 3, μ)
G   = Inti.SingleLayerKernel(pde)
dG  = Inti.DoubleLayerKernel(pde)

## choice of a integral representation
T = SVector{3,Float64}
σ = zeros(T, length(Q))
𝒮 = Inti.IntegralPotential(G, Q)
𝒟 = Inti.IntegralPotential(dG, Q)
u = (x) -> 𝒟[σ](x) - 𝒮[σ](x)

## Dirichlet trace on Q (constant velocity field)
f = map(Q) do q
    return T(v, 0.0, 0.0)
end

Sop = Inti.IntegralOperator(G, Q, Q)
Smat = Inti.assemble_matrix(Sop)

## integral operators defined on the boundary
S, D = Inti.single_double_layer(;
    pde,
    target = Q,
    source = Q,
    compression = (method = :none,),
    correction = (method = :dim,),
)

## combining the operators
L = I / 2 + D + μ * S

## HACK: to solve the resulting system using gmres we need to wrap L so that it
## works on scalars
using IterativeSolvers, LinearMaps

L_ = LinearMap{Float64}(3 * size(L, 1)) do y, x
    σ = reinterpret(T, x)
    μ = reinterpret(T, y)
    mul!(μ, L, σ)
    return y
end

σ_ = reinterpret(Float64, σ)
f_ = reinterpret(Float64, f)

_, hist = gmres!(σ_, L_, f_; abstol = 1e-8, maxiter = 200, restart = 200, log = true)

@show hist

## F = ∫ σ dS
drag = μ * sum(eachindex(Q)) do i
    return σ[i] * Q[i].weight
end

exact = 6π * μ * R * v

@show (norm(drag) - exact) / exact

#-
tend = time() # hide
@info "Example completed in $(tend - tinit) seconds" # hide
