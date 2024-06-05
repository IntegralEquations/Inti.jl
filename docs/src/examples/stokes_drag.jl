using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

#nb ## Environment setup
#nb const DEPENDENCIES = ["Meshes", "HMatrices", "Gmsh", "LinearAlgebra", "GLMakie"];
#nb ## __NOTEBOOK_SETUP__

# # Stokes drag

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](stokes_drag.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/stokes_drag.ipynb)

using Inti
using StaticArrays
using LinearAlgebra
using HMatrices
using Gmsh

# create a sphere using gmsh
msh_file = joinpath(tempdir(), "stokes-drag.msh")
gmsh.initialize()
gmsh.model.add("stokes-drag")
# set max and min meshsize to meshsize
meshsize = 0.2
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.model.occ.addSphere(0, 0, 0, 1)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)
gmsh.write(msh_file)
gmsh.finalize()

# import the geometry and mesh
Inti.clear_entities!()
msh = Inti.import_mesh(msh_file)
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, Inti.entities(msh))
Γ = Inti.boundary(Ω)

# create a quadrature
Γ_msh = view(msh, Γ)
Q = Inti.Quadrature(Γ_msh; qorder = 4)

# check error in surface area
@show length(Q)
@show Inti.integrate(x -> 1, Q) - 4π

# the pde and its integral kernels
pde = Inti.Stokes(; dim = 3, μ = 1.0)
G   = Inti.SingleLayerKernel(pde)
dG  = Inti.DoubleLayerKernel(pde)

# choice of a integral representation
T = SVector{3,Float64}
σ = zeros(T, length(Q))
𝒮 = Inti.IntegralPotential(G, Q)
𝒟 = Inti.IntegralPotential(dG, Q)
u = (x) -> 𝒟[σ](x) - 𝒮[σ](x)

# Dirichlet trace on Q (constant velocity field)
f = map(Q) do q
    return T(1.0, 0.0, 0.0)
end

# integral operators defined on the boundary
using FMM3D
S, D = Inti.single_double_layer(;
    pde,
    target = Q,
    source = Q,
    compression = (method = :none,),
    correction = (method = :dim,),
)

# create a dense approximation of the integral operators
t_dense = @elapsed begin
    S = Inti.BlockMatrix(Sop)
    D = Inti.BlockMatrix(Dop)
end

L0 = I / 2 + (D - S)

# corrections using boundary DIM method
t_sparse = @elapsed begin
    δS, δD = Inti.bdim_correction(pde, Q, Q, S, D)
end

t_axpy = @elapsed begin
    axpy!(1.0, δS, S)
    axpy!(1.0, δD, D)
end

# combining the operators
t_comb = @elapsed begin
    L = axpy!(-1, S, D) # D <- D - S
    foreach(i -> L[i, i] -= I / 2, 1:size(L, 1)) # L <- L + 0.5*I
end

# solving the resulting system using gmres
using IterativeSolvers
gmres!(σ, L0, f; verbose = true, abstol = 1e-8)
gmres!(σ, L, f; verbose = true, abstol = 1e-8)

@show t_dense
@show t_sparse
@show t_axpy
@show t_comb
