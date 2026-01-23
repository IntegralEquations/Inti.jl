## Load Inti and prepare the environment with weak dependencies
using Inti
Inti.stack_weakdeps_env!(; update = false)

#-

## Load the necessary packages
using StaticArrays
using LinearAlgebra
using Meshes
using Gmsh
using GLMakie
using HMatrices
using IterativeSolvers
using LinearMaps

#-

## Physical parameters
λ = 0.25 #
k = 2π / λ # wavenumber
θ = π / 4 # angle of incident wave

## Mesh parameters
meshsize = λ / 10 # mesh size
gorder = 2 # polynomial order for geometry
qorder = 4 # quadrature order

## Import the mesh
filename = joinpath(Inti.PROJECT_ROOT, "docs", "assets", "elliptic_cavity_2D.geo")
gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.open(filename)
gmsh.model.mesh.generate(1)
gmsh.model.mesh.setOrder(gorder)
msh = Inti.import_mesh(; dim = 2)
gmsh.finalize()

## Extract the entities and elements of interest
ents = Inti.entities(msh)
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, ents)
Γ = Inti.boundary(Ω)
Γ_msh = view(msh, Γ)

#-

## The incident field
d = SVector(cos(θ), sin(θ)) # incident direction
uᵢ = (x) -> exp(im * k * dot(x, d)) # incident plane wave

## Create the quadrature
Q = Inti.Quadrature(Γ_msh; qorder)
println("Number of quadrature points: ", length(Q))

## Setup the integral operators
op = Inti.Helmholtz(; dim = 2, k)
S, D = Inti.single_double_layer(;
    op
    target = Q,
    source = Q,
    correction = (method = :none,),
    # compression = (method = :hmatrix, tol = 1e-4),
    compression = (method = :none,),
)

## Right-hand side given by Dirichlet trace of plane wave
g = map(Q) do q
    # normal derivative of e^{ik*d⃗⋅x}
    x, ν = q.coords, q.normal
    return -uᵢ(x)
end ## Neumann trace on boundary

## Use GMRES to solve the linear system
L = I / 2 + LinearMap(D) - im * k * LinearMap(S)
σ = gmres(L, g; restart = 1000, maxiter = 400, abstol = 1e-4, verbose = true)

## Plot a heatmap of the solution
𝒮, 𝒟 = Inti.single_double_layer_potential(; op, source = Q)
u = (x) -> 𝒟[σ](x) - im * k * 𝒮[σ](x)
xx = yy = range(-2, 2; step = meshsize)
colorrange = (-2, 2)
vals = map(Iterators.product(xx, yy)) do x
    x = SVector(x)
    return Inti.isinside(x, Q) ? Complex(NaN) : uᵢ(x) + u(x)
end
fig, ax, hm = heatmap(
    xx,
    yy,
    real(vals);
    colormap = :viridis,
    colorrange,
    interpolate = true,
    axis = (aspect = DataAspect(), xgridvisible = false, ygridvisible = false),
)
viz!(Γ_msh; segmentsize = 4, color = :white)
Colorbar(fig[1, 2], hm; label = "Re(u)")
display(fig)
##
