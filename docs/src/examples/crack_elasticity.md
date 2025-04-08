# Elastic crack in 2D

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this example"
      - Solving a problem with an open surface (crack)
      - Using the hypersingular operator
      - Defining a custom kernel with a weight function
      - Dealing with vector-valued problems
  
## Problem definition

In this example, we solve a disk crack problem in the context of linear elasticity using
boundary integral equations. The problem involves determining the displacement field
$\boldsymbol{u}$ in an infinite elastic domain containing a disk-shaped crack. It is
possible to show that the problem can be reduced to a boundary integral equation of the
form:

```math
T[\boldsymbol{u}] = -\boldsymbol{f},
```

where $T$ represents the integral operator associated with the hypersingular kernel, and
$\boldsymbol{f}$ is the applied traction on the boundary. This example demonstrates the
formulation, solution, and visualization of the problem, highlighting the use of integral
operators.

## Geometry and mesh

The domain is a disk of radius 1:

```@example crack_elasticity
using Inti
using StaticArrays
using Gmsh

meshsize = 0.2
qorder  = 2 # avoid 3 since it contains a negative weight
rx = ry = 1
gmsh.initialize(String[], false)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.model.occ.addDisk(0.0, 0.0, 0.0, rx, ry)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(1)
msh = Inti.import_mesh(; dim = 3)
Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
Γ_msh = msh[Γ]
Q = Inti.Quadrature(Γ_msh; qorder = 2)
gmsh.finalize()
```

Note that we have used second-order elements for the mesh.

## Integral operators

We will now build an approximation to $T$ using:

- A hierarchical matrix representation
- An adaptive correction to the singularity

```@example crack_elasticity
using HMatrices
using LinearMaps
using LinearAlgebra
#Elastic properties
μ = 1; ν = 0.15;
E = 2*μ*(1+ν)
λ = ν*E / ((1+ν)*(1-2*ν))
op = Inti.Elastostatic(; λ, μ, dim = 3)
K = Inti.HyperSingularKernel(op)
Top = Inti.IntegralOperator(K, Q)
T₀ = Inti.assemble_hmatrix(Top)
δT = Inti.adaptive_correction(Top)
```

## Boundary conditions

We consider a constant normal loading on the crack surface.

```@example crack_elasticity
f = 1.0
t = -[SVector(0,0,f) for _ in Q]
```

## Solution

The exact solution is known for this problem:

```@example crack_elasticity
σ = 1
uz(r) = 4*(1-ν)*σ / (π*μ) * sqrt(1-r^2)
uexact(x) = SVector(0, 0, uz(norm(x)))
```

To compute the approximate solution, we will need to solve the linear system:

```math
T[\boldsymbol{u}] = \boldsymbol{f},
```

where $\boldsymbol{u}$ is the unknown vector of displacements. One difficulty that arises is
related to the fact that in our implementation, both `u` and `f` are represented as
`Vector`s of `SVector`s. While convenient for some operations, this can lead to difficulties
when trying to solve the linear system since most linear algebra libraries expect matrices
of a scalar field (usually either ``\mathbb{R}`` or ``\mathbb{C}``). To address this, we
will write a short function `solve` that will convert between the *vector of vectors* and
the vector of *scalars*.

```@example crack_elasticity
using IterativeSolvers
function solve!(u, T₀, δT, t)
    L_ = LinearMap{Float64}(3 * size(T₀, 1)) do y, x
        σ = reinterpret(SVector{3,Float64}, x)
        μ = reinterpret(SVector{3,Float64}, y)
        mul!(μ, T₀, σ)
        mul!(μ, δT, σ, 1, 1)
        return y
    end
    u_ = reinterpret(Float64, u)
    t_ = reinterpret(Float64, t)
    u_, gmres_hist = gmres!(u_, L_, t_, restart = 1000, maxiter = 1000, log=true)
    @show gmres_hist
    return u
end
solve(T₀, δT, t) = solve!(zero(t), T₀, δT, t)
```

We can now easily call `solve` to obtain our approximate solution:

```@example crack_elasticity
u = solve(T₀, δT, t)
nothing # hide
```

## Visualization

```@example crack_elasticity
using LinearAlgebra
using GLMakie
rr = []
uu = []
for i in eachindex(Q)
    push!(rr, norm(Inti.coords(Q[i])))
    push!(uu, u[i][3])
end
scatter(rr, uu, label = "Numerical solution")
lines!(0:0.01:1, uz, label = "Exact solution", color = :red, linewidth = 4)
axislegend()
current_figure()
```

## Improving the accuracy

It is beneficial to add a weight function:

```@example crack_elasticity
weight(x) = sqrt(1 - norm(x))
Kw = let w = weight, K = K
    (p,q) ->  K(p,q) * w(q.coords)
end
Inti.singularity_order(::typeof(Kw)) = -3
Tw_op = Inti.IntegralOperator(Kw, Q)
Tw₀ = Inti.assemble_hmatrix(Tw_op)
δTw = Inti.adaptive_correction(Tw_op; maxdist = 2*meshsize, atol = 1e-2)
uw = solve(Tw₀, δTw, t)
u = uw .* [weight(q.coords) for q in Q]
```

Check that it is indeed better:

```@example crack_elasticity
using LinearAlgebra
using GLMakie
rr = []
uu = []
for i in eachindex(Q)
    x = Inti.coords(Q[i])
    push!(rr, norm(x))
    push!(uu, u[i][3])
end
scatter(rr, uu, label = "Numerical solution")
lines!(0:0.01:1, uz, label = "Exact solution", color = :red, linewidth = 4)
axislegend()
current_figure()
```

Plotting on the mesh

```@example crack_elasticity
using Meshes
u3w_nodes = Inti.quadrature_to_node_vals(Q, [u[3] for u in uw])
msh_nodes = Inti.nodes(Q.mesh)
w_nodes = [weight(x) for x in msh_nodes]
u3_nodes = u3w_nodes .* w_nodes
colorrange = extrema(u3_nodes)
fig = Figure(; size = (800, 600))
ax = Axis3(fig[1, 1])
n = length(Q.mesh.nodes)
viz!(Q.mesh; color = u3_nodes, interpolate = false, showsegments=true)
cb = Colorbar(fig[1, 2]; label = "uz", colorrange)
fig
```
