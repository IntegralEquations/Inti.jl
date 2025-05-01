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
boundary integral equations. The problem involves determining the displacement jump field
$\boldsymbol{\phi}$ in an infinite elastic domain containing a disk-shaped crack. It is
possible to show that the problem can be reduced to a boundary integral equation of the form
(e.g. [bonnet1995; Chapter 13](@cite)):

```math
T[\boldsymbol{\phi}] = -\boldsymbol{f},
```

where $T$ represents the integral operator associated with the hypersingular kernel, defined
on the crack surface $\Gamma$; $\boldsymbol{f}$ is the applied traction on the boundary,
which is symmetric on the two crack lips; and $\boldsymbol{\phi}$ is the so-called crack
opening displacement (COD), defined as the "displacement" jump that occurs through the crack
: $\boldsymbol{\phi}=\boldsymbol u^+-\boldsymbol u^-$.

!!! details "Details"
    Being considered an open surface, the crack $\Gamma$ is arbitrarily extended onto a
    closed surface $\tilde\Gamma$. Then, we consider $\boldsymbol{u}^+$ and
    $\boldsymbol{u}^-$ as the interior and exterior displacements, depending on the
    convention used. The crack opening displacement is then defined as the difference
    between the two displacements at the two crack lips, mathematically superposed. It has
    to be understood as a mathematical limit of the displacement field as a point approaches
    one lip or the other. This method is called the Displacement Discontinuity Method.

This example demonstrates the formulation, solution, and visualization of the problem,
highlighting the use of integral operators.

## Geometry and mesh

The domain is a disk of radius 1 on the plane $z=0$. We use the GMSH library to create the
mesh, and `Inti`'s [`import_mesh`](@ref) function to import it.

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

Note that we have used second-order elements for the mesh, which is useful for better
representing the edges of the circular crack.

## Integral operators

We will now build an approximation to $T$ using:

- A hierarchical matrix representation for the integral operator
- An adaptive correction to account for the singular and nearly singular interactions

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
nothing # hide
```

## Boundary conditions

For the boundary conditions, we consider a constant normal loading on the crack surface,
simply given by:

```@example crack_elasticity
f = 1.0
t = -[SVector(0,0,f) for _ in Q]
nothing # hide
```

Note that we used an `SVector` to represent the traction at a point on the crack surface.
For vector-valued problems, `SVector`s and `SMatrix`s are often used to represent vectors
and tensors, respectively, since their size is known at compile time (and small). This
avoids the overhead of dynamic arrays.

We are now ready to compute the approximate solution.

## Solution

The exact solution for this problem can be obtained by separation of variables in
cylindrical coordinates, and can be shown to be:

```@example crack_elasticity
σ = 1
φ₃(r) = 4*(1-ν)*σ / (π*μ) * sqrt(1-r^2)
uexact(x) = SVector(0, 0, φ₃(norm(x)))
```

To compute the approximate solution, we will need to solve the linear system:

```math
T[\boldsymbol{\phi}] = \boldsymbol{f},
```

where $\boldsymbol{\phi}$ is the unknown vector of displacements. One difficulty that arises
is related to the fact that in our implementation, both `\phi` and `f` are represented as
`Vector`s of `SVector`s. While convenient for some operations, this can lead to difficulties
when trying to solve the linear system since most linear algebra libraries expect matrices
over a scalar field (usually either ``\mathbb{R}`` or ``\mathbb{C}``). To address this, we
will write a short function `solve` that will convert between the *vector of vectors* and
the vector of *scalars*.

```@example crack_elasticity
using IterativeSolvers
function solve!(u, T₀, δT, t)
    @assert eltype(T₀) == eltype(δT) == SMatrix{3,3,Float64,9}
    @assert eltype(t) == eltype(u) == SVector{3,Float64}
    # write a LinearMap over scalars by reinterpreting them as vectors of SVectors, 
    # applying our operators T₀ and δT, and converting back before returning
    L_ = LinearMap{Float64}(3 * size(T₀, 1)) do y, x
        σ = reinterpret(SVector{3,Float64}, x)
        μ = reinterpret(SVector{3,Float64}, y)
        mul!(μ, T₀, σ)
        mul!(μ, δT, σ, 1, 1)
        return y
    end
    # flatten our input vectors and call gmres on the Float64 version
    u_ = reinterpret(Float64, u)
    t_ = reinterpret(Float64, t)
    u_, gmres_hist = gmres!(u_, L_, t_, restart = 1000, maxiter = 1000, log=true)
    @show gmres_hist
    # since u_ is just a reinterpretation of u, we can simply return u when done
    return u
end
solve(T₀, δT, t) = solve!(zero(t), T₀, δT, t)
```

We can now easily call `solve` to obtain our approximate solution:

```@example crack_elasticity
φ = solve(T₀, δT, t)
nothing # hide
```

## Visualization

Next we show a crude visualization by plotting the displacement value at each point of the
quadrature (as a function of the radius), and comparing it to the exact solution. The
displacement is a vector, but we will only plot the $z$ component, which is the only one
that is non-zero in this case.

```@example crack_elasticity
using LinearAlgebra
using GLMakie
r    = map(q -> norm(Inti.coords(q)), Q)
vals = getindex.(φ, 3)
scatter(r, vals, label = "Numerical solution")
lines!(0:0.01:1, φ₃, label = "Exact solution", color = :red, linewidth = 4)
axislegend()
current_figure()
```

Although the solution is not perfect, it captures the general behavior of the displacement
field. One way to make the error smaller is to use a finer mesh and/or higher order
quadrature. An alternative way, however, is to use a weight function to incorporate the
singular behavior of the displacement field near the edge of the crack, as shown next.

## Improving the accuracy

It is beneficial to add a weight function to help the solution being more accurate near the
crack, where the displacement is singular, asymptotically equal to $d^{1/2}$ ($d$ is the
distance from a point to the crack front) according to the Williams' asymptotic expansion.
For this simple example we take the weight function as:

$$w(\boldsymbol x):=\sqrt{1-||\boldsymbol x||}\underset{d\rightarrow
0}{\sim}\sqrt{d(\boldsymbol x)}$$

```@example crack_elasticity
weight(x) = sqrt(1 - norm(x))
```

and define a modified kernel $K_w$ as:

```@example crack_elasticity
Kw = let w = weight, K = K
    (p,q) ->  K(p,q) * w(q.coords)
end
Inti.singularity_order(::typeof(Kw)) = -3
```

With this new kernel, we can build our new integral operator $T_w$ and solve the
displacement jump equation for a modified density $\boldsymbol{\phi}_w = \boldsymbol{\phi} /
w$:

```@example crack_elasticity
Tw_op = Inti.IntegralOperator(Kw, Q)
Tw₀ = Inti.assemble_hmatrix(Tw_op)
δTw = Inti.adaptive_correction(Tw_op; maxdist = 2*meshsize, atol = 1e-2)
φw = solve(Tw₀, δTw, t)
```

We now plot the displacement field again, but this with the weighted kernel approach, and
compare it to the previous approach. Note that we must multiply the solution `φw` by the
weight function to obtain the actual displacement jump `φ`.

```@example crack_elasticity
weights = [weight(q.coords) for q in Q]
scatter(r, getindex.(φw,3) .* weights, label = "Numerical solution (weighted)")
lines!(0:0.01:1, φ₃, label = "Exact solution", color = :red, linewidth = 4)
axislegend()
current_figure()
```

The solution is now much more accurate, especially near the crack front, even though the
same mesh and quadrature were used. This is a common technique in boundary integral
equation: factoring out the asymptotic (non-smooth) behavior of the solution using a weight
function.

Finally, we can visualize the displacement field on the mesh by interpolating the computed
values on the quadrature points to the mesh nodes. We use `Meshes` to visualize the
solution:

```@example crack_elasticity
using Meshes
φ3w_nodes = Inti.quadrature_to_node_vals(Q, getindex.(φw, 3))
msh_nodes = Inti.nodes(Q.mesh)
w_nodes = [weight(x) for x in msh_nodes]
φ3_nodes = φ3w_nodes .* w_nodes
colorrange = extrema(φ3_nodes)
fig = Figure(; size = (800, 600))
ax = Axis3(fig[1, 1])
n = length(Q.mesh.nodes)
viz!(Q.mesh; color = φ3_nodes, interpolate = false, showsegments=true)
cb = Colorbar(fig[1, 2]; label = "φ₃", colorrange)
fig
```
