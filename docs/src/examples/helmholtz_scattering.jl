using Markdown                       #src
import Pkg                           #src
docsdir = joinpath(@__DIR__,"../..") #src
Pkg.activate(docsdir)                #src

# # [Helmholtz scattering](@id helmholtz_scattering)

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](helmholtz_scattering.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/helmholtz_scattering.ipynb)

# !!! note "Important points covered in this example"
#       - Creating a geometry using the *Gmsh* API
#       - Assembling integral operators and integral potentials
#       - Setting up a sound-soft problem in both 2 and 3 spatial dimensions
#       - Compressing integral operators using [`HMatrices.jl`](https://github.com/WaveProp/HMatrices.jl)
#       - Solving a boundary integral equation
#       - Exporting the solution to *Gmsh* for visualization

# In this tutorial we will show how to solve an acoustic scattering problem in
# the context of Helmholtz equation. We will focus on a *smooth* sound-soft
# obstacle for simplicity, and introduce along the way the necessary techniques
# used to handle some difficulties encountered. We will use various packages
# throughout this example (including of course `Inti.jl`); if they are not on
# your environment, you can install them using `] add <package>` in the REPL.

# In the [following section](@ref helmholtz-soundsoft), we will provide a brief
# mathematical description of the problem (valid in both $2$ and $3$
# dimensions). We will tackle the [two-dimensional problem](@ref
# helmholtz-scattering-2d) first, for which we do not need to worry much about
# performance issues (e.g. compressing the integral operators, or exporting the
# solution to *Gmsh* for visualization). Finally, we present a [three-dimensional
# example](@ref helmholtz-scattering-3d), where we will use
# [`HMatrices.jl`](https://github.com/WaveProp/HMatrices.jl) to compress the
# underlying integral operators.

# ## [Sound-soft problem](@id helmholtz-soundsoft)

# This example concerns the sound-soft acoustic scattering problem.
# Mathematically, this means solving an exterior problem governed by Helmholtz
# equation (time-harmonic acoustics) with a Dirichlet boundary condition. More
# precisely, letting ``\Omega \subset \mathbb{R}^d`` be a bounded domain, and denoting
# by ``\Gamma = \partial \Omega`` its boundary, we wish to solve

# ```math
#     \Delta u + k^2 u = 0 \quad \text{on} \quad \mathbb{R}^d \setminus \bar{\Omega},
# ```

# subject to Dirichlet boundary conditions on ``\Gamma``

# ```math
#     u(\boldsymbol{x}) = g(\boldsymbol{x}) \quad \text{for} \quad \boldsymbol{x} \in \Gamma.
# ```

# and the *Sommerfeld radiation condition* at infinity

# ```math
#     \lim_{|\boldsymbol{x}| \to \infty} \|\boldsymbol{x}|^{(d-1)/2} \left( \frac{\partial u}{\partial |\boldsymbol{x}|} - i k u \right) = 0.
# ```

# Here ``g`` is a (given) boundary datum, and ``k`` is the constant wavenumber.

# For simplicity, we will take ``\Gamma`` circle/sphere, and focus on the
# *plane-wave scattering* problem. This means we will seek a solution ``u`` of
# the form ``u = u_s + u_i``, where ``u_i`` is a known incident field, and
# ``u_s`` is the scattered field we wish to compute.

# !!! note "Complex geometries"
#       The main reason for focusing on such a simple example is two-folded. First,
#       it alleviates the complexities associated with the mesh generation. Second,
#       since exact solutions are known for this problem (in the form of a series),
#       it is easy to assess the accuracy of the solution obtained. In practice, you
#       can use the same techniques to solve the problem on more complex geometries
#       by providing a `.msh` file containing the mesh.

# Using the theory of
# boundary integral equations, we can express ``u_s`` as
#
# ```math
#     u_s(\boldsymbol{r}) = \mathcal{D}[\sigma](\boldsymbol{r}) - i k \mathcal{S}[\sigma](\boldsymbol{r}),
# ```
#
# where ``\mathcal{S}`` is the so-called single layer potential, ``\mathcal{D}``
# is the double-layer potential, and ``\sigma : \Gamma \to \mathbb{C}`` is a
# surface density. This is an indirect formulation (because ``\sigma`` is an
# *auxiliary* density, not necessarily physical) commonly referred to as a
# *combined field formulation*. Taking the limit ``\mathbb{R}^d \setminus \bar
# \Omega \ni x \to \Gamma``, it can be shown that the following equation holds
# on ``\Gamma``:
#
# ```math
#     \left( \frac{\mathrm{I}}{2} + \mathrm{D} - i k \mathrm{S} \right)[\sigma] = g,
# ```
#
# where $\mathrm{I}$ is the identity operator, and $\mathrm{S}$ and $\mathrm{D}$
# are the single- and double-layer operators. This is the **combined field
# integral equation** that we will solve. The boundary data ``g`` is obtained by
# applying the sound-soft condition ``u=0`` on ``\Gamma``, from which it readily
# follows that ``u_s = -u_i`` on ``\Gamma``.
#
# We are now have the necessary background to solve this problem in both 2 and 3
# spatial dimensions. Let's load `Inti.jl`

using Inti

# ## [Two-dimensional scattering](@id helmholtz-scattering-2d)

# We use [Gmsh
# API](https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-application-programming-interface)
# for creating `.msh` file containing the desired geometry and mesh, and then
# import it into `Inti` using [`gmsh_read_msh`](@ref Inti.gmsh_read_msh). Here
# is a function to mesh the circle:
using Gmsh # this will trigger the loading of Inti's Gmsh extension

function gmsh_circle(;name, meshsize, order=1, radius=1, center = (0,0))
    try
        gmsh.initialize()
        gmsh.model.add("circle-mesh")
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addDisk(center[1], center[2], 0, radius, radius)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(1)
        gmsh.model.mesh.setOrder(order)
        gmsh.write(name)
    finally
        gmsh.finalize()
    end
end

# Let us now use to create a `circle.msh` file:

name = joinpath(@__DIR__, "circle.msh")
gmsh_circle(;meshsize=0.1, order = 2, name)

# We can now import the file and parse the mesh and domain information into
# `Inti.jl` using the [`gmsh_read_msh`](@ref Inti.gmsh_read_msh) function:

Ω,msh = Inti.gmsh_read_msh(name; dim=2)
@show Ω
#-
@show msh

# The code above will parse all entities in the `name` of dimension `2` as a
# [`Domain`](@ref Inti.Domain), and also import the underlying mesh (that, in
# this case, is projected into two dimensions by ignoring the third component).
# Note that in the example above, `Ω` is a two-dimensional domain containing a
# single `GmshEntity` which represents the disk. To extract the boundary $\Gamma
# = \partial \Omega$, we can use the [`boundary`](@ref Inti.boundary) function:

Γ = Inti.boundary(Ω)

# !!! tip "Views of a mesh"
#       In `Inti.jl`, you can use domain to create a view of a mesh containing *only
#       the elements in the domain*. For example `view(msh,Γ)` will return an
#       abstract mesh that you can use to iterate over the elements in the boundary
#       of the disk.



# To solve our boundary integral equation usign a Nyström method, we actually
# need a quadrature of our curve/surface (and possibly the normal vectors at the
# quadrature nodes). Once a mesh is available, creating a quadrature object can
# be done via the [`Quadrature`](@ref Inti.Quadrature) constructor, which
# requires passing a mesh the domain that one wishes to generate a quadrature
# for:

Γ = Inti.boundary(Ω)
Q = Inti.Quadrature(msh, Γ; qorder = 4)
nothing #hide

# The object `Q` now contains a quadrature (of order `4`) that can be used to
# solve a boundary integral equation on `Γ`. As a sanity check, let's make sure
# integrating the function `x->1` over `Q` gives an approximation to the perimeter:

@assert abs(Inti.integrate(x->1,Q) - 2π) < 1e-5 #hide
Inti.integrate(x->1,Q) - 2π

# With the [`Quadrature`](@ref Inti.Quadrature) constructed, we now can define
# discrete approximation to the integral operators ``\mathrm{S}`` and
# ``\mathrm{D}`` as follows:

k = 4π
pde   = Inti.Helmholtz(;k, dim=2)
G     = Inti.SingleLayerKernel(pde)
dGdny = Inti.DoubleLayerKernel(pde)
Sop = Inti.IntegralOperator(G,Q,Q)
Dop = Inti.IntegralOperator(dGdny,Q,Q)

# Both `Sop` and `Dop` are of type [`IntegralOperator`](@ref
# Inti.IntegralOperator), which is a subtype of `AbstractMatrix`. There are two
# well-known difficulties related to the discretization of these
# `IntegralOperator`s:
# - The kernel of the integral operator is not smooth, and thus specialized
# quadrature rules are required to accurately approximate the matrix entries for
# which the target and source point lie *close* (relative to some scale) to each
# other.
# - The underlying matrix is dense, and thus the storage and computational cost
# of the operator is prohibitive for large problems unless acceleration
# techniques such as *Fast Multipole Methods* or *Hierarchical Matrices* are
# employed.

# `Inti.jl` tries to provide a modular and transparent interface for dealing
# with both of these difficulties, where the general approach for solving a BIE
# will be to first construct a (possible compressed) naive representation of the
# integral operator where singular and nearly-singular integrals are ignored,
# followed by a the creation of a (sparse) correction intended to account for
# such singular interactions.

# The first part of this example, being two-dimensional, does not require any
# compression algorithm, so we will simply represent the operators `Sop` and
# `Dop` as dense matrices:

S_mat = Matrix(Sop)
D_mat = Matrix(Dop)

# Assembling our discrete approximatio to the combined-field operator is now
# simply a matter of linear algebra:
using LinearAlgebra

L = I/2 + D_mat - im*k*S_mat

# where `I` is the identity matrix. Assuming an incident field along the $x_1$
# direction of the form $u_i =e^{ikx_1}$, the right-hand side of the equation
# can be construted using:

uᵢ = x -> exp(im*k*x[1]) # plane-wave incident field
rhs = map(Q) do q
    x = q.coords
    -uᵢ(x)
end

# !!! note "Iterating over a quadrature"
#       In computing `rhs` above, we used `map` to evaluate the incident field at
#       all quadrature nodes. When iterating over `Q`, the iterator returns a
#       [`QuadratureNode`](@ref Inti.QuadratureNode), and not simply the
#       *coordinate* of the quadrature node. This is so that you can access
#       additional information, such as the `normal` vector, at the quadrature node.

# We can now solve the integral equation using e.g. the backslash operator:

σ = L \ rhs

# The variable `σ` contains the value of the approximate density at the
# quadrature nodes, which can be used to reconstruct the solution using the
# [`IntegralPotential`](@ref Inti.IntegralPotential):

𝒮 = Inti.IntegralPotential(G,Q)
𝒟 = Inti.IntegralPotential(dGdny,Q)
uₛ = x -> 𝒟[σ](x) - im*k*𝒮[σ](x)

# where `uₛ` is an anonymous/lambda function representing the approximate
# scattered field.

# To assess the accuracy of the solution, we can compare it to the exact
# solution (obtained by separation of variables in polar coordinates):

using SpecialFunctions # for bessel functions

function circle_helmholtz_soundsoft(pt;radius=1,k,θin)
    x = pt[1]
    y = pt[2]
    r = sqrt(x^2+y^2)
    θ = atan(y,x)
    u = 0.0
    r < radius && return u
    c(n) = -exp(im*n*(π/2-θin))*besselj(n,k*radius)/besselh(n,k*radius)
    u    = c(0)*besselh(0,k*r)
    n = 1;
    while (abs(c(n)) > 1e-12)
        u += c(n)*besselh(n,k*r)*exp(im*n*θ) + c(-n)*besselh(-n,k*r)*exp(-im*n*θ)
        n += 1
    end
    return u
end

# Here is the maximum error on some points located on a circle of radius `2`:

uₑ = x -> circle_helmholtz_soundsoft(x,k=k,radius=1,θin=0) # exact solution
er = maximum(0:0.01:2π) do θ
    R = 2
    x = (R*cos(θ),R*sin(θ))
    abs(uₛ(x) - uₑ(x))
end
@info "error without correction = $er"

# We see that the error is quite large, and somewhat not satisfactory! The main
# issue here is that the quadrature `Q` was designed to integrate smooth
# functions over `Γ`, and it is not expected to be accurate enough in the
# presence of singular and nearly-singular kernels (ubiquitous in BIEs). To
# remedy this, we need to compute a *correction* for the integral operators
# $\mathrm{S}$ and $\mathrm{D}$. In this example, we will use the [general
# purpose density interpolation
# method](https://www.sciencedirect.com/science/article/pii/S0045782521000396?casa_token=fG6da2Kb12EAAAAA:_BJ1-uC5gIeEBA08K_ip2nyDVwz9UF3TTBAg--a-vLKfGWljhIMrcJDWUudou3fr19VPEx9ftw),
# implemented in the [`bdim_correction`](@ref Inti.bdim_correction) function,
# for computing a sparse correction to the integral operators above (see the
# docstring of the function for more details):

δS, δD = Inti.bdim_correction(pde, Q, Q, S_mat, D_mat)
nothing #hide

# This will construct a sparse matrix `δS` and `δD` that can be used to correct
# `S_mat` and `D_mat`. Assembling a corrected version of the combined-field BIE
# is done through:

L = I/2 + (D_mat + δD) - im*k*(S_mat + δS)
nothing #hide

# We can now solve the linear system again, and assemble a corrected version of
# the scattered field:

σ = L \ rhs
uₛ = x -> 𝒟[σ](x) - im*k*𝒮[σ](x)

# Let us check the error again:
er = maximum(0:0.01:2π) do θ
    R = 2
    x = (R*cos(θ),R*sin(θ))
    abs(uₛ(x) - uₑ(x))
end
@assert er < 1e-5 #hide
@info "error with correction = $er"

# As we can see, the error is much smaller! This example was intended to
# illustrate the importance properly handling the singularities: correcting for
# the nearly-singular integrals is essential, and should always be done in
# practice.

# To visualize the solution in this simple (2d) example, we could simply use
# `Makie`:

using CairoMakie
xx = yy = range(-4,stop=4,length=200)
vals = map(pt-> norm(pt) > 1 ? real(uₛ(pt) + uᵢ(pt)) : NaN, Iterators.product(xx,yy))
fig,ax,hm = heatmap(xx,yy,vals;
                    colormap=:inferno, interpolate=true,
                    axis=(aspect=DataAspect(),xgridvisible = false, ygridvisible = false),
                    )
lines!(ax,[cos(θ) for θ in 0:0.01:2π], [sin(θ) for θ in 0:0.01:2π], color=:black, linewidth=4)
Colorbar(fig[1,2], hm)
fig

# More complex problems, however, may require a mesh-based visualization, where
# we would first need to create a mesh for the places where we want to visualize
# the solution. In the 3D example that follows, we will use the `Gmsh` API to
# create a *view* (in the sense of *Gmsh*) of the solution on a punctured plane.
# For now, we create a Gmsh *view*; it can be visualized by running `gmsh.fltk.run()`
# or by saving with `gmsh.view.write(vg, "2d_scatter.pos")`.

data = Vector{Float64}(undef, length(xx)*length(yy)*4)
data[1:4:end] = map(pt-> pt[1], Iterators.product(xx,yy))
data[2:4:end] = map(pt-> pt[2], Iterators.product(xx,yy))
data[3:4:end] .= 0
data[4:4:end] = vals
gmsh.initialize()
vg = gmsh.view.add("grid")
gmsh.view.addListData(vg, "SP", length(xx)*length(yy), vec(vals))
gmsh.view.write(vg, "2d_scatter.pos")
gmsh.finalize()

# ## [Three-dimensional scattering](@id helmholtz-scattering-3d)
#
# We now consider the same problem in 3D, being a bit more terse in the
# explanation since many of the steps are similar. The main difficulty we will
# try to address here is related to the computational complexity associated to
# three-dimensional problems, where one needs to be careful not to assemble
# and/or factor huge dense matrices! Instead of representing `Sop` and `Dop` as
# dense matrices, and solving the system using `\` (which dispatches to to an
# `LU` factorization in our case) we will:
# - use hierarchical matrices to compress the integral operators
# - rely on an iterative solver to solve the linear system

# The following function will create a mesh of a sphere using the `Gmsh` API:

function gmsh_sphere(;name, meshsize, order=1, radius=1, center = (0,0,0))
    try
        gmsh.initialize()
        gmsh.model.add("sphere-mesh")
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addSphere(center[1], center[2], center[3], radius)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        gmsh.write(name)
    finally
        gmsh.finalize()
    end
end

# As before, lets write a file with our mesh, and import it into `Inti.jl`, and
# create a surface quadrature:

name = joinpath(@__DIR__, "sphere.msh")
gmsh_sphere(;meshsize=0.1, order = 2, name)
Ω,msh = Inti.gmsh_read_msh(name; dim=3)
Γ = Inti.boundary(Ω)
Q  = Inti.Quadrature(msh, Γ; qorder = 4)

# !!! tip "Writing/reading a mesh from disk"
#       Writing and reading a mesh to/from disk can be time consuming. You can
#       avoid doing so by using [`gmsh_import_mesh!`](@ref Inti.gmsh_import_mesh)
#       and [`gmsh_import_domain`](@ref Inti.gmsh_import_domain) functions on an
#       active `gmsh` model without writing it to disk.

# We can now assemble the integral operators as before, being careful to use
# `dim=3` when defining the Helmholtz PDE:

pde = Inti.Helmholtz(;k, dim=3)
G  = Inti.SingleLayerKernel(pde)
dGdny = Inti.DoubleLayerKernel(pde)

Sop = Inti.IntegralOperator(G,Q,Q)
Dop = Inti.IntegralOperator(dGdny,Q,Q)

# Here is how much memory it would take to store the dense representation of
# these matrices:

mem = 2*length(Sop)*16/1e9 # 16 bytes per complex number, 1e9 bytes per GB, two matrices
println("memory required to store Sop and Dop: $(mem) GB")

# Even for this simple example, the dense representation of the integral
# operators as matrix is already quite expensive. We will thus use hierarchical
# matrix representation to compress things a bit:

import HMatrices
S_hmat = HMatrices.assemble_hmatrix(Sop;atol=1e-6)

#-

D_hmat = HMatrices.assemble_hmatrix(Dop;atol=1e-6)

# !!! note "Compression methods"
#       It is worth mentioning that hierchical matrices are not the only way to
#       compress such integral operators, and may in fact not even be the best
#       for the problem at hand. For example, one could use a fast multipole
#       method (FMM), which has a much lighter memory footprint, and is also
#       faster to assemble. The main advantage of hierarchical matrices is that
#       they are purely algebraic, allowing for the use of *direct solver*.
#       Hierarchical matrices also tend to give a faster matrix-vector product
#       after the (offline) assembly stage.

# As in the 2D case, we need to correct `S_hmat` and `D_hmat` to account for the
# singular and nearly-singular integrals. We will again use the density
# interpolation method here:

δS, δD = Inti.bdim_correction(pde, Q, Q, S_hmat, D_hmat)
nothing #hide

# We will use the generalized minimal residual (GMRES) iterative solver, for the
# linear system. This requires us to define a linear operator `L`, approximating
# the combined-field operator, that supports the matrix-vector product. In what
# follows we use `LinearMaps` to *lazily* assemble `L`:

using LinearMaps
L = I/2 + (LinearMap(D_hmat) + LinearMap(δD)) - im*k*(LinearMap(S_hmat) + LinearMap(δS))

# We can now solve the linear system using GMRES solver (implemented in
# `IterativeSolvers`):

using IterativeSolvers
rhs = map(Q) do q
    x = q.coords
    -uᵢ(x)
end
σ, hist = gmres(L, rhs; log=true, abstol=1e-6)
@show hist

# As before, let us represent the solution using `IntegralPotential`s:

𝒮 = Inti.IntegralPotential(G,Q)
𝒟 = Inti.IntegralPotential(dGdny,Q)
uₛ = x -> 𝒟[σ](x) - im*k*𝒮[σ](x)

# To check the result, we compare against the exact solution obtained through a
# series:
using GSL
sphbesselj(l,r) = sqrt(π/(2r)) * besselj(l+1/2,r)
sphbesselh(l,r) = sqrt(π/(2r)) * besselh(l+1/2,r)
sphharmonic(l,m,θ,ϕ) = GSL.sf_legendre_sphPlm(l,abs(m),cos(θ))*exp(im*m*ϕ)
function sphere_helmholtz_soundsoft(xobs;radius=1,k=1,θin=0,ϕin=0)
    x = xobs[1]
    y = xobs[2]
    z = xobs[3]
    r = sqrt(x^2+y^2+z^2)
    θ = acos(z/r)
    ϕ = atan(y,x)
    u = 0.0
    r < radius && return u
    c(l,m) = -4π*im^l*sphharmonic(l,-m,θin,ϕin)*sphbesselj(l,k*radius)/sphbesselh(l,k*radius)
    l = 0
    for l=0:60
        for m=-l:l
            u += c(l,m)*sphbesselh(l,k*r)*sphharmonic(l,m,θ,ϕ)
        end
        l += 1
    end
    return u
end

# We will compute the error on some point on the sphere of radius `2`:

uₑ = (x) -> sphere_helmholtz_soundsoft(x;radius=1,k=k,θin=π/2,ϕin=0)
er = maximum(1:100) do _
    x̂ = rand(Inti.Point3D) |> normalize # an SVector of unit norm
    x = 2*x̂
    abs(uₛ(x) - uₑ(x))
end
@assert er < 1e-5 #hide
@info "error with correction = $er"

# We see that the approximation is quite accurate. We can now export the
# solution to *Gmsh* for visualization. To do so, we will first create a Gmsh
# view of our solution data evaluated on a meshed planar slice of ℝ³.

meshsize = 0.05
order = 2
radius = 1
center = (0, 0, 0)
gmsh.initialize()
gmsh.model.add("slice-sphere-mesh")
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)

# Set up a rectangular outer boundary with a circular hole as the rectangle
# intersects the sphere; use gmsh CSG to construct the set difference, and mesh
px = -2.0; py = -2.0; pz = 0.0
dx = 4.0; dy = 4.0
p1 = gmsh.model.occ.addPoint(px, py, pz)
p2 = gmsh.model.occ.addPoint(px + dx, py, pz) 
p3 = gmsh.model.occ.addPoint(px + dx, py + dy, pz)
p4 = gmsh.model.occ.addPoint(px, py + dy, pz)
l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p1)
curve_rtag = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
sphere_ctag = gmsh.model.occ.addSphere(center[1], center[2], center[3], radius)
ptag = gmsh.model.occ.addPlaneSurface([curve_rtag])
outDimTags, outDimTagsMap = gmsh.model.occ.cut([(2, ptag)], [(3, sphere_ctag)])
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(order)
Ωₑ =  Inti.gmsh_import_domain(;dim=2)
eval_msh = Inti.gmsh_import_mesh(Ωₑ;dim=3)
name = joinpath(@__DIR__, "slice-sphere.msh")
gmsh.write(name)

# Evaluate the solution on the nodes (`order = 2` corresponds to element
# type `9` or a 6-node triangle) of the 2D mesh:
ntags, xyz, _ = gmsh.model.mesh.getNodes()
etags, vtags = gmsh.model.mesh.getElementsByType(9)
nodes = eachcol(reshape(xyz, 3, :))[vtags]
u_eval_msh = map(nodes) do q
    x = Inti.Point3D(q)
    uₛ(x) + uᵢ(x) |> real
end
nothing #hide

# Add a gmsh view of the solution and save it:
s1 = gmsh.view.add("Solution Slice")
gmsh.view.addHomogeneousModelData(s1, 0, "slice-sphere-mesh", "ElementNodeData", etags, u_eval_msh)
gmsh.view.write(s1, "3d_nodedata.pos")
# To visualize in gmsh's FLTK GUI, execute here `gmsh.fltk.run()`. For now, just finish up.
gmsh.finalize()