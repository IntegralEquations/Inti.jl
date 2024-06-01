using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

#nb ## Environment setup
#nb const DEPENDENCIES = ["GLMakie", "Gmsh", "HMatrices", "IterativeSolvers","LinearAlgebra", "LinearMaps", "SpecialFunctions", "GSL", "FMM3D", "FMM2D", "Meshes"];
#nb ## __NOTEBOOK_SETUP__

# # [Helmholtz scattering](@id helmholtz_scattering)

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](helmholtz_scattering.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/helmholtz_scattering.ipynb)

# !!! note "Important points covered in this example"
#       - Creating a geometry using the *Gmsh* API
#       - Assembling integral operators and integral potentials
#       - Setting up a sound-soft problem in both 2 and 3 spatial dimensions
#       - Using *GMRES* to solve the linear system
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
# spatial dimensions. Let's load `Inti.jl` and setup some of the (global)
# problem parameters:

using Inti

k        = 4π
λ        = 2π / k
meshsize = λ / 5
qorder   = 4 # quadrature order
gorder   = 2 # order of geometrical approximation
nothing #hide

# ## [Two-dimensional scattering](@id helmholtz-scattering-2d)

# We use [Gmsh
# API](https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-application-programming-interface)
# for creating `.msh` file containing the desired geometry and mesh. Here is a
# function to mesh the circle:

using Gmsh # this will trigger the loading of Inti's Gmsh extension

function gmsh_circle(; name, meshsize, order = 1, radius = 1, center = (0, 0))
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
nothing #hide

# Let us now use `gmsh_circle` to create a `circle.msh` file. As customary in
# wave-scattering problems, we will choose a mesh size that is proportional to
# wavelength:
name = joinpath(@__DIR__, "circle.msh")
gmsh_circle(; meshsize, order = gorder, name)
nothing #hide

# We can now import the file and parse the mesh and domain information into
# `Inti.jl` using the [`import_mesh`](@ref Inti.import_mesh) function:

Inti.clear_entities!() # empty the entity cache
msh = Inti.import_mesh(name; dim = 2)
@show msh

# The code above will parse all entities in the `name` of dimension `2` as a
# [`Domain`](@ref Inti.Domain), and also import the underlying mesh (that, in
# this case, is projected into two dimensions by ignoring the third component).
# Note that in the example above, `Ω` is a two-dimensional domain containing a
# single `GmshEntity` which represents the disk. To extract the boundary $\Gamma
# = \partial \Omega$, we can use the [`boundary`](@ref Inti.boundary) function:

Γ = Inti.boundary(Ω)

# To solve our boundary integral equation usign a Nyström method, we actually
# need a quadrature of our curve/surface (and possibly the normal vectors at the
# quadrature nodes). Once a mesh is available, creating a quadrature object can
# be done via the [`Quadrature`](@ref Inti.Quadrature) constructor, which
# requires passing a mesh the domain that one wishes to generate a quadrature
# for:

Γ = Inti.boundary(Ω)
Γ_msh = view(msh, Γ)
Q = Inti.Quadrature(Γ_msh; qorder)
nothing #hide

# !!! tip "Views of a mesh"
#       In `Inti.jl`, you can use domain to create a *view* of a mesh containing *only
#       the elements in the domain*. For example `view(msh,Γ)` will return an
#       `SubMesh` type that you can use to iterate over the elements in the boundary
#       of the disk without actually creating a new mesh. You can use `msh[Γ]`,
#       or `collect(view(msh,Γ))` to create a new mesh containing *only* the
#       elements and nodes in `Γ`.

# The object `Q` now contains a quadrature (of order `4`) that can be used to
# solve a boundary integral equation on `Γ`. As a sanity check, let's make sure
# integrating the function `x->1` over `Q` gives an approximation to the perimeter:

@assert abs(Inti.integrate(x -> 1, Q) - 2π) < 1e-5 #hide
Inti.integrate(x -> 1, Q) - 2π

# With the [`Quadrature`](@ref Inti.Quadrature) constructed, we now can define
# discrete approximation to the integral operators ``\mathrm{S}`` and
# ``\mathrm{D}`` as follows:

pde = Inti.Helmholtz(; k, dim = 2)
S, D = Inti.single_double_layer(;
    pde,
    target = Q,
    source = Q,
    compression = (method = :none,),
    correction = (method = :dim,),
)
nothing #hide

# There are two well-known difficulties related to the discretization of
# the boundary integral operators $S$ and $D$:
# - The kernel of the integral operator is not smooth, and thus specialized
#   quadrature rules are required to accurately approximate the matrix entries for
#   which the target and source point lie *close* (relative to some scale) to each
#   other.
# - The underlying matrix is dense, and thus the storage and computational cost
#   of the operator is prohibitive for large problems unless acceleration
#   techniques such as *Fast Multipole Methods* or *Hierarchical Matrices* are
#   employed.

# `Inti.jl` tries to provide a modular and transparent interface for dealing
# with both of these difficulties, where the general approach for solving a BIE
# will be to first construct a (possible compressed) naive representation of the
# integral operator where singular and nearly-singular integrals are ignored,
# followed by a the creation of a (sparse) correction intended to account for
# such singular interactions. See [`single_double_layer`](@ref
# Inti.single_double_layer) for more details on the various options available.

# We can now combine `S` and `D` to form the combined-field operator:
using LinearAlgebra
L = I / 2 + D - im * k * S
nothing #hide

# where `I` is the identity matrix. Assuming an incident field along the $x_1$
# direction of the form $u_i =e^{ikx_1}$, the right-hand side of the equation
# can be construted using:

uᵢ = x -> exp(im * k * x[1]) # plane-wave incident field
rhs = map(Q) do q
    x = q.coords
    return -uᵢ(x)
end
nothing #hide

# !!! note "Iterating over a quadrature"
#       In computing `rhs` above, we used `map` to evaluate the incident field at
#       all quadrature nodes. When iterating over `Q`, the iterator returns a
#       [`QuadratureNode`](@ref Inti.QuadratureNode), and not simply the
#       *coordinate* of the quadrature node. This is so that you can access
#       additional information, such as the `normal` vector, at the quadrature node.

# We can now solve the integral equation using e.g. the backslash operator:

σ = L \ rhs
nothing #hide

# The variable `σ` contains the value of the approximate density at the
# quadrature nodes. To reconstruct a continuous approximation to the solution,
# we can use [`single_double_layer_potential`](@ref Inti.single_double_layer_potential) to obtain the single- and
# double-layer potentials, and then combine them as follows:

𝒮, 𝒟 = Inti.single_double_layer_potential(; pde, source = Q)
uₛ   = x -> 𝒟[σ](x) - im * k * 𝒮[σ](x)
nothing #hide

# The variable `uₛ` is an anonymous/lambda function representing the approximate
# scattered field.

# To assess the accuracy of the solution, we can compare it to the exact
# solution (obtained by separation of variables in polar coordinates):

using SpecialFunctions # for bessel functions

function circle_helmholtz_soundsoft(pt; radius = 1, k, θin)
    x = pt[1]
    y = pt[2]
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    u = 0.0
    r < radius && return u
    c(n) = -exp(im * n * (π / 2 - θin)) * besselj(n, k * radius) / besselh(n, k * radius)
    u    = c(0) * besselh(0, k * r)
    n    = 1
    while (abs(c(n)) > 1e-12)
        u +=
            c(n) * besselh(n, k * r) * exp(im * n * θ) +
            c(-n) * besselh(-n, k * r) * exp(-im * n * θ)
        n += 1
    end
    return u
end
nothing #hide

# Here is the maximum error on some points located on a circle of radius `2`:

uₑ = x -> circle_helmholtz_soundsoft(x; k, radius = 1, θin = 0) # exact solution
er = maximum(0:0.01:2π) do θ
    R = 2
    x = (R * cos(θ), R * sin(θ))
    return abs(uₛ(x) - uₑ(x))
end
@assert er < 1e-3 #hide
@info "maximum error = $er"

# As we can see, the error is quite small! To visualize the solution in this
# simple (2d) example, we could simply use `Makie`:

using GLMakie
xx = yy = range(-4; stop = 4, length = 200)
vals = map(pt -> norm(pt) > 1 ? real(uₛ(pt) + uᵢ(pt)) : NaN, Iterators.product(xx, yy))
fig, ax, hm = heatmap(
    xx,
    yy,
    vals;
    colormap = :inferno,
    interpolate = true,
    axis = (aspect = DataAspect(), xgridvisible = false, ygridvisible = false),
)
lines!(
    ax,
    [cos(θ) for θ in 0:0.01:2π],
    [sin(θ) for θ in 0:0.01:2π];
    color = :black,
    linewidth = 4,
)
Colorbar(fig[1, 2], hm)
fig

# More complex problems, however, may require a mesh-based visualization, where
# we would first need to create a mesh for the places where we want to visualize
# the solution. In the 3D example that follows, we will use the `Gmsh` API to
# create a a mesh of a punctured plane where we will visualize the solution.

# Before moving on to the 3D example let us simply mention that, besides the
# fact that an analytic solution was available for comparisson, there was
# nothing special about the unit disk in the example above. We could have, for
# instance, replaced the disk by a kite-like shape:

f = (s) -> (cospi(2 * s[1]) + 0.65 * cospi(4 * s[1]) - 0.65, 1.5 * sinpi(2 * s[1]))
Inti.clear_entities!() # empty the entity cacheg
gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
## parametrization of a kite-like shape
tag = Inti.gmsh_curve(f, 0, 1; npts = 100)
## create a surface from the curve
tl = gmsh.model.occ.addCurveLoop([tag])
ta = gmsh.model.occ.addPlaneSurface([tl])
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(gorder)
msh = Inti.import_mesh_from_gmsh_model(; dim = 2)
gmsh.finalize()

# ## [Three-dimensional scattering](@id helmholtz-scattering-3d)
#
# We now consider the same problem in 3D. Unlike the 2D case, assembling dense
# matrix representations of the integral operators quickly becomes unfeasiable
# as the problem size increases. `Inti` adds support for compressing the
# underlying linear operators by wrapping external libraries. In this example,
# we will rely on [`HMatrices.jl`](https://github.com/WaveProp/HMatrices.jl) to
# handle the compression.

# The visualization is also more involved, and we will
# use instead the `Gmsh` API to create a view of the solution on a punctured
# plane. Let us begin by creating our domain containing both the sphere and the
# puctured plane where we will visualize the solution:

function gmsh_sphere(; meshsize, order = gorder, radius = 1, visualize = false, name)
    gmsh.initialize()
    gmsh.model.add("sphere-scattering")
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    sphere_tag = gmsh.model.occ.addSphere(0, 0, 0, radius)
    xl, yl, zl = -2 * radius, -2 * radius, 0
    Δx, Δy = 4 * radius, 4 * radius
    rectangle_tag = gmsh.model.occ.addRectangle(xl, yl, zl, Δx, Δy)
    outDimTags, _ =
        gmsh.model.occ.cut([(2, rectangle_tag)], [(3, sphere_tag)], -1, true, false)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [sphere_tag], -1, "omega")
    gmsh.model.addPhysicalGroup(2, [dt[2] for dt in outDimTags], -1, "sigma")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    visualize && gmsh.fltk.run()
    gmsh.option.setNumber("Mesh.SaveAll", 1) # otherwise only the physical groups are saved
    gmsh.write(name)
    return gmsh.finalize()
end
nothing #hide

# As before, lets write a file with our mesh, and import it into `Inti.jl`:

name = joinpath(@__DIR__, "sphere.msh")
gmsh_sphere(; meshsize, order = gorder, name, visualize = false)
Inti.clear_entities!()
Ω, msh = Inti.import_mesh(name; dim = 3)
Γ = Inti.boundary(Ω)
nothing #hide

# Note that for this example we relied instead on the labels to the entities in
# order to extract the relevant domains `Ω` and `Σ`. We can now create a
# quadrature as before

Γ_msh = view(msh, Γ)
Q = Inti.Quadrature(Γ_msh; qorder)
nothing #hide

# !!! tip
#       If you pass `visualize=true` to `gmsh_sphere`, it will open a window
#       with the current mode. This is done by calling `gmsh.fltk.run()`. Note
#       that the main julia thread will be blocked until the window is closed.

# !!! tip "Writing/reading a mesh from disk"
#       Writing and reading a mesh to/from disk can be time consuming. You can
#       avoid doing so by using [`import_mesh`](@ref Inti.import_mesh)
#       and [`import_mesh`](@ref Inti.import_mesh) functions on an
#       active `gmsh` model without writing it to disk.

# We can now assemble the integral operators, indicating that we
# wish to compress them using hierarchical matrices:
using HMatrices
pde = Inti.Helmholtz(; k, dim = 3)
S, D = Inti.single_double_layer(;
    pde,
    target = Q,
    source = Q,
    compression = (method = :hmatrix, tol = 1e-6),
    correction = (method = :dim,),
)
nothing #hide

# Here is how much memory it would take to store the dense representation of
# these matrices:

mem = 2 * length(S) * 16 / 1e9 # 16 bytes per complex number, 1e9 bytes per GB, two matrices
println("memory required to store S and D: $(mem) GB")

# Even for this simple example, the dense representation of the integral
# operators as matrix is already quite expensive!

# !!! note "Compression methods"
#       It is worth mentioning that hierchical matrices are not the only way to
#       compress such integral operators, and may in fact not even be the best
#       for the problem at hand. For example, one could use a fast multipole
#       method (FMM), which has a much lighter memory footprint, and is also
#       faster to assemble. The main advantage of hierarchical matrices is that
#       they are purely algebraic, allowing for the use of *direct solver*.
#       Hierarchical matrices also tend to give a faster matrix-vector product
#       after the (offline) assembly stage.

# We will use the generalized minimal residual (GMRES) iterative solver, for the
# linear system. This requires us to define a linear operator `L`, approximating
# the combined-field operator, that supports the matrix-vector product. In what
# follows we use `LinearMaps` to *lazily* assemble `L`:

using LinearMaps
L = I / 2 + LinearMap(D) - im * k * LinearMap(S)
nothing #hide

# Note that wrapping `S` and `D` in `LinearMap` allows for combining them in a
# *lazy* fashion. Alternatively, you can use e.g. `axpy!` to add two
# hierarchical matrices.

# We can now solve the linear system using GMRES solver:

using IterativeSolvers
rhs = map(Q) do q
    x = q.coords
    return -uᵢ(x)
end
σ, hist =
    gmres(L, rhs; log = true, abstol = 1e-6, verbose = false, restart = 100, maxiter = 100)
@show hist

# As before, let us represent the solution using `IntegralPotential`s:

𝒮, 𝒟 = Inti.single_double_layer_potential(; pde, source = Q)
uₛ = x -> 𝒟[σ](x) - im * k * 𝒮[σ](x)
nothing #hide

# To check the result, we compare against the exact solution obtained through a
# series:
using GSL
sphbesselj(l, r) = sqrt(π / (2r)) * besselj(l + 1 / 2, r)
sphbesselh(l, r) = sqrt(π / (2r)) * besselh(l + 1 / 2, r)
sphharmonic(l, m, θ, ϕ) = GSL.sf_legendre_sphPlm(l, abs(m), cos(θ)) * exp(im * m * ϕ)
function sphere_helmholtz_soundsoft(xobs; radius = 1, k = 1, θin = 0, ϕin = 0)
    x = xobs[1]
    y = xobs[2]
    z = xobs[3]
    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    ϕ = atan(y, x)
    u = 0.0
    r < radius && return u
    function c(l, m)
        return -4π * im^l * sphharmonic(l, -m, θin, ϕin) * sphbesselj(l, k * radius) /
               sphbesselh(l, k * radius)
    end
    l = 0
    for l in 0:60
        for m in -l:l
            u += c(l, m) * sphbesselh(l, k * r) * sphharmonic(l, m, θ, ϕ)
        end
        l += 1
    end
    return u
end
nothing #hide

# We will compute the error on some point on the sphere of radius `2`:

uₑ = (x) -> sphere_helmholtz_soundsoft(x; radius = 1, k = k, θin = π / 2, ϕin = 0)
er = maximum(1:100) do _
    x̂ = rand(Inti.Point3D) |> normalize # an SVector of unit norm
    x = 2 * x̂
    return abs(uₛ(x) - uₑ(x))
end
@assert er < 1e-3 #hide
@info "error with correction = $er"

# We see that, once again, the approximation is quite accurate. Let us now
# visualize the solution on the punctured plane (which we labeled as "sigma").
# Since evaluating the integral representation of the solution at many points is
# expensive, we will use a compression method to accelerate the evaluation as
# well. In the example below, we use the fast-multipole method:

using FMM3D

Σ = Inti.Domain(e -> "sigma" ∈ Inti.labels(e), Inti.entities(msh))
Σ_msh = view(msh, Σ)
target = Inti.nodes(Σ_msh)

S, D = Inti.single_double_layer(;
    pde,
    target,
    source = Q,
    compression = (method = :fmm, tol = 1e-6),
    ## correction for the nearfield (for visual purposes, set to `:none` to disable)
    correction = (method = :dim, maxdist = meshsize, target_location = :outside),
)

ui_eval_msh = uᵢ.(target)
us_eval_msh = D * σ - im * k * S * σ
u_eval_msh = ui_eval_msh + us_eval_msh
nothing #hide

# Finalize, we use [`viz`](@ref Meshes.viz) to visualize the scattered field:

using Meshes
using GLMakie # or your preferred Makie backend

nv = length(Inti.nodes(Γ_msh))
colorrange = extrema(real(u_eval_msh))
colormap = :inferno
fig, ax, pl = viz(Γ_msh; colorrange, colormap, color = zeros(nv))
viz!(Σ_msh; colorrange, colormap, color = real(u_eval_msh))
cb = Colorbar(fig[1, 2]; label = "real(u)", colormap, colorrange)
fig
