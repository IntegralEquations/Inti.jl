using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

#nb ## Environment setup
#nb const DEPENDENCIES = ["GLMakie", "Gmsh", "HMatrices", "IterativeSolvers","LinearAlgebra", "LinearMaps", "SpecialFunctions", "GSL", "Meshes"];
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
# performance issues (e.g. compressing the integral operators). Finally, we
# present a [three-dimensional example](@ref helmholtz-scattering-3d), where we
# will use [`HMatrices.jl`](https://github.com/IntegralEquatins/HMatrices.jl) to
# compress the underlying integral operators.

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
#       The main reason for focusing on such a simple example is twofold. First,
#       it alleviates the complexities associated with the mesh generation. Second,
#       since exact solutions are known for this problem (in the form of a series),
#       it is easy to assess the accuracy of the solution obtained. In practice, you
#       can use the same techniques to solve the problem on more complex geometries
#       by providing a `.msh` file containing the mesh.

# Using the theory of boundary integral equations, we can express ``u_s`` as
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
# spatial dimensions. Let's load `Inti.jl` as well as the required dependencies
using Inti
using LinearAlgebra
using StaticArrays
using Gmsh
using Meshes
using GLMakie
using SpecialFunctions
using GSL
using IterativeSolvers
using LinearMaps

# and setup some of the (global) problem parameters:

k      = 4π
λ      = 2π / k
qorder = 4 # quadrature order
gorder = 2 # order of geometrical approximation
nothing #hide

# ## [Two-dimensional scattering](@id helmholtz-scattering-2d)

# We will use [Gmsh
# API](https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-application-programming-interface)
# for creating `.msh` file containing the desired geometry and mesh. Here is a
# function to mesh the circle:

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
meshsize = λ / 5
gmsh_circle(; meshsize, order = gorder, name)
nothing #hide

# We can now import the file and parse the mesh and domain information into
# `Inti.jl` using the [`import_mesh`](@ref Inti.import_mesh) function:

Inti.clear_entities!() # empty the entity cache
msh = Inti.import_mesh(name; dim = 2)

# The code above will import the mesh with all of its geometrical entities. The
# `dim=2` projects all points to two dimensions by ignoring the third
# component. To extract the domain ``\Omega`` we need to filter the entities in
# the mesh; here we will simply filter them based on the `geometric_dimension`:

Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))

# To solve our boundary integral equation usign a Nyström method, we actually
# need a quadrature of our curve/surface (and possibly the normal vectors at the
# quadrature nodes). Once a mesh is available, creating a quadrature object can
# be done via the [`Quadrature`](@ref Inti.Quadrature) constructor, which
# requires passing a mesh of the domain that one wishes to generate a quadrature
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
# integrating the function `x->1` over `Q` gives an approximation to the
# perimeter:

@assert abs(Inti.integrate(x -> 1, Q) - 2π) < 1e-5 #hide
abs(Inti.integrate(x -> 1, Q) - 2π)

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
# we can use [`single_double_layer_potential`](@ref
# Inti.single_double_layer_potential) to obtain the single- and double-layer
# potentials, and then combine them as follows:

𝒮, 𝒟 = Inti.single_double_layer_potential(; pde, source = Q)
uₛ   = x -> 𝒟[σ](x) - im * k * 𝒮[σ](x)
nothing #hide

# The variable `uₛ` is an anonymous/lambda function representing the approximate
# scattered field.

# To assess the accuracy of the solution, we can compare it to the exact
# solution (obtained by separation of variables in polar coordinates):

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

# As we can see, the error is quite small! Let's use `Makie` to visualize the solution in this
# simple (2d) example:

xx = yy = range(-4; stop = 4, length = 200)
vals =
    map(pt -> Inti.isinside(pt, Q) ? NaN : real(uₛ(pt) + uᵢ(pt)), Iterators.product(xx, yy))
fig, ax, hm = heatmap(
    xx,
    yy,
    vals;
    colormap = :inferno,
    interpolate = true,
    axis = (aspect = DataAspect(), xgridvisible = false, ygridvisible = false),
)
viz!(Γ_msh; color = :white, segmentsize = 5)
Colorbar(fig[1, 2], hm)
fig

# While here we simply used a heatmap to visualize the solution, more complex
# problems may require a mesh-based visualization, where we would first create a
# mesh for the places where we want to visualize the solution.

# Before moving on to the 3D example let us simply mention that, besides the
# fact that an analytic solution was available for comparisson, there was
# nothing special about the unit disk (or the use of GMSH). We could have, for
# instance, replaced the disk by shapes created parametrically:

## vertices of an equilateral triangle centered at the origin with a vertex at (0,1)
a, b, c = SVector(0, 1), SVector(sqrt(3) / 2, -1 / 2), SVector(-sqrt(3) / 2, -1 / 2)
circle_f(center, radius) = s -> center + radius * SVector(cospi(2 * s[1]), sinpi(2 * s[1]))
disk1 = Inti.parametric_curve(circle_f(a, 1 / 2), 0, 1)
disk2 = Inti.parametric_curve(circle_f(b, 1 / 2), 0, 1)
disk3 = Inti.parametric_curve(circle_f(c, 1 / 2), 0, 1)
Γ = disk1 ∪ disk2 ∪ disk3
msh = Inti.meshgen(Γ; meshsize)
Γ_msh = view(msh, Γ)
Q = Inti.Quadrature(Γ_msh; qorder)
S, D = Inti.single_double_layer(;
    pde,
    target = Q,
    source = Q,
    compression = (method = :none,),
    correction = (method = :dim,),
)
L = I / 2 + D - im * k * S
rhs = map(q -> -uᵢ(q.coords), Q)
σ = L \ rhs
𝒮, 𝒟 = Inti.single_double_layer_potential(; pde, source = Q)
uₛ = x -> 𝒟[σ](x) - im * k * 𝒮[σ](x)
vals =
    map(pt -> Inti.isinside(pt, Q) ? NaN : real(uₛ(pt) + uᵢ(pt)), Iterators.product(xx, yy))
colorrange = (-2, 2)
fig, ax, hm = heatmap(
    xx,
    yy,
    vals;
    colormap = :inferno,
    colorrange,
    interpolate = true,
    axis = (aspect = DataAspect(), xgridvisible = false, ygridvisible = false),
)
viz!(Γ_msh; color = :black, segmentsize = 4)
Colorbar(fig[1, 2], hm)
fig

#=

!!! note "Near-field evaluation"
    In the example above we employed a naive evaluation of the integral
    potentials, and therefore the computed solution is expected to become
    innacurate near the obstacles. See the [layer potential tutorial](@ref
    "Layer potentials") for more information on how to correct for this.

=#

# ## [Three-dimensional scattering](@id helmholtz-scattering-3d)
#
# We now consider the same problem in 3D. Unlike the 2D case, assembling dense
# matrix representations of the integral operators quickly becomes unfeasiable
# as the problem size increases. `Inti` adds support for compressing the
# underlying linear operators by wrapping external libraries. In this example,
# we will rely on [`HMatrices.jl`](https://github.com/WaveProp/HMatrices.jl) to
# handle the compression.

# The visualization is also more involved, and we will use the `Gmsh` API to
# create a not only a mesh of the scatterer, but also of a punctured plane where
# we will visualize the solution. Here is the function that setups up the mesh:

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
msh = Inti.import_mesh(name; dim = 3)

# !!! tip
#       If you pass `visualize=true` to `gmsh_sphere`, it will open a window
#       with the current model. This is done by calling `gmsh.fltk.run()`. Note
#       that the main julia thread will be blocked until the window is closed.

# Since we created physical groups in `Gmsh`, we can use them to extract the
# relevant domains `Ω` and `Σ`:

Ω = Inti.Domain(e -> "omega" ∈ Inti.labels(e), Inti.entities(msh))
Σ = Inti.Domain(e -> "sigma" ∈ Inti.labels(e), Inti.entities(msh))
Γ = Inti.boundary(Ω)
nothing #hide

# We can now create a quadrature as before

Γ_msh = view(msh, Γ)
Q = Inti.Quadrature(Γ_msh; qorder)
nothing #hide

# !!! tip "Writing/reading a mesh from disk"
#       Writing and reading a mesh to/from disk can be time consuming. You can
#       avoid doing so by using [`import_mesh`](@ref Inti.import_mesh) without a
#       file name to import the mesh from the current `gmsh` session without the
#       need to write it to disk.

# Next we assemble the integral operators, indicating that we wish to compress
# them using hierarchical matrices:
using HMatrices
pde = Inti.Helmholtz(; k, dim = 3)
S, D = Inti.single_double_layer(;
    pde,
    target = Q,
    source = Q,
    compression = (method = :hmatrix, tol = 1e-4),
    correction = (method = :dim,),
)
nothing #hide

# Here is how much memory it would take to store the dense representation of
# these matrices:

mem = 2 * length(S) * 16 / 1e9 # 16 bytes per complex number, 1e9 bytes per GB, two matrices
println("memory required to store S and D: $(mem) GB")

# Even for this simple example, the dense representation of the integral
# operators as matrix is already quite expensive!

#=

!!! note "Compression methods"
    It is worth mentioning that hierchical matrices are not the only way to
    compress such integral operators, and may in fact not even be the best for
    the problem at hand. For example, one could use a fast multipole method
    (FMM), which has a much lighter memory footprint. See the the [tutorial on
    compression methods](@ref "Compression methods") for more information.

=#

# We will use the generalized minimal residual (GMRES) iterative solver, for the
# linear system. This requires us to define a linear operator `L`, approximating
# the combined-field operator, that supports the matrix-vector product. While it
# is possible to add two `HMatrix` objects to obtain a new `HMatrix`, this is
# somewhat more involved due to the addition of low-rank blocks (which requires
# a recompression). To keep things simple, we will use `LinearMaps` to lazily
# compose the operators:
L = I / 2 + LinearMap(D) - im * k * LinearMap(S)
nothing #hide

# We can now solve the linear system using GMRES solver:

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
# expensive, we will use again use a method to accelerate the evaluation:

Σ_msh = view(msh, Σ)
target = Inti.nodes(Σ_msh)

S, D = Inti.single_double_layer(;
    pde,
    target,
    source = Q,
    compression = (method = :hmatrix, tol = 1e-4),
    ## correction for the nearfield (for visual purposes, set to `:none` to disable)
    correction = (method = :dim, maxdist = meshsize, target_location = :outside),
)

ui_eval_msh = uᵢ.(target)
us_eval_msh = D * σ - im * k * S * σ
u_eval_msh = ui_eval_msh + us_eval_msh
nothing #hide

# Finalize, we use [`Meshes.viz`](@extref) to visualize the scattered field:

nv = length(Inti.nodes(Γ_msh))
colorrange = extrema(real(u_eval_msh))
colormap = :inferno
fig = Figure(; size = (800, 500))
ax = Axis3(fig[1, 1]; aspect = :data)
viz!(Γ_msh; colorrange, colormap, color = zeros(nv), interpolate = true)
viz!(Σ_msh; colorrange, colormap, color = real(u_eval_msh))
cb = Colorbar(fig[1, 2]; label = "real(u)", colormap, colorrange)
fig # hide
