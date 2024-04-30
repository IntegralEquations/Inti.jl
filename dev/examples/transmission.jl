# using Markdown                       #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

using Inti

k₁ = 8π
k₂ = 2π
λ₁ = 2π / k₁
λ₂ = 2π / k₂
meshsize   = min(λ₁,λ₂) / 5
qorder     = 4 # quadrature order
gorder     = 2 # order of geometrical approximation
nothing #hide

using Gmsh # this will trigger the loading of Inti's Gmsh extension


# function gmsh_kite(; radius = 1, center = (0,0,0), npts = ceil(Int,radius*10))
#     f = (s) -> center .+ radius .* (cospi(2 * s[1]) + 0.65 * cospi(4 * s[1]) - 0.65,
#             1.5 * sinpi(2 * s[1]))
#     pt_tags = Int32[]
#     for i in 0:npts-1
#         s = i / npts
#         x = center[1] + radius * (cospi(2 * s[1]) + 0.65 * cospi(4 * s[1]) - 0.65)
#         y = center[2] + radius * (1.5 * sinpi(2 * s[1]))
#         z = 0
#         t = gmsh.model.occ.addPoint(x,y,z)
#         push!(pt_tags,t)
#     end
#     # close the curve by adding the first point again
#     push!(pt_tags,pt_tags[1])
#     gmsh.model.occ.addSpline(pt_tags, 1000)
#     gmsh.model.occ.synchronize()
# end


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


name = joinpath(@__DIR__, "circle.msh")
# name = joinpath(@__DIR__, "kite.msh")
gmsh_circle(; meshsize, order = gorder, name)
# gmsh_kite(; meshsize, order = gorder, name)
# gmsh_kite()

Ω, msh = Inti.import_mesh_from_gmsh_file(name; dim = 2)

Γ = Inti.boundary(Ω)
Γ_msh = view(msh,Γ)

Q = Inti.Quadrature(Γ_msh; qorder)

pde₁ = Inti.Helmholtz(; k=k₁, dim = 2)
pde₂ = Inti.Helmholtz(; k=k₂, dim = 2)
using HMatrices
using FMMLIB2D
S₁, D₁ = Inti.single_double_layer(;
    pde=pde₁,
    target = Q,
    source = Q,
    # compression = (method = :none,),
    # compression = (method = :hmatrix,tol=:1e-8),
    compression = (method = :fmm,tol=:1e-8),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

K₁, N₁ = Inti.adj_double_layer_hypersingular(;
    pde=pde₁,
    target = Q,
    source = Q,
    # compression = (method = :none,),
    # compression = (method = :hmatrix,tol=:1e-8),
    compression = (method = :fmm,tol=:1e-8),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

S₂, D₂ = Inti.single_double_layer(;
    pde=pde₂,
    target = Q,
    source = Q,
    # compression = (method = :none,),
    # compression = (method = :hmatrix,tol=:1e-8),
    compression = (method = :fmm,tol=:1e-8),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

K₂, N₂ = Inti.adj_double_layer_hypersingular(;
    pde=pde₂,
    target = Q,
    source = Q,
    # compression = (method = :none,),
    # compression = (method = :hmatrix,tol=:1e-8),
    compression = (method = :fmm,tol=:1e-8),
    correction = (method = :dim, maxdist = 5 * meshsize),
)

using LinearAlgebra
using LinearMaps
L =I+[LinearMap(D₂)-LinearMap(D₁) -LinearMap(S₂)+LinearMap(S₁);LinearMap(N₂)-LinearMap(N₁) -LinearMap(K₂)+LinearMap(K₁)]

θ = π/4; 𝐝 = [cos(θ),sin(θ)]
uᵢ = x -> exp(im * k₁ * x⋅𝐝) # plane-wave incident field
∇uᵢ = x -> im*k₁*uᵢ(x)*𝐝     # gradient of incident field

rhs₁ = map(Q) do q
    x = q.coords
    return uᵢ(x)
end

rhs₂ = map(Q) do q
    x = q.coords
    n = q.normal
    return ∇uᵢ(x)⋅n
end

rhs = [rhs₁;rhs₂]

using IterativeSolvers
sol, hist =
    gmres(L, rhs; log = true, abstol = 1e-6, verbose = false, restart = 100, maxiter = 100)
@show hist

# sol = L \ rhs
nQ = size(Q,1)
sol = reshape(sol,nQ,2)
φ,ψ = sol[:,1],sol[:,2]

𝒮₁, 𝒟₁ = Inti.single_double_layer_potential(; pde=pde₁, source = Q)
𝒮₂, 𝒟₂ = Inti.single_double_layer_potential(; pde=pde₂, source = Q)

uₛ  = x ->  𝒟₁[φ](x) - 𝒮₁[ψ](x)
uₜ  = x -> -𝒟₂[φ](x) + 𝒮₂[ψ](x)

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

using CairoMakie
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
# create a *view* (in the sense of *Gmsh*) of the solution on a punctured plane.

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
    xl,yl,zl = -2*radius,-2*radius,0
    Δx, Δy = 4*radius, 4*radius
    rectangle_tag = gmsh.model.occ.addRectangle(xl, yl, zl, Δx, Δy)
    outDimTags, _ = gmsh.model.occ.cut([(2, rectangle_tag)], [(3, sphere_tag)], -1, true, false)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [sphere_tag], -1, "omega")
    gmsh.model.addPhysicalGroup(2, [dt[2] for dt in outDimTags], -1, "sigma")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    visualize && gmsh.fltk.run()
    gmsh.option.setNumber("Mesh.SaveAll", 1) # otherwise only the physical groups are saved
    gmsh.write(name)
    gmsh.finalize()
end

# As before, lets write a file with our mesh, and import it into `Inti.jl`:

name = joinpath(@__DIR__, "sphere.msh")
gmsh_sphere(; meshsize, order = gorder, name, visualize=false)
Inti.clear_entities!()
Ω, msh = Inti.import_mesh_from_gmsh_file(name; dim = 3)
Γ = Inti.boundary(Ω)

# Note that for this example we relied instead on the labels to the entities in
# order to extract the relevant domains `Ω` and `Σ`. We can now create a
# quadrature as before

Γ_msh = view(msh,Γ)
Q = Inti.Quadrature(Γ_msh; qorder = 4)

# !!! tip
#       If you pass `visualize=true` to `gmsh_sphere`, it will open a window
#       with the current mode. This is done by calling `gmsh.fltk.run()`. Note
#       that the main julia thread will be blocked until the window is closed.

# !!! tip "Writing/reading a mesh from disk"
#       Writing and reading a mesh to/from disk can be time consuming. You can
#       avoid doing so by using [`import_mesh_from_gmsh_file`](@ref Inti.import_mesh_from_gmsh_file)
#       and [`import_mesh_from_gmsh_model`](@ref Inti.import_mesh_from_gmsh_model) functions on an
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
Σ_msh = view(msh,Σ)
target = Inti.nodes(Σ_msh)

S,D = Inti.single_double_layer(;
    pde,
    target,
    source = Q,
    compression = (method = :fmm, tol=1e-4),
    correction = (method = :none, maxdist = 5 * meshsize),
)

ui_eval_msh = uᵢ.(target)
us_eval_msh = D*σ - im*k*S*σ
u_eval_msh = ui_eval_msh + us_eval_msh
nothing #hide

# Finalize, we use gmsh to visualize the scattered field:
gmsh.initialize()
Inti.write_gmsh_model(msh)
Inti.write_gmsh_view!(Σ_msh, real(u_eval_msh); name="sigma real")
# Inti.write_gmsh_view!(Σ_msh, imag(us_eval_msh); name="sigma imag")
Inti.write_gmsh_view!(Γ_msh, x -> 0, name = "gamma real")
# Inti.write_gmsh_view!(Γ_msh, x -> -imag(uᵢ(x)), name = "gamma imag")
# Launch the GUI to see the results:
"-nopopup" in ARGS || gmsh.fltk.run()
gmsh.finalize()
# Add a gmsh view of the solution and save it:
