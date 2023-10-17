import Pkg               #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir) #src

#=

# [Sphere scattering](@id sphere-scattering)

=#

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](sphere_scattering.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/sphere_scattering.ipynb)

#=

!!! note "Important points covered in this tutorial"
    - Create a high-order surface mesh using `Gmsh`
    - Compressing boundary integral operators using `HMatrices`
    - Using `IterativeSolvers` and `LinearMaps` to solve a boundary integral equation

In this tutorial we solve a three-dimensional scattering problem. The scatterer
will be a sphere, but we will use curved elements to mesh its surface!

First, let us load all the packages we will need:

=#

using Inti             # for the boundary integral operators
using Gmsh             # for meshing
using HMatrices        # for compressing the operators
using LinearMaps       # for lazily combining various linear operators
using IterativeSolvers # for solving the linear system usign GMRES

#=

In the following subsections, we will

- [create the mesh](@ref mesh-creation)
- [set up the boundary integral equation](@ref boundary-integral-equation)
- [compress the operators](@ref operator-compression)
- [solve the boundary integral equation](@ref solving-the-equation)
- [compare to an exact solution](@ref exact-solution)

## [Mesh creation](@id mesh-creation)

We delegate the mesh creation to [`Gmsh`](https://gmsh.info), which is a
powerful open source meshing tool. For this example, we will create a sphere of
radius `r` centered at the origin. We will mesh the sphere with
[`LagrangeElement`](@ref)s of order `gorder`, and we will use a mesh size of
`h`. These are kept as parameters to the script for convenience, so that you can
play with them and see how they may affect the runtime and accuracy of the
solution. To have an idea of the absolute and relative times of the various
steps involved, we will collect the time spent in each step using the `@elapsed`
macro.

=#

## parameters definition
r = 0.5
k = 2π * 5
λ = 2π / k
h = λ / 3
qorder = 2 # quadrature order
gorder = 2 # geometrical order
atol = 1e-6 # absolute tolerance of the iterative solver and `HMatrix` assembly

# create a sphere with gmsh and mesh is using second order elements
t_mesh = @elapsed begin
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.model.occ.addSphere(0, 0, 0, r)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(gorder)
    Inti.clear_entities!()
    Ω   = Inti.gmsh_import_domain(; dim = 3)
    msh = Inti.gmsh_import_mesh(Ω)
    gmsh.finalize()
end

t_quad = @elapsed begin
    Γ = Inti.external_boundary(Ω)
    Q = Inti.Quadrature(view(msh, Γ); qorder)
end

#=

We can now set up the boundary integral equation. We will use a direct
formulation, where we seek to solve the following equation for the density

```math
    \frac{I}{2} u + \mathrm{D}[u] = \mathrm{S}[\partial_n u]
```

where $u$ is the solution to the scattering problem.
```

=#

pde = Inti.Helmholtz(; dim = 3, k = k)
G   = Inti.SingleLayerKernel(pde)
dG  = Inti.DoubleLayerKernel(pde)
Sop = Inti.IntegralOperator(G, Q, Q)
Dop = Inti.IntegralOperator(dG, Q, Q)

#=

Note that the `Sop` and `Dop` are lazy operators; that is, they don't actually
create a matrix representation of the underlying integral operator. And that is
a good thing: representing Sop and Dop as dense matrices would already require

=#

size_bytes = (length(Sop) + length(Dop)) * 16 # 16 bytes per complex entry
println(trunc(size_bytes / 10^9; digits = 2), " GB of memory")

#=

which is more than the memory available on most personal compters.

To deal with this common issue, we will use `HMatrices` to compress the operators:

=#

using HMatrices, LinearAlgebra
BLAS.set_num_threads(1)

t_shmat = @elapsed S_hmat = assemble_hmatrix(Sop; atol = atol, threads = true)
t_dhmat = @elapsed D_hmat = assemble_hmatrix(Dop; atol = atol, threads = true)
@info "HMatrix assembly took $(t_shmat + t_dhmat) ($t_shmat + $t_dhmat) seconds"

#=

We will know solve for a boundary density using a combined-field formulation. It
will convenient to lazily combine the operators using `LinearMaps`, and to use
the `gmres` solver:

=#

using LinearAlgebra
using IterativeSolvers
using LinearMaps

L0 = 0.5 * I + LinearMap(D_hmat) - im * k * LinearMap(S_hmat)

# incident wave
uᵢ  = x -> exp(im * k * x[1])
rhs = [-uᵢ(q.coords) for q in Q]

σ = gmres(L0, rhs; abstol = 1e-6, verbose = false, restart = 10_000)

#=

We will check against an exact solution

=#
using SpecialFunctions
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

#=

Let us now build the representation, and evaluate at some point far from the sphere:

=#

using StaticArrays
𝒮 = Inti.IntegralPotential(G, Q)
𝒟 = Inti.IntegralPotential(dG, Q)
u0 = x -> 𝒟[σ](x) - im * k * 𝒮[σ](x)

#=

The object `u` above is a numerical approximation of the solutinThe next step
consists of computing a correction to the `HMatrix` approximation to take into
account the singular and nearly-singular behavior of the integral kernel. In
this example, we use *density interpolation method* for that purpose:

=#

t_correction = @elapsed begin
    δS, δD = Inti.bdim_correction(pde, Q, Q, S_hmat, D_hmat)
end

@info "Correction took $t_correction seconds"

L1 = L0 + LinearMap(δD) - im * k * LinearMap(δS)
μ = gmres(L1, rhs; abstol = 1e-4, verbose = false, restart = 10_000)

u1 = x -> 𝒟[μ](x) - im * k * 𝒮[μ](x)

#=

We can now compare the error in the uncorrected and corrected solutions on a
sphere of radius `5r`:

=#

e0 = Float64[]
e1 = Float64[]
for θ in 0:π/10:π
    for φ in 0:2π/10:2π
        x  = 5r * SVector(cos(φ) * sin(θ), sin(φ) * sin(θ), cos(θ))
        ve = sphere_helmholtz_soundsoft(x; radius = r, k = k, θin = π / 2)
        v0 = u0(x)
        v1 = u1(x)
        push!(e0, abs(v0 - ve))
        push!(e1, abs(v1 - ve))
    end
end

#=

Finally, we display all of the information:

=#
