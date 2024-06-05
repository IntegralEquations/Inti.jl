# Layer potentials

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this tutorial"
    - Nearly singular evaluation of layer potentials
    - Creating a smooth domain with splines using Gmsh.jl

## Direct evaluation of layer potentials

```@example layer_potentials
using Inti, StaticArrays, LinearAlgebra, Meshes, GLMakie, Gmsh
# define the PDE
k = 4π
pde = Inti.Helmholtz(; dim = 2, k)
meshsize = 2π / k / 10
# create the domain and mesh using the Gmsh API
gmsh.initialize()
kite = Inti.gmsh_curve(0, 1; meshsize) do s
    SVector(0.25, 0.0) + SVector(cos(2π * s) + 0.65 * cos(4π * s[1]) - 0.65, 1.5 * sin(2π * s))
end
cl = gmsh.model.occ.addCurveLoop([kite])
surf = gmsh.model.occ.addPlaneSurface([cl])
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
msh = Inti.import_mesh(; dim = 2)
gmsh.finalize()
# extract the domain Ω from the mesh entities
ents = Inti.entities(msh)
Ω = Inti.Domain(e->Inti.geometric_dimension(e) == 2, ents)
# create a quadrature on the boundary
Γ = Inti.boundary(Ω)
Q = Inti.Quadrature(view(msh,Γ); qorder = 5)
# construct an exact interior solution as a sum of random plane waves
dirs  = [SVector(cos(θ), sin(θ)) for θ in 2π*rand(10)]
coefs = rand(ComplexF64, 10)
u  =  (x)   -> sum(c*exp(im*k*dot(x, d)) for (c,d) in zip(coefs, dirs))
du =  (x,ν) -> sum(c*im*k*dot(d, ν)*exp(im*k*dot(x, d)) for (c,d) in zip(coefs, dirs))
# plot it 
Ω_msh = view(msh, Ω)
target = Inti.nodes(Ω_msh)
viz(Ω_msh; showsegments = false, axis = (aspect = DataAspect(), ), color = real(u.(target)))
```

Let us now compute the layer potentials of the exact solution on the boundary,
and evaluate the error on the target nodes:

```@example layer_potentials
# evaluate the layer potentials
𝒮, 𝒟 = Inti.single_double_layer_potential(; pde, source = Q)
γ₀u = map(q -> u(q.coords), Q)
γ₁u = map(q -> du(q.coords, q.normal), Q)
uₕ = x -> 𝒮[γ₁u](x) - 𝒟[γ₀u](x)
# plot the error on the target nodes
er_log10 = log10.(abs.(u.(target) - uₕ.(target)))
colorrange = extrema(er_log10)
fig, ax, pl = viz(Ω_msh;
    color = er_log10,
    colormap = :viridis,
    colorrange,
    axis = (aspect = DataAspect(),), 
    interpolate=true
)
Colorbar(fig[1, 2]; label = "log₁₀(error)", colorrange)
fig
```

## Near-field correction of layer potentials

```@example layer_potentials
S, D = Inti.single_double_layer(; pde, target, source = Q,
    compression = (method = :none, ),
    correction = (method = :dim, target_location = :inside, maxdist = 0.2)
)
er_log10_cor = log10.(abs.(S*γ₁u - D*γ₀u - u.(target)))
colorrange = extrema(er_log10_cor)
fig, ax, pl = viz(Ω_msh;
    color = er_log10_cor,
    colormap = :viridis,
    colorrange,
    axis = (aspect = DataAspect(),), 
    interpolate=true
)
Colorbar(fig[1, 2]; label = "log₁₀(error)", colorrange)
fig
```
