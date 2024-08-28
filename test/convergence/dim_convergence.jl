using Test
using LinearAlgebra
using Inti
using Random
using CairoMakie
using LaTeXStrings
using Gmsh
Random.seed!(1)

atol = 0
rtol = 1e-8
t = :exterior
σ = t == :interior ? 1 / 2 : -1 / 2
# For N = 3 one needs to use compression, e.g. `compression = :fmm`, for the operators below
N = 2
pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; dim = N, k = 2π)
# pde = Inti.Stokes(; dim = N, μ = 1.2)
@info "Greens identity ($t) $(N)d $pde"
Inti.clear_entities!()
center = Inti.svector(i -> 0.1, N)
radius = 1
n = 6
qorder = n - 1
hh = [1 / 2^i for i in 1:5]
ee0 = Float64[]
ee1 = Float64[]

for h in hh
    meshsize = h
    # set meshsize in gmsh
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    # set gmsh verbosity to 2
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.mesh.setOrder(2)
    Inti.clear_entities!()
    if N == 2
        gmsh.model.occ.addDisk(center[1], center[2], 0, 2 * radius, radius)
    else
        gmsh.model.occ.addSphere(center[1], center[2], center[3], radius)
    end
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    M = Inti.import_mesh(; dim = N)
    gmsh.finalize()
    Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == N - 1, Inti.entities(M))
    Q = Inti.Quadrature(view(M, Γ); qorder)
    @show Q
    xs = if t == :interior
        center + Inti.svector(i -> 2 * radius, N)
    else
        center + Inti.svector(i -> 0.5 * radius, N)
    end
    T = Inti.default_density_eltype(pde)
    c = rand(T)
    G = Inti.SingleLayerKernel(pde)
    dG = Inti.DoubleLayerKernel(pde)
    u = (qnode) -> G(xs, qnode) * c
    dudn = (qnode) -> transpose(dG(xs, qnode)) * c
    γ₀u = map(u, Q)
    γ₁u = map(dudn, Q)
    γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
    γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
    # single and double layer
    S0, D0 = Inti.single_double_layer(;
        pde,
        target = Q,
        source = Q,
        compression = (method = :none,),
        correction = (method = :none,),
    )
    S1, D1 = Inti.single_double_layer(;
        pde,
        target = Q,
        source = Q,
        compression = (method = :none,),
        correction = (method = :dim, target_location = :on),
    )
    e0 = norm(S0 * γ₁u - D0 * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
    e1 = norm(S1 * γ₁u - D1 * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
    push!(ee0, e0)
    push!(ee1, e1)
end

##
order = n + 1
fig = Figure()
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "h", ylabel = "error")
scatterlines!(ax, hh, ee0; marker = :x, label = "nocorrection")
scatterlines!(ax, hh, ee1; marker = :x, label = "dim correction")
ref = hh .^ order
iref = length(ref)
lines!(
    ax,
    hh,
    ee1[iref] / ref[iref] * ref;
    label = L"\mathcal{O}(h^%$order)",
    linestyle = :dash,
)
axislegend(ax)
fig
