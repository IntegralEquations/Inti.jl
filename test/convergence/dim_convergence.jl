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
t = :interior
σ = t == :interior ? 1 / 2 : -1 / 2
N = 2
# pde = Inti.Laplace(; dim=N)
pde = Inti.Helmholtz(; dim = N, k = 2π)
# pde = Inti.Stokes(; dim = N, μ = 1.2)
@info "Greens identity ($t) $(N)d $pde"
Inti.clear_entities!()
center = Inti.svector(i -> 0.1, N)
radius = 1
n = 3
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
    gmsh.model.occ.addDisk(center[1], center[2], 0, radius, radius)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    Ω = Inti.gmsh_import_domain(; dim = 2)
    M = Inti.gmsh_import_mesh(Ω; dim = 2)
    gmsh.finalize()
    Γ = Inti.external_boundary(Ω)
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
    S = Inti.IntegralOperator(G, Q)
    S0 = Inti.assemble_dense_matrix(S)
    D = Inti.IntegralOperator(dG, Q)
    D0 = Inti.assemble_dense_matrix(D)
    e0 = norm(S0 * γ₁u - D0 * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
    δS, δD = Inti.bdim_correction(pde, Q, Q, S0, D0)
    Smat, Dmat = S0 + δS, D0 + δD
    e1 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
    push!(ee0, e0)
    push!(ee1, e1)
end

##
order = n + 1
fig = Figure()
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "h", ylabel = "error")
scatterlines!(ax, hh, ee0; m = :x, label = "nocorrection")
scatterlines!(ax, hh, ee1; m = :x, label = "dim correction")
ref = hh .^ order
iref = length(ref)
lines!(ax, hh, ee1[iref] / ref[iref] * ref; label = L"\mathcal{O}(h^%$order)", ls = :dash)
axislegend(ax)
fig
