using Test
using LinearAlgebra
using Inti
using Random
using Meshes
using CairoMakie

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
pde = Inti.Laplace(; dim = N)
qorder = 3

K = 5:5
H = [0.2 * 2.0^(-i) for i in 0:3]
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "h", ylabel = "error", xscale = log10, yscale = log10)
k = 5
err = Float64[]
for h in H
    Inti.clear_entities!()
    Ω, msh = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = h, order = 2)
    Γ = Inti.external_boundary(Ω)

    ##

    quad = Inti.Quadrature(msh[Γ]; qorder)
    σ = t == :interior ? 1 / 2 : -1 / 2
    xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
    T = Inti.default_density_eltype(pde)
    c = rand(T)
    u = (qnode) -> Inti.SingleLayerKernel(pde)(qnode, xs) * c
    dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(pde)(qnode, xs) * c
    γ₀u = map(u, quad)
    γ₁u = map(dudn, quad)
    γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
    γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
    # single and double layer
    G = Inti.SingleLayerKernel(pde)
    S = Inti.IntegralOperator(G, quad)
    Smat = Inti.assemble_matrix(S)
    dG = Inti.DoubleLayerKernel(pde)
    D = Inti.IntegralOperator(dG, quad)
    Dmat = Inti.assemble_matrix(D)
    e0 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm

    green_multiplier = fill(-0.5, length(quad))
    # δS, δD = Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)

    # qnodes = Inti.local_bdim_correction(pde, quad, quad; green_multiplier)
    # X = [q.coords[1] for q in qnodes]; Y = [q.coords[2] for q in qnodes]
    # u = [q.normal[1] for q in qnodes]; v = [q.normal[2] for q in qnodes]
    # fig, _, _ = scatter(X, Y)
    # arrows!(X, Y, u, v, lengthscale=0.01)
    # display(fig)

    δS, δD = Inti.local_bdim_correction(pde, quad, quad; green_multiplier, kneighbor = k)
    Sdim = Smat + δS
    Ddim = Dmat + δD
    # Sdim, Ddim = Inti.single_double_layer(;
    #     pde,
    #     target      = quad,
    #     source      = quad,
    #     compression = (method = :none,),
    #     correction  = (method = :ldim,),
    # )
    e1 = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
    # @show norm(e0, Inf)
    @show norm(e1, Inf)
    push!(err, e1)
end

scatterlines!(ax, H, err; linewidth = 2, marker = :circle, label = " k = $k")

# add some reference slopes
for slope in (qorder-2):(qorder+2)
    ref = err[end] / H[end]^slope
    lines!(ax, H, ref * H .^ slope; linestyle = :dash, label = "slope $slope")
end
axislegend()

display(fig)
