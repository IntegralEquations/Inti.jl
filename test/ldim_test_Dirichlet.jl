using Test
using LinearAlgebra
using Inti
using Random
using Meshes
using CairoMakie
using StaticArrays

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
# t = :exterior
α, β = 1, 0
pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
# pde = Inti.Stokes(; dim = 2, μ = 1.0)
qorder = 5

K = 5:5
H = [0.2 * 2.0^(-i) for i in 2:6]
err0 = Float64[]
err1 = Float64[]
err2 = Float64[]

Inti.clear_entities!()
##### Circle
χ = s -> SVector(1cos(s), 1sin(s))
Γ = Inti.parametric_curve(χ, -π, π) |> Inti.Domain
if t == :interior
    xs = ntuple(i -> 3, N)
    tset = [0.5 * χ(s) for s in LinRange(-π, π, 10)]
    # tset = [SVector(10, 0)]
else
    xs = ntuple(i -> 0.1, N)
    tset = [2 * χ(s) for s in LinRange(-π, π, 10)]
end

##### Kite
# χ = s -> SVector(
#     2.5 + cos(s) + 0.65 * cos(2s) - 0.65,
#     1.5 * sin(s),
# )
# Γ = Inti.parametric_curve(χ, 0.0, 2π)
# if t == :interior
#     xs = ntuple(i -> 3, N)
#     tset = [SVector(0.5cos(s)+2.5, 0.5sin(s)) for s in LinRange(-π, π, 10)]
# else
#     xs = SVector(2.5, 0.1)
#     tset = [SVector(3cos(s)+2.5, 3sin(s)) for s in LinRange(-π, π, 10)]
# end

##### Two circles
# δ = 0.01
# χ = s -> SVector(cos(s), sin(s))
# Γ₁ = Inti.parametric_curve(0.0, 2π) do s
#     return χ(s) - SVector(1 + δ/2, 0)
# end |> Inti.Domain
# Γ₂ = Inti.parametric_curve(0.0, 2π) do s
#     return χ(s) + SVector(1 + δ/2, 0)
# end |> Inti.Domain
# Γ = Γ₁ ∪ Γ₂
# if t == :interior
#     xs = ntuple(i -> 3, N)
#     tset = [SVector(0.5cos(s) - 1 - δ/2, 0.5sin(s)) for s in LinRange(-π, π, 10)] ∪
#            [SVector(0.5cos(s) + 1 + δ/2, 0.5sin(s)) for s in LinRange(-π, π, 10)]
# else
#     xs = SVector(1, 0.1)
#     tset = [SVector(5cos(s), 5sin(s)) for s in LinRange(-π, π, 10)]
# end

##### 8-like
# δ = 0.001
# χ = s -> SVector(
#     (1 + cos(2s)/2) * cos(s),
#     (1 + (2-δ)*cos(2s)/2) * sin(s),
# )
# Γ = Inti.parametric_curve(χ, -π, π) |> Inti.Domain
# if t == :interior
#     xs = ntuple(i -> 1, N)
#     tset = [0.5SVector(cos(s), sin(s)) - SVector(0.8, 0) for s in LinRange(-π, π, 10)] ∪
#            [0.5SVector(cos(s), sin(s)) + SVector(0.8, 0) for s in LinRange(-π, π, 10)]
# else
#     xs = SVector(1, 0.1)
#     tset = [SVector(5cos(s), 5sin(s)) for s in LinRange(-π, π, 10)]
# end
##

FIG = Figure()
AX  = Axis(FIG[1, 1]; aspect=1)
lines!(AX, χ.(LinRange(-π, π, 100)))
# lines!(AX, [χ(s) - SVector(1 + δ/2, 0) for s in LinRange(-π, π, 100)])
# lines!(AX, [χ(s) + SVector(1 + δ/2, 0) for s in LinRange(-π, π, 100)])
scatter!(AX, [xs])
scatter!(AX, tset)
display(FIG)

for h in H
    # k = ceil(Int, 0.1 / h)
    k = 5

    # Ω, msh = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = h, order = 2)
    # Γ = Inti.external_boundary(Ω)


    msh = Inti.meshgen(Γ; meshsize = h)
    Γ_msh = msh[Γ]
    nel = sum(Inti.element_types(Γ_msh)) do E
        return length(Inti.elements(Γ_msh, E))
    end
    @info h, k, nel
    ##

    quad = Inti.Quadrature(Γ_msh; qorder)
    T = Inti.default_density_eltype(pde)
    c = rand(T)
    # u = (qnode) -> Inti.SingleLayerKernel(pde)(qnode, xs) * c
    u = qnode -> 1
    ubnd = map(u, quad)
    utst = map(u, tset)
    utst_norm = norm(utst, Inf)
    # single and double layer
    G = Inti.SingleLayerKernel(pde)
    S = Inti.IntegralOperator(G, quad)
    Smat  = Inti.assemble_matrix(S)
    Stest = Inti.IntegralOperator(G, tset, quad)

    dG = Inti.DoubleLayerKernel(pde)
    D = Inti.IntegralOperator(dG, quad)
    Dmat  = Inti.assemble_matrix(D)
    Dtest = Inti.IntegralOperator(dG, tset, quad)

    μ = t == :interior ? -0.5 : 0.5
    σ    = (α * Smat + β * (Dmat + μ*I)) \ ubnd
    @show norm(α * Smat * σ - ubnd, Inf)
    @show norm(ubnd, Inf)
    @show norm(Stest, Inf)
    usol = (α * Stest + β * Dtest) * σ
    @show  utst
    e0   = norm(usol - utst, Inf) / utst_norm

    green_multiplier = fill(-0.5, length(quad))
    # δS, δD = Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)

    # qnodes = Inti.local_bdim_correction(pde, quad, quad; green_multiplier)
    # X = [q.coords[1] for q in qnodes]; Y = [q.coords[2] for q in qnodes]
    # u = [q.normal[1] for q in qnodes]; v = [q.normal[2] for q in qnodes]
    # fig, _, _ = scatter(X, Y)
    # arrows!(X, Y, u, v, lengthscale=0.01)
    # display(fig)

    tldim = @elapsed δS, δD = Inti.local_bdim_correction(
        pde,
        quad,
        quad;
        green_multiplier,
        kneighbor = k,
        maxdist = 10 * h,
        qorder_aux = 20 * ceil(Int, abs(log(h))),
    )
    Sdim = Smat + δS
    Ddim = Dmat + δD
    # Sdim, Ddim = Inti.single_double_layer(;
    #     pde,
    #     target      = quad,
    #     source      = quad,
    #     compression = (method = :none,),
    #     correction  = (method = :ldim,),
    # )
    σ    = (α * Sdim + β * (Ddim + μ*I)) \ ubnd
    usol = (α * Stest + β * Dtest) * σ
    e1   = norm(usol - utst, Inf) / utst_norm

    tdim = @elapsed δS, δD =
        Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
    Sdim = Smat + δS
    Ddim = Dmat + δD
    σ    = (α * Sdim + β * (Ddim + μ*I)) \ ubnd
    e2   = norm(usol - utst, Inf) / utst_norm
    # @show norm(e0, Inf)
    @show e0
    @show e1
    @show e2
    @show tldim
    @show tdim
    push!(err0, e0)
    push!(err1, e1)
    push!(err2, e2)
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "h", ylabel = "error", xscale = log10, yscale = log10)

scatterlines!(ax, H, err0; linewidth = 2, marker = :circle, label = " no correction")
scatterlines!(ax, H, err1; linewidth = 2, marker = :circle, label = " local")
scatterlines!(ax, H, err2; linewidth = 2, marker = :circle, label = "global")

# add some reference slopes
for slope in (qorder-2):(qorder+2)
    ref = err2[end] / H[end]^slope
    lines!(ax, H, ref * H .^ slope; linestyle = :dash, label = "slope $slope")
end
axislegend(; position = :lt)

display(fig)
