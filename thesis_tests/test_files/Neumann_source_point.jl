using Test
using LinearAlgebra
using Inti
using Random
using Meshes
using CairoMakie
using StaticArrays

include("../test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
# pde = Inti.Stokes(; dim = 2, μ = 1.0)
qorder = 5

K = 5:5
H = [0.2 * 2.0^(-i) for i in 2:6]
errl = Float64[]
errg = Float64[]

FIG = Figure()
AX  = Axis(FIG[1, 1]; aspect=1)
Inti.clear_entities!()
    ##### Circle
    χ = s -> SVector(cos(s), sin(s))
    xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
    Γ = Inti.parametric_curve(χ, -π, π) |> Inti.Domain
    tset = [0.5 * χ(s) for s in LinRange(-π, π, 10)]

    ##### Kite
    # Γ =
    #     Inti.parametric_curve(0.0, 1.0; labels = ["kite"]) do s
    #         return SVector(
    #             2.5 + cos(2π * s[1]) + 0.65 * cos(4π * s[1]) - 0.65,
    #             1.5 * sin(2π * s[1]),
    #         )
    #     end |> Inti.Domain
    # xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
    
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
    # xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
    # tset1 = map(s -> 0.5χ(s) - SVector(1 + δ/2, 0), LinRange(0, 2π, 10))
    # tset2 = map(s -> 0.5χ(s) + SVector(1 + δ/2, 0), LinRange(0, 2π, 10))
    # tset  = tset1 ∪ tset2

    ##### 8-like
    # δ = 0.001
    # χ = s -> SVector(
    #     (1 + cos(2s)/2) * cos(s),
    #     (1 + (2-δ)*cos(2s)/2) * sin(s),
    # )
    # xs = t == :interior ? ntuple(i -> 1, N) : ntuple(i -> 0.1, N)
    # Γ = Inti.parametric_curve(χ, -π, π) |> Inti.Domain
    # tset = [0.5SVector(cos(s), sin(s)) - SVector(0.8, 0) for s in LinRange(-π, π, 10)] ∪
    #        [0.5SVector(cos(s), sin(s)) + SVector(0.8, 0) for s in LinRange(-π, π, 10)]
    ##

    lines!(AX, χ.(LinRange(-π, π, 100)))
    scatter!(AX, [xs])
    scatter!(AX, tset)
    display(FIG)

for h in H
    # k = ceil(Int, 0.1 / h)
    k = 10

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
    u  = (qnode) -> Inti.SingleLayerKernel(pde)(qnode, xs) * c
    un = (qnode) -> Inti.AdjointDoubleLayerKernel(pde)(qnode, xs) * c
    ubnd = map(un, quad)
    utst = map(u, tset)
    utst_norm = norm(utst, Inf)
    # single and double layer
    G = Inti.SingleLayerKernel(pde)
    S = Inti.IntegralOperator(G, quad)
    Smat = Inti.assemble_matrix(S)
    dG = Inti.DoubleLayerKernel(pde)
    D = Inti.IntegralOperator(dG, quad)
    Dmat = Inti.assemble_matrix(D)

    σ    = (Smat + I/2) \ ubnd
    usol = Inti.IntegralOperator(G, tset, quad) * σ
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
    σ    = (Sdim + I/2) \ ubnd
    usol = Inti.IntegralOperator(G, tset, quad) * σ
    eloc   = norm(usol - utst, Inf) / utst_norm

    tdim = @elapsed δS, δD =
        Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
    Sdim = Smat + δS
    Ddim = Dmat + δD
    σ    = (Sdim + I/2) \ ubnd
    usol = Inti.IntegralOperator(G, tset, quad) * σ
    eglo   = norm(usol - utst, Inf) / utst_norm
    # @show norm(e0, Inf)
    @show eloc
    @show eglo
    @show tldim
    @show tdim
    push!(errl, eloc)
    push!(errg, eglo)
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "h", ylabel = "error", xscale = log10, yscale = log10)

scatterlines!(ax, H, errl; linewidth = 2, marker = :circle, label = " local")
scatterlines!(ax, H, errg; linewidth = 2, marker = :circle, label = "global")

# add some reference slopes
for slope in (qorder-2):(qorder+2)
    ref = errg[end] / H[end]^slope
    lines!(ax, H, ref * H .^ slope; linestyle = :dash, label = "slope $slope")
end
axislegend(; position = :lt)

display(fig)
