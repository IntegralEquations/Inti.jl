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
pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
# pde = Inti.Stokes(; dim = 2, μ = 1.0)
qorder = 5

K = 5:5
H = [0.2 * 2.0^(-i) for i in 2:6]
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "h", ylabel = "error", xscale = log10, yscale = log10)
err1 = Float64[]
err2 = Float64[]
for h in H
    # k = ceil(Int, 0.1 / h)
    k = 10
    Inti.clear_entities!()

    # Ω, msh = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = h, order = 2)
    # Γ = Inti.external_boundary(Ω)

    ##### Circle
    # Γ = Inti.parametric_curve(0, 2π) do s
    #     return SVector(cos(s), sin(s))
    # end |> Inti.Domain

    ##### Kite
    # Γ =
    #     Inti.parametric_curve(0.0, 1.0; labels = ["kite"]) do s
    #         return SVector(
    #             2.5 + cos(2π * s[1]) + 0.65 * cos(4π * s[1]) - 0.65,
    #             1.5 * sin(2π * s[1]),
    #         )
    #     end |> Inti.Domain
    
    ##### Two circles
    # δ = 0.01
    # Γ₁ = Inti.parametric_curve(0.0, 2π) do s
    #     return SVector(cos(s) - 1 - δ / 2, sin(s))
    # end |> Inti.Domain
    # Γ₂ = Inti.parametric_curve(0.0, 2π) do s
    #     return SVector(cos(s) + 1 + δ / 2, sin(s))
    # end |> Inti.Domain
    # Γ = Γ₁ ∪ Γ₂

    ##### 8-like
    δ = 0.001
    Γ = Inti.parametric_curve(-π, π) do s
            return SVector(
                (1 + cos(2s)/2) * cos(s),
                (1 + (2-δ)*cos(2s)/2) * sin(s),
            )
        end |> Inti.Domain
    ##

    msh = Inti.meshgen(Γ; meshsize = h)
    Γ_msh = msh[Γ]
    nel = sum(Inti.element_types(Γ_msh)) do E
        return length(Inti.elements(Γ_msh, E))
    end
    @info h, k, nel
    ##

    quad = Inti.Quadrature(Γ_msh; qorder)
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

    tldim = @elapsed δS, δD = Inti.local_bdim_correction(
        pde,
        quad,
        quad;
        green_multiplier,
        kneighbor = k,
        maxdist = 10 * h,
        qorder_aux = 100 * ceil(Int, abs(log(h))),
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
    e1 = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm

    tdim = @elapsed δS, δD =
        Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
    Sdim = Smat + δS
    Ddim = Dmat + δD
    e2 = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
    # @show norm(e0, Inf)
    @show e1
    @show e2
    @show tldim
    @show tdim
    push!(err1, e1)
    push!(err2, e2)
end

scatterlines!(ax, H, err1; linewidth = 2, marker = :circle, label = " local")
scatterlines!(ax, H, err2; linewidth = 2, marker = :circle, label = "global")

# add some reference slopes
for slope in (qorder-2):(qorder+2)
    ref = err2[end] / H[end]^slope
    lines!(ax, H, ref * H .^ slope; linestyle = :dash, label = "slope $slope")
end
axislegend(; position = :lt)

display(fig)
