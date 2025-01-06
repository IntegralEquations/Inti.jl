using Test
using LinearAlgebra
using Inti
using Random
using Meshes
using CairoMakie
using StaticArrays
using QuadGK
using ForwardDiff

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
# pde = Inti.Stokes(; dim = 2, μ = 1.0)
qorder = 5

K = 5:5
H = [0.2 * 2.0^(-i) for i in 2:10]
err1 = Float64[]
err2 = Float64[]

k = 5
Inti.clear_entities!()

# Ω, msh = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = h, order = 2)
# Γ = Inti.external_boundary(Ω)

##### Circle
# a, b = 0, 1
# χ = s -> SVector(cos(2π * s), sin(2π * s))
# xt = χ((3a+b)/4)
# Γ1 = Inti.parametric_curve(χ, a, (a+b)/2) |> Inti.Domain
# Γ2 = Inti.parametric_curve(χ, (a+b)/2, b) |> Inti.Domain
# Γ = Γ1 ∪ Γ2

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
δ = 1e-8
a, b = 0, 1
χ = s -> SVector(
    (1 + cos(2s * 2π)/2) * cos(s * 2π),
    (1 + (2-δ)*cos(2s * 2π)/2) * sin(s * 2π),
)
xt = χ((3a+b)/4)
Γ1 = Inti.parametric_curve(χ, a, (a+b)/2) |> Inti.Domain
Γ2 = Inti.parametric_curve(χ, (a+b)/2, b) |> Inti.Domain
Γ = Γ1 ∪ Γ2
# ##

FIG = Figure()
AX  = Axis(FIG[1, 1]; aspect=1)
msh = Inti.meshgen(Γ; meshsize = H[1])
quad = Inti.Quadrature(msh[Γ]; qorder=1)
scatter!(AX, Inti.coords.(quad))
scatter!(AX, [xt])
display(FIG)

for h in H
    # k = ceil(Int, 0.1 / h)
    
    msh = Inti.meshgen(Γ; meshsize = h)
    Γ_msh = msh[Γ]
    nel = sum(Inti.element_types(Γ_msh)) do E
        return length(Inti.elements(Γ_msh, E))
    end
    @info h, k, nel
    ##

    quad = Inti.Quadrature(Γ_msh; qorder)
    qs = argmin(quad) do q
        norm(q.coords-xt)
    end
    xs = qs.coords
    @info xs
    # ε = 1
    # b = x -> -ε < x < ε ? exp(-1/(1 - (x / ε)^2)) / ε : 0
    # u = x -> 0 ≤ x % (2π) < π || -2π ≤ x % (2π) < -π ? 1 : 0
    # bu = x -> quadgk(y -> b(y) * u(x - y), -ε, ε, atol=1e-15)[1]
    # bu = x -> sin(x)

    # u = x -> x[2] > 0 ? 1 : 0
    u = x -> x[2]
    u = x -> x[2] / norm(x)
    # u = x -> 1
    uvec = map(u, Inti.coords.(quad))
    # uvec = [bu(atan(x[2], x[1])) for x in Inti.coords.(quad)]
    uvec_norm = norm(uvec, Inf)
    # single and double layer
    G = Inti.SingleLayerKernel(pde)
    S = Inti.IntegralOperator(G, [xs], quad)
    Smat = Inti.assemble_matrix(S)
    dG = Inti.DoubleLayerKernel(pde)
    D = Inti.IntegralOperator(dG, [xs], quad)
    Dmat = Inti.assemble_matrix(D)
    ref, err_ref = quadgk(a, b, atol=1e-15) do s
        G(xs, χ(s)) * u(χ(s)) * norm(ForwardDiff.derivative(χ, s))        
    end
    @show err_ref
    e0 = norm(Smat ⋅ uvec - ref, Inf) / uvec_norm

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
        [xs],
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
    e1 = norm(Sdim ⋅ uvec - ref, Inf) / uvec_norm

    tdim = @elapsed δS, δD =
        Inti.bdim_correction(pde, [xs], quad, Smat, Dmat; green_multiplier)
    Sdim = Smat + δS
    Ddim = Dmat + δD
    e2 = norm(Sdim ⋅ uvec - ref, Inf) / uvec_norm
    # @show norm(e0, Inf)
    @show e1
    @show e2
    @show tldim
    @show tdim
    push!(err1, e1)
    push!(err2, e2)
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "h", ylabel = "error", xscale = log10, yscale = log10)

scatterlines!(ax, H, err1; linewidth = 2, marker = :circle, label = " local")
scatterlines!(ax, H, err2; linewidth = 2, marker = :circle, label = "global")

# add some reference slopes
q = div((qorder + 1), 2)
for slope in (1,q+1)
    ref = err2[1] / H[1]^slope
    lines!(ax, H, ref * H .^ slope; linestyle = :dash, label = "slope $slope")
end
axislegend(; position = :lt)

display(fig)


# ε = 1
# b = x -> -ε < x < ε ? exp(-1/(1 - (x / ε)^2)) / ε : 0
# u = x -> 0 ≤ x % (2π) < π || -2π ≤ x % (2π) < -π ? 1 : 0
# bu = x -> quadgk(y -> b(y) * u(x - y), -ε, ε, atol=1e-15)[1]

# fig = Figure()
# ax = Axis(fig[1, 1])
# X = LinRange(-2π-1, 2π+1, 1000)
# lines!(ax, X, u.(X))
# lines!(ax, X, b.(X))
# lines!(ax, X, bu.(X))
# display(fig)