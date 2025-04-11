using Test
using StaticArrays
using LinearAlgebra
using Inti
using Random
using Meshes
using CairoMakie
using LaTeXStrings
using QuadGK

SAVE = false

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
μ = t == :interior ? -0.5 : 0.5
pde = Inti.Laplace(; dim = N)
G = Inti.SingleLayerKernel(pde)
dG = Inti.DoubleLayerKernel(pde)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
# pde = Inti.Stokes(; dim = N, μ = 1.2)
α, β = 1, 0

##
##

GEOMETRY = "geometries/8-like.jl"
include(GEOMETRY)

# calculate reference boundary value by quadgk
σ = x -> cos(Inti.coords(x)[1]) * exp(Inti.coords(x)[2])
# u = x -> quadgk(a, b, atol=1e-15) do s
#     y  = s -> (coords=χ(s), normal=normalize(χn(s)))
#     (G(x, χ(s)) * σ_ref(χ(s))) * norm(χn(s))        
# end[1]

##
qorder, h, k = 5, 2.0^(-5), 10
msh = Inti.meshgen(Γ; meshsize = h)
Γ_msh = msh[Γ]
quad = Inti.Quadrature(Γ_msh; qorder)
σ_vec = map(σ, quad)
# ubnd = map(u, quad)
# single and double layer
S = Inti.IntegralOperator(G, quad)
Smat  = Inti.assemble_matrix(S)
D = Inti.IntegralOperator(dG, quad)
Dmat  = Inti.assemble_matrix(D)
green_multiplier = fill(-0.5, length(quad))

# calculate reference boundary value by global DIM
qorder_ref, h_ref = 5, 1e-4
msh = Inti.meshgen(Γ; meshsize = h_ref)
Γ_msh = msh[Γ]
quad_ref = Inti.Quadrature(Γ_msh; qorder = qorder_ref)
σ_ref = map(σ, quad_ref)
S = Inti.IntegralOperator(G, quad, quad_ref)
Sref  = Inti.assemble_matrix(S)
D = Inti.IntegralOperator(dG, quad, quad_ref)
Dref  = Inti.assemble_matrix(D)
δS, δD = Inti.bdim_correction(pde, quad, quad_ref, Sref, Dref; green_multiplier)
Sdim = Sref + δS
Ddim = Dref + δD
ubnd = (α*Sdim + β*Ddim) * σ_ref + β*μ*σ_vec

## calculate numeric values

δS, δD = Inti.local_bdim_correction(
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
σl   = (α*Sdim + β*(Ddim + μ*I)) \ ubnd
errl = abs.(σl - σ_vec)
# @show σ

tdim = @elapsed δS, δD =
    Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
Sdim = Smat + δS
Ddim = Dmat + δD
σg   = (α*Sdim + β*(Ddim + μ*I)) \ ubnd
errg = abs.(σg - σ_vec)

# normalize data
q = map(Inti.coords, quad)
# Merr = max(maximum(errl), maximum(errg))
errl = log10.(errl)
errg = log10.(errg)

theme = Theme(;
    Axis = (     
        autolimitaspect = 1,
        markersize = 30,
        # aspect = DataAspect(),
    ),
    fontsize = 20,
)
Makie.set_theme!(theme)

fig = Figure(size=(1400,500))
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
hideydecorations!(ax2)
m, M = min(minimum(errl), minimum(errg)), max(maximum(errl), maximum(errg))

cmap = :viridis
ms = 20
# m, M = min(minimum(El), minimum(Eg)), max(maximum(El), maximum(Eg))
scatter!(ax1, q, color=errl, colormap=cmap, markersize=ms, colorrange=[m,M])
scatter!(ax2, q, color=errg, colormap=cmap, markersize=ms, colorrange=[m,M])
Colorbar(fig[1, 3], colormap=cmap, colorrange=[m,M])

display(fig)

save("thesis_tests/density_plots/density_error.png", fig)
