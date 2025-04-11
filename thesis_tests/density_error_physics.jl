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

u = x -> cos(x.coords[1]) * exp(x.coords[2])
du= x -> SVector(-sin(x.coords[1])*exp(x.coords[2]), cos(x.coords[1])*exp(x.coords[2])) ⋅ x.normal

##
qorder, h, k = 5, 2.0^(-5), 10
msh = Inti.meshgen(Γ; meshsize = h)
Γ_msh = msh[Γ]
quad = Inti.Quadrature(Γ_msh; qorder)
ubnd = map( u, quad)
dubnd= map(du, quad)
# single and double layer
S = Inti.IntegralOperator(G, quad)
Smat  = Inti.assemble_matrix(S)
D = Inti.IntegralOperator(dG, quad)
Dmat  = Inti.assemble_matrix(D)
green_multiplier = fill(-0.5, length(quad))

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
dul  = Sdim \ (ubnd + (Ddim + μ*I) * ubnd)
errl = abs.(dul - dubnd)

tdim = @elapsed δS, δD =
    Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
Sdim = Smat + δS
Ddim = Dmat + δD
dug   = Sdim \ (ubnd + (Ddim + μ*I) * ubnd)
errg = abs.(dug - dubnd)

# normalize data
q = map(Inti.coords, quad)
Merr = max(maximum(errl), maximum(errg))
errl = log10.(errl ./ Merr)
errg = log10.(errg ./ Merr)

theme = Theme(;
    Axis = (     
        autolimitaspect = 1,
        # aspect = DataAspect(),
    ),
    fontsize = 20,
)
Makie.set_theme!(theme)

fig = Figure(size=(1400,1000))
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
hideydecorations!(ax2)
m, M = min(minimum(errl), minimum(errg)), max(maximum(errl), maximum(errg))

cmap = :viridis
# m, M = min(minimum(El), minimum(Eg)), max(maximum(El), maximum(Eg))
scatter!(ax1, q, color=errl, colormap=cmap, colorrange=[m,M])
scatter!(ax2, q, color=errg, colormap=cmap, colorrange=[m,M])
Colorbar(fig[1, 3], colormap=cmap, colorrange=[m,M])

display(fig)

# save("thesis_tests/density_plots/density_error.png", fig)
