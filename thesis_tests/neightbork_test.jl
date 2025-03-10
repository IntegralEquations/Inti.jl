using Test
using StaticArrays
using LinearAlgebra
using Inti
using Random
using Meshes
using CairoMakie
using LaTeXStrings
using QuadGK
using ForwardDiff

SAVE = true
TEST_TYPE = "K"

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
# pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
pde = Inti.Stokes(; dim = N, μ = 1.2)
qorder = 5
K = (3, 5, 10)

ii = 1:5
H = [0.1 * 2.0^(-i) for i in ii]

Errl = Dict(k => Float64[] for k in K)
Errg = Float64[]

##
##

GEOMETRY = "geometries/kite.jl"
# k = 10      # number of neighbors for local correction
# α, β = 0, 1 # coefficients for single, double layer
## suggested values are include in geometry files
TESTFILE = "test_files/Integral_Green_identity.jl"

Inti.clear_entities!()
include(GEOMETRY)
include(TESTFILE)

theme = Theme(;
    Axis = (
        xlabel = L"Average mesh size $(h)$",
        ylabel = L"\text{Relative error}",
        xscale = log2,
        yscale = log10,
        xticks = (H, [L"$0.1\times2^{-%$i}$" for i in ii]),
        linewidth = 2,
        # autolimitaspect = 1,
        # aspect = DataAspect(),
    ),
    fontsize = 20,
)
Makie.set_theme!(theme)

##
fig = Figure()
ax = Axis(fig[1, 1])
for (i, k) in enumerate(K)
    # err0 = Err0[qorder]
    errl = Errl[k]
    # scatterlines!(ax, H, err0;colormap=:tab10, colorrange=(1, 10), color=3, marker = :x,    label=qorder == Q[1] ? "no correction" : nothing)
    scatterlines!(ax, H, errl;colormap=Reverse(:viridis), colorrange=(1, 10), color=k-3, marker=:rect, markersize=15, label=L"k=%$k")
end

scatterlines!(ax, H, Errg;colormap=Reverse(:viridis), colorrange=(1, 10), color=8, marker=:circle, markersize=15, label=L"\text{global}")

# add reference slopes
P = div(qorder + 1, 2)
slope = P + 1
ref = 0.8 * Errg[1] / H[1]^slope
lines!(ax, H, ref * H .^ slope;color=:black, linestyle = :dash, label =nothing)
text!(ax, H[2]*1.2, Errg[2], text=L"$P=%$P$";align=(:left, :top))
text!(ax, H[2]*0.99, 0.4*Errg[2], text=L"$\text{slope}=%$slope$";align=(:left, :top))
axislegend(; position = :lt)

display(fig)
##
GEOM = splitdir(GEOMETRY)[2][1:end-3]
TEST = splitdir(TESTFILE)[2][1:end-3]
SAVE && save("thesis_tests/ktest_plots/$(GEOM)_$(TEST).png", fig)