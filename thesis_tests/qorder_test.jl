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

SAVE = false
TEST_TYPE = "QORDER"

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
# pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
pde = Inti.Stokes(; dim = N, μ = 1.2)
Q = (5, 7)

ii = 1:5
H = [0.2 * 2.0^(-i) for i in ii]

Err0 = Dict(qorder => Float64[] for qorder in Q)
Errl = Dict(qorder => Float64[] for qorder in Q)
Errg = Dict(qorder => Float64[] for qorder in Q)

##
##

GEOMETRY = "geometries/8-like_discontinuous.jl"
# k = 10      # number of neighbors for local correction
# α, β = 0, 1 # coefficients for single, double layer
## suggested values are include in geometry files
TESTFILE = "test_files/Integral_discontinuous.jl"

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
for qorder in Q
    # err0 = Err0[qorder]
    errl = Errl[qorder]
    errg = Errg[qorder]
    P = div(qorder + 1, 2)
    # scatterlines!(ax, H, err0;colormap=:tab10, colorrange=(1, 10), color=3, marker = :x,    label=qorder == Q[1] ? "no correction" : nothing)
    scatterlines!(ax, H, errl;colormap=Reverse(:viridis), colorrange=(1, 10), color=P-1, marker=:rect, markersize=15, label=L"\text{  local }P=%$P")
    scatterlines!(ax, H, errg;colormap=Reverse(:viridis), colorrange=(1, 10), color=qorder+3, marker=:circle, markersize=15, label=L"\text{global }P=%$P")

    # add reference slopes
    slope = 1
    ref = 0.8 * errl[2] / H[2]^slope
    lines!(ax, H, ref * H .^ slope;color=:black, linestyle = :dash, label =nothing)
    # text!(ax, H[2]*1.2, 0.4*errl[2], text=L"$P=%$P$";align=(:left, :top))
    text!(ax, H[2]*0.99, 0.4*errl[2], text=L"$\text{slope}=%$slope$";align=(:left, :top))
end
axislegend(; position = :lt)

display(fig)
##
GEOM = splitdir(GEOMETRY)[2][1:end-3]
TEST = splitdir(TESTFILE)[2][1:end-3]
SAVE && save("thesis_tests/plots/$(GEOM)_$(TEST).png", fig)