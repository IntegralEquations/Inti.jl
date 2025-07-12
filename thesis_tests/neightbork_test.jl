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
using LinearMaps
using IterativeSolvers

SAVE = true
TEST_TYPE = "K"

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
# pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
pde = Inti.Stokes(; dim = N, μ = 1.2)

K = (1, 3, 5)
qorders = [3 5 7]

fig = Figure(; size = (600, 600))
layout = [(1, 1), (2, 1), (3, 1)]
# layout = [(1, 1)]
ii = 1:5
H = [0.5 * 2.0^(-i) for i in ii]

for (cc, qq) in enumerate(qorders)
    global qorder = qq
    global Errl = Dict(k => Float64[] for k in K)
    global Errg = Float64[]
    global GEOMETRY = "geometries/kite.jl"
    # k = 10      # number of neighbors for local correction
    # α, β = 0, 1 # coefficients for single, double layer
    ## suggested values are include in geometry files
    global TESTFILE = "test_files/Integral_Green_identity.jl"

    Inti.clear_entities!()
    include(GEOMETRY)
    include(TESTFILE)

    ##
    lastplot = cc == length(layout)
    ax = Axis(
        fig[layout[cc]...];
        xlabel = lastplot ? L"\text{Mesh size} $h$" : "",
        ylabel = L"\text{Relative error}",
        xticks = (H, [L"$2^{-%$(i+1)}$" for i in ii]),
        yscale = log10,
        xscale = log2,
    )
    lastplot || (ax.xticklabelsvisible = false)
    for (i, k) in enumerate(K)
        # err0 = Err0[qorder]
        errl = Errl[k]
        # scatterlines!(ax, H, err0;colormap=:tab10, colorrange=(1, 10), color=3, marker = :x,    label=qorder == Q[1] ? "no correction" : nothing)
        scatterlines!(
            ax,
            H,
            errl;
            colormap = Reverse(:viridis),
            colorrange = (1, 10),
            color = k - 3,
            marker = :rect,
            markersize = 15,
            label = L"k=%$k",
        )
    end

    scatterlines!(
        ax,
        H,
        Errg;
        colormap = Reverse(:viridis),
        colorrange = (1, 10),
        color = 8,
        marker = :circle,
        markersize = 15,
        label = L"\text{global}",
    )

    # add reference slopes
    global P = div(qorder + 1, 2)
    slope = P + 1
    ref = 0.8 * Errg[1] / H[1]^slope
    lines!(ax, H, ref * H .^ slope; color = :black, linestyle = :dash, label = nothing)
    text!(
        ax,
        H[2] * 1.2,
        Errg[2];
        text = L"     $P=%$P$ \n slope=%$slope",
        align = (:left, :top),
    )
end

leg = Legend(fig[0, 1], current_axis(); orientation = :horizontal)

display(fig)
##
GEOM = splitdir(GEOMETRY)[2][1:end-3]
TEST = splitdir(TESTFILE)[2][1:end-3]
# SAVE && save("thesis_tests/ktest_plots/$(GEOM)_$(TEST)_P_$(P).png", fig)
