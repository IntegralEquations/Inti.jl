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

SAVE = false
TEST_TYPE = "QORDER"

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
# pde = Inti.Stokes(; dim = N, μ = 1.2)
G  = Inti.SingleLayerKernel(pde)
dG = Inti.DoubleLayerKernel(pde)
Q = (3, 5)

ii = 2:6
H = [2.0^(-i) for i in ii]

Err0 = Dict(qorder => Float64[] for qorder in Q)
Errl = Dict(qorder => Float64[] for qorder in Q)
Errg = Dict(qorder => Float64[] for qorder in Q)
##
##

GEOMETRY = "geometries/8-like.jl"
# k = 10      # number of neighbors for local correction
## suggested values are include in geometry files
TESTFILE = "test_files/auto_converge_Integral_discontinuous.jl"

Inti.clear_entities!()
include(GEOMETRY)
xt = χ((3a+b)/4)
# xt = χ(0)
k = 10
α, β = 1, 0 # coefficients for single, double layer
onSurf = true
include(TESTFILE)

theme = Theme(;
    Axis = (
        xlabel = L"Average mesh size $(h)$",
        ylabel = L"\text{Relative error}",
        xscale = log2,
        yscale = log10,
        xticks = (H, [L"$2^{-%$i}$" for i in ii]),
        linewidth = 2,
        
        # autolimitaspect = 1,
        # aspect = DataAspect(),
    ),
    fontsize = 20,
)
Makie.set_theme!(theme)

##
fig = Figure(size=(700,400))
ax = Axis(fig[1, 1])
for qorder in Q
    # err0 = Err0[qorder]
    errl = Errl[qorder]
    errg = Errg[qorder]
    P = div(qorder + 1, 2)
    # scatterlines!(ax, H, err0;colormap=:tab10, colorrange=(1, 10), color=3, marker = :x,    label=qorder == Q[1] ? "no correction" : nothing)
    scatterlines!(ax, H, errl;colormap=Reverse(:viridis), colorrange=(1, 10), color=P-1, marker=:rect, markersize=15, label=L"\text{  local }P=%$P")
    scatterlines!(ax, H, errg;colormap=Reverse(:viridis), colorrange=(1, 10), color=qorder+3, marker=:circle, markersize=15, label=L"\text{global }P=%$P")
end
# add reference slopes
params = [(0, Errg[5], 2, 0.99, 0.6),
          (1, Errl[3], 3, 0.99, 10),
          (3, Errl[5], 5, 0.99, 0.6)]
for (slope, err, i, tx, ty) in params
    ref = 0.7 * err[i] / H[i]^slope
    lines!(ax, H, ref * H .^ slope;color=:black, linestyle = :dash, label =nothing)
    # text!(ax, H[2]*1.2, 0.4*errl[2], text=L"$P=%$P$";align=(:left, :top))
    text!(ax, H[4]*tx, ty*err[4], text=L"$\text{slope}=%$slope$";align=(:left, :top))
end
fig[1,2] = Legend(fig, ax, framevisible=false, valign=:top)

display(fig)
##
GEOM = splitdir(GEOMETRY)[2][1:end-3]
TEST = splitdir(TESTFILE)[2][1:end-3]
SorD = α == 1 ? "single" : "double"
surf = onSurf ? "onsurface" : "offsurface"
SAVE && save("thesis_tests/plots/$(GEOM)_$(TEST)_$(SorD)_$(surf).png", fig)