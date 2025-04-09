using Test
using StaticArrays
using LinearAlgebra
using Inti
using Random
using Meshes
using CairoMakie
using QuadGK
using ForwardDiff
using LinearMaps
using IterativeSolvers

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

ii = 3:7; Hi = [2.0^(-i) for i in ii]
jj = 0:3; Hj = [2.0^(j)*1e-4 for j in jj]
H = union(Hi, Hj)

Err0 = Dict(qorder => Float64[] for qorder in Q)
Errl = Dict(qorder => Float64[] for qorder in Q)
Errg = Dict(qorder => Float64[] for qorder in Q)
##
##

GEOMETRY = "geometries/8-like.jl"
# k = 10      # number of neighbors for local correction
## suggested values are include in geometry files
TESTFILE = "test_files/auto_converge_Integral_discontinuous.jl"

SAVE = true
Inti.clear_entities!()
include(GEOMETRY)
xt = χ((3a+b)/4)
# xt = χ(0)
k = 10
α, β = 0, 1 # coefficients for single, double layer
Dirichlet = false
if Dirichlet
    u = x -> cos(x.coords[1]) * exp(x.coords[2])
else
    u = x -> SVector(-sin(x.coords[1])*exp(x.coords[2]), cos(x.coords[1])*exp(x.coords[2])) ⋅ x.normal
end
# if Dirichlet
#     u = x -> cos(x.coords[1])
# else
#     u = x -> SVector(-sin(x.coords[1]), 0) ⋅ x.normal
# end
include(TESTFILE)

theme = Theme(;
    Axis = (
        # xlabel = L"Average mesh size $(h)$",
        xscale = log2,
        yscale = log10,
        linewidth = 2,
        # autolimitaspect = 1,
        # aspect = DataAspect(),
    ),
    fontsize = 20,
)
Makie.set_theme!(theme)

##
fig = Figure(size=(1400,400))
ax1 = Axis(fig[1, 2], xticks=(H[1:5], [Lpower(2, -i) for i in ii]))
ax2 = Axis(fig[1, 1], xticks=(H[6:9], [L"10^{-4}", L"2\times10^{-4}", L"2^{2}\times10^{-4}", L"2^{3}\times10^{-4}"]))
# , ylabel = L"\text{Relative error}"
# hidexdecorations!(ax1)
# hidexdecorations!(ax2)

for qorder in Q
    P = div(qorder + 1, 2)
    # coarse mesh
    rge = 1:5
    errl = Errl[qorder][rge]
    errg = Errg[qorder][rge]
    scatterlines!(ax1, H[rge], errl;colormap=Reverse(:viridis), colorrange=(1, 10), color=P-1, marker=:rect, markersize=15, label=L"\text{  local }P=%$P")
    scatterlines!(ax1, H[rge], errg;colormap=Reverse(:viridis), colorrange=(1, 10), color=qorder+3, marker=:circle, markersize=15, label=L"\text{global }P=%$P")
    # fine mesh
    rge = 6:9
    errl = Errl[qorder][rge]
    errg = Errg[qorder][rge]
    scatterlines!(ax2, H[rge], errl;colormap=Reverse(:viridis), colorrange=(1, 10), color=P-1, marker=:rect, markersize=15, label=L"\text{  local }P=%$P")
    scatterlines!(ax2, H[rge], errg;colormap=Reverse(:viridis), colorrange=(1, 10), color=qorder+3, marker=:circle, markersize=15, label=L"\text{global }P=%$P")
end
# add reference slopes
# coarse
params = [(1, Errg[5][1:5], 2, 0.99, 0.6),
          (2, Errl[3][1:5], 2, 0.99, 0.4),
          (4, Errl[5][1:5], 2, 0.99, 0.4)]
for (slope, err, i, tx, ty) in params
    ref = 0.7 * err[i] / Hi[i]^slope
    lines!(ax1, Hi, ref * Hi .^ slope;color=:black, linestyle = :dash, label =nothing)
    # text!(ax, H[2]*1.2, 0.4*errl[2], text=L"$P=%$P$";align=(:left, :top))
    text!(ax1, Hi[3]*tx, ty*err[3], text=L"$\text{slope}=%$slope$";align=(:left, :top))
end
# fine
# params = [
#           (3, Errg[3][5:8], 2, 0.99, 1.4),
#           (4, Errg[5][5:8], 2, 0.99, 0.4)]
# for (slope, err, i, tx, ty) in params
#     ref = 0.7 * err[i] / Hj[i]^slope
#     lines!(ax2, Hj, ref * Hj .^ slope;color=:black, linestyle = :dash, label =nothing)
#     # text!(ax, H[2]*1.2, 0.4*errl[2], text=L"$P=%$P$";align=(:left, :top))
#     text!(ax2, Hj[2]*tx, ty*err[2], text=L"$\text{slope}=%$slope$";align=(:left, :top))
# end
axislegend(ax2; position=:lt)
# fig[1,3] = Legend(fig, ax1, framevisible=false, valign=:top)

display(fig)
##
GEOM = splitdir(GEOMETRY)[2][1:end-3]
# TEST = splitdir(TESTFILE)[2][1:end-3]
SorD = α == 1 ? "single" : "double"
DorN = Dirichlet ? "Dirichlet" : "Neumann"
SAVE && save("thesis_tests/integral_test_plots/$(GEOM)_$(SorD)_$(DorN).png", fig)