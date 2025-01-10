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

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
pde = Inti.Laplace(; dim = N)
# pde = Inti.Helmholtz(; k = 2.1, dim = N)
# pde = Inti.Stokes(; dim = 2, μ = 1.0)
Q = (3, 5)

ii = 1:5
H = [0.1 * 2.0^(-i) for i in ii]

Err0 = Dict(qorder => Float64[] for qorder in Q)
Err1 = Dict(qorder => Float64[] for qorder in Q)
Err2 = Dict(qorder => Float64[] for qorder in Q)

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
        xlabel = L"Average mesh size $(h\sim N^{-1})$",
        ylabel = "Relative error",
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

fig = Figure()
ax = Axis(fig[1, 1])

##
for qorder in Q
    # err0 = Err0[qorder]
    err1 = Err1[qorder]
    err2 = Err2[qorder]
    # scatterlines!(ax, H, err0; marker = :x, label = "no correction")
    scatterlines!(ax, H, err2; marker = :circle, label = "global")
    scatterlines!(ax, H, err1; marker = :rect,   label = "local")

    # add some reference slopes
    P = div(qorder + 1, 2)
    slope = P + 1
    ref = err2[end] / H[end]^slope
    lines!(ax, H, ref * H .^ slope; linestyle = :dash, label = "slope $slope")
end
axislegend(; position = :lt)

display(fig)
