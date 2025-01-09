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

H = [0.2 * 2.0^(-i) for i in 2:6]

err0 = Float64[]
err1 = Float64[]
err2 = Float64[]

GEOMETRY = "geometries/8-like_discontinuous.jl"
# k = 10      # number of neighbors for local correction
# α, β = 0, 1 # coefficients for single, double layer
## suggested values are include in geometry files
TESTFILE = "test_files/Integral_discontinuous.jl"

Inti.clear_entities!()
include(GEOMETRY)
include(TESTFILE)

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "h", ylabel = "error", xscale = log10, yscale = log10)
# scatterlines!(ax, H, err0; linewidth = 2, marker = :circle, label = "no correction")
scatterlines!(ax, H, err1; linewidth = 2, marker = :circle, label = " local")
scatterlines!(ax, H, err2; linewidth = 2, marker = :circle, label = "global")

# add some reference slopes
for slope in (qorder-2):(qorder+2)
    ref = err2[end] / H[end]^slope
    lines!(ax, H, ref * H .^ slope; linestyle = :dash, label = "slope $slope")
end
axislegend(; position = :lt)

display(fig)
