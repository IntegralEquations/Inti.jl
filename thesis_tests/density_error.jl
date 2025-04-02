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

# separate low-error points from high-error ones
El_low = Float64[]
Eg_low = Float64[]
quad_low = Tuple{Float64, Float64}[]
El_high = Float64[]
Eg_high = Float64[]
quad_high = Tuple{Float64, Float64}[]
for (el, eg, q) in zip(errl, errg, quad) 
    x = Tuple(Inti.coords(q))
    if abs(x[1]) > 0.1
        push!(El_low, el)
        push!(Eg_low, eg)
        push!(quad_low, x)
    else
        push!(El_high, el)
        push!(Eg_high, eg)
        push!(quad_high, x)
    end
end

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
ax3 = Axis(fig[2, 1])
ax4 = Axis(fig[2, 2])
hideydecorations!(ax4)
m, M = min(minimum(errl), minimum(errg)), max(maximum(errl), maximum(errg))

q = quad_low
El = El_low
Eg = Eg_low
cmap = :viridis
# m, M = min(minimum(El), minimum(Eg)), max(maximum(El), maximum(Eg))
scatter!(ax1, q, color=El, colormap=cmap, colorrange=[m,M])
scatter!(ax2, q, color=Eg, colormap=cmap, colorrange=[m,M])
Colorbar(fig[1, 3], colormap=cmap, colorrange=[m,M])

q = quad_high
El = El_high
Eg = Eg_high
cmap = :viridis
# m, M = min(minimum(El), minimum(Eg)), max(maximum(El), maximum(Eg))
scatter!(ax3, q, color=El, colormap=cmap, colorrange=[m,M])
scatter!(ax4, q, color=Eg, colormap=cmap, colorrange=[m,M])
Colorbar(fig[2, 3], colormap=cmap, colorrange=[m,M])

display(fig)

save("thesis_tests/density_plots/density_error.png", fig)
