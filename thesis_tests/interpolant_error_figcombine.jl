using Test
using StaticArrays
using LinearAlgebra
using Inti
using Random
using Meshes
using CairoMakie

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

ii = 3:7; H = [2.0^(-i) for i in ii]

##
##

GEOMETRY = "geometries/8-like.jl"
# k = 10      # number of neighbors for local correction
## suggested values are include in geometry files

SAVE = true
Inti.clear_entities!()
include(GEOMETRY)
xt = χ((3a+b)/4)
# xt = χ(0)
theme = Theme(;
    Axis = (
        xlabel = L"Average mesh size $(h)$",
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
fig = Figure(size=(700,1000))
row = 0
for (α, β) in ((0, 1), (1, 0)) # coefficients for basis of interpolants
for Dirichlet in (true, false) # type of the density to be interpolated
row += 1
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

Err = Dict(qorder => Float64[] for qorder in Q)
Err_oppo = Dict(qorder => Float64[] for qorder in Q)
for qorder in Q
    for h in H
        P = div(qorder + 1, 2)
        quad = get_quad(Γ, h, qorder)
        source = quad
        # get the index of the element containing xt
        dict_near = Inti.etype_to_nearest_points([xt], quad; maxdist=Inf)
        Etype = first(keys(dict_near))
        list_near = dict_near[Etype]
        i = 0
        for j in eachindex(list_near)
            if !isempty(list_near[j])
                i = j
                break
            end
        end
        # create the source points
        parameters = Inti.DimParameters()
        qmax = sum(size(mat, 1) for mat in values(source.etype2qtags)) # max number of qnodes per el
        ns   = ceil(Int, parameters.sources_oversample_factor * qmax)
        # compute a bounding box for source points
        low_corner = reduce((p, q) -> min.(Inti.coords(p), Inti.coords(q)), source)
        high_corner = reduce((p, q) -> max.(Inti.coords(p), Inti.coords(q)), source)
        xc = (low_corner + high_corner) / 2
        R = parameters.sources_radius_multiplier * norm(high_corner - low_corner) / 2
        Y = Inti.uniform_points_circle(ns, R, xc)
        # create the interpolant
        M, N = P, ns
        X = [quad[j] for j in P*(i-1)+1:P*i]
        A = Matrix{Float64}(undef, 2*M, N)
        for i in 1:M 
            for j in 1:N 
                A[i,j]   =  G(Y[j], X[i])
                A[i+M,j] = dG(Y[j], X[i])
            end
        end
        b = append!([α*u(x) for x in X], [β*u(x) for x in X])
        c = A \ b
        # calculate the error
        push!(Err[qorder], abs([G(xt, y) for y in Y] ⋅ c - α*u((coords=xt,normal=SVector(0,-1)))))
        push!(Err_oppo[qorder], abs([G(-xt, y) for y in Y] ⋅ c - α*u((coords=-xt,normal=SVector(0,-1)))))   
    end    
end

j = α == 1 ? 0 : 1
i = Dirichlet ? 0 : 1
ax = Axis(fig[row, 1], ylabel=L"|\gamma_%$j\sigma^{h,p}_{\mathbf{x}_t}-\gamma_%$i\sigma|")
if row != 4
    hidexdecorations!(ax)    
end
for q in Q
    P = div(q + 1, 2)
    scatterlines!(ax, H, Err[q];colormap=Reverse(:viridis), colorrange=(1, 10), color=P, marker=:rect, markersize=15, label=L"P=%$P,\text{ target's side}")
    scatterlines!(ax, H, Err_oppo[q];colormap=Reverse(:viridis), colorrange=(1, 10), color=5+q, marker=:circle, markersize=15, label=L"P=%$P,\text{ opposite side}")
end
# axislegend(ax; position=:lt)
fig[row,2] = Legend(fig, ax, framevisible=false, valign=:top)

params = [(2, Err[3], 3, 5, 0.99, 0.6),
          (4, Err[5], 3, 3, 0.99, 0.6),
         ]
for (slope, err, i, ti, tx, ty) in params
    ref = 0.7 * err[i] / H[i]^slope
    lines!(ax, H, ref * H .^ slope;color=:black, linestyle = :dash, label =nothing)
    # text!(ax, H[2]*1.2, 0.4*errl[2], text=L"$P=%$P$";align=(:left, :top))
    text!(ax, H[ti]*tx, ty*err[ti], text=L"$\text{slope}=%$slope$";align=(:left, :top))
end
end
end

display(fig)
##
GEOM = splitdir(GEOMETRY)[2][1:end-3]
# TEST = splitdir(TESTFILE)[2][1:end-3]
SAVE && save("thesis_tests/interpolant_error/$(GEOM)_interpolant_error.png", fig)