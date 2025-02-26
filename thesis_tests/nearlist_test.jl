using LinearAlgebra
using StaticArrays
using Inti
using Random
using Meshes
using CairoMakie
using Gmsh

theme = Theme(;
    Axis = (
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
        # autolimitaspect = 1,
        aspect = DataAspect(),
    ),
    fontsize = 20,
)
Makie.set_theme!(theme)

include("test_utils.jl")
Random.seed!(1)

h = 0.2

t = :interior
N = 2
GEOMETRY = "geometries/circle_with_narrow_connection.jl"
Inti.clear_entities!()
include(GEOMETRY)

msh = Inti.meshgen(Γ; meshsize = h)
Γ_msh = msh[Γ]
ee = Inti.element_types(Γ_msh)
E = first(ee)
Q = Inti.Quadrature(Γ_msh; qorder = 2)

# hmin = Inf
# for idxs in eachcol(Inti.etype2qtags(Q, E))
#     @show sum(Inti.weight(Q[i]) for i in idxs)
# end

## topological neighbors
viz(Γ_msh; showsegments = true)
topo_nei = Inti.topological_neighbors(Γ_msh, 20)

iel = 18
el = Inti.elements(Γ_msh, E)[iel]

for (E′, i′) in topo_nei[(E, iel)]
    el′ = Inti.elements(Γ_msh, E′)[i′]
    viz!(el′; color = :green, segmentsize = 10)
end

viz!(el; color = :red, segmentsize = 10)

current_figure()

## target_to_near_elements
Q = Inti.Quadrature(Γ_msh; qorder = 2)
X = [q.coords for q in Q]
maxdist = 2 * h
dict = Inti.target_to_near_elements(X, Γ_msh; tol = maxdist)

viz(Γ_msh; showsegments = true)

itarget = 30

for (E′, i′) in dict[itarget]
    el′ = Inti.elements(Γ_msh, E′)[i′]
    viz!(el′; color = :green, segmentsize = 10)
end
scatter!(map(Inti.center, Inti.elements(Γ_msh, E)); color = :blue, markersize = 5)
scatter!(X[itarget]; color = :red, markersize = 10)
arc!(X[itarget], maxdist, 0, 2π)

current_figure()

## connected components

Q = Inti.Quadrature(Γ_msh; qorder = 2)
X = [q.coords for q in Q]
maxdist = 2 * h
dict = Inti.target_to_near_elements(X, Γ_msh; tol = maxdist)

viz(Γ_msh; showsegments = true)

itarget = 30

topo_nei = Inti.topological_neighbors(Γ_msh, 1)

colors = (:red, :green, :blue)
for (k, comp) in enumerate(Inti.connected_components(dict[itarget], topo_nei))
    bnd = Inti.boundary(comp, Γ_msh)
    for (E′, i′) in comp
        el′ = Inti.elements(Γ_msh, E′)[i′]
        viz!(el′; color = colors[k], segmentsize = 10)
    end
    scatter!(bnd; color = :yellow, markersize = 20, marker = :ltriangle)
end

scatter!(map(Inti.center, Inti.elements(Γ_msh, E)); color = :blue, markersize = 5)
scatter!(X[itarget]; color = :black, markersize = 10)
arc!(X[itarget], maxdist, 0, 2π)

current_figure()

## nearest element in connected component

Q = Inti.Quadrature(Γ_msh; qorder = 2)
X = [q.coords for q in Q]
maxdist = 2 * h

viz(Γ_msh; showsegments = true)

itarget = 30

near_els = Inti.nearest_element_in_connected_components(X, Γ_msh; maxdist)

colors = (:red, :green, :blue)
for k in 1:length(near_els[itarget])
    (E′, i′) = near_els[itarget][k]
    el′ = Inti.elements(Γ_msh, E′)[i′]
    viz!(el′; color = colors[k], segmentsize = 10)
end

scatter!(map(Inti.center, Inti.elements(Γ_msh, E)); color = :blue, markersize = 5)
scatter!(X[itarget]; color = :black, markersize = 10)
arc!(X[itarget], maxdist, 0, 2π)

current_figure()

## local_bdim_element_to_target
Q = Inti.Quadrature(Γ_msh; qorder = 2)
X = [q.coords for q in Q]
maxdist = 3 * h

viz(Γ_msh; showsegments = true)

e2t = Inti.local_bdim_element_to_target(X, Γ_msh; maxdist)

etag = 15
colors = (:red, :green, :blue)
el = Inti.elements(Γ_msh, E)[etag]
viz!(el; color = :red, segmentsize = 10)
for i in e2t[E][etag]
    scatter!(X[i]; color = :black, markersize = 10)
end

current_figure()

## boundary

topo_nei = Inti.topological_neighbors(Γ_msh, 3)

etag = 3
nei = topo_nei[(E, etag)]
maxdist = 3 * h
viz(Γ_msh; showsegments = true)

for (E′, i′) in nei
    el′ = Inti.elements(Γ_msh, E′)[i′]
    viz!(el′; color = :green, segmentsize = 10)
end
viz!(Inti.elements(Γ_msh, E)[etag]; color = :red, segmentsize = 10)

bnd = Inti.boundary(nei, Γ_msh)

scatter!(bnd; color = :black, markersize = 10)

current_figure()

## auxiliary domain

topo_nei = Inti.topological_neighbors(Γ_msh, 3)

etag = 3
el = Inti.elements(Γ_msh, E)[etag]
nei = topo_nei[(E, etag)]
maxdist = 3 * h
viz(Γ_msh; showsegments = true)
ν = Inti.normal(el, 0.5)
# aux_els = Inti.local_bdim_auxiliary_domain(nei, Γ_msh, ν, 2 * h)
x = el(0.5) - 0.03 * ν
aux_els, orientation = Inti.local_bdim_auxiliary_els(nei, Γ_msh, Q, x)
for (E′, i′) in nei
    el′ = Inti.elements(Γ_msh, E′)[i′]
    viz!(el′; color = :green, segmentsize = 10)
end
for el′ in aux_els
    viz!(el′; color = :blue, segmentsize = 10)
end
viz!(el; color = :red, segmentsize = 10)

aux_quad = Inti.local_bdim_auxiliary_quadrature(aux_els, 4)

scatter!([q.coords for q in aux_quad]; color = :black, markersize = 10)
scatter!(x; color = :red, markersize = 10)

current_figure()
