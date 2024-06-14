using Test
using LinearAlgebra
using Inti
using Random
using Meshes
using GLMakie

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
pde = Inti.Laplace(; dim = N)
Inti.clear_entities!()
Ω, msh = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.2, order = 2)
# Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius=1.0, meshsize = 0.2)
Γ = Inti.external_boundary(Ω)

k = 2
Γ_msh = msh[Γ]
Ω_msh = msh[Ω]
test_msh = Ω_msh
nei = Inti.topological_neighbors(test_msh, 1)
E = first(Inti.element_types(test_msh))
function viz_neighbors(i, msh)
    k, v = nei[i]
    E, idx = k
    el = Inti.elements(msh, E)[idx]
    fig, _, _ = viz(el; color = 0, showsegments = true)
    for (E, i) in v
        el = Inti.elements(msh, E)[i]
        viz!(el; color = 1 / 2, showsegments = true, alpha = 0.2)
    end
    return display(fig)
end

function viz_elements(els, msh)
    Els = [Inti.elements(msh, E)[i] for (E, i) in els]
    fig, _, _ = viz(Els)
    viz!(msh; color = 0, showsegments = true,alpha=0.3)
    display(fig)
end

function viz_elements_bords(Ei, els, bords, msh)
    el = Inti.elements(msh, Ei[1])[Ei[2]]
    fig, _, _ = viz(msh; color = 0, showsegments = true,alpha=0.3)
    viz!(el; color = 0, showsegments = true,alpha=0.5)
    for (E, i) in els
        el = Inti.elements(msh, E)[i]
        viz!(el; showsegments = true, alpha=0.7)
    end
    viz!(bords;color=4,showsegments = false,segmentsize=5,segmentcolor=4)
    display(fig)
end

# el_in_set(el, set) = any(x->sort(x) == sort(el), set)

I, J = 1, 3
test_els = union(copy(nei[(E,1)]), nei[(E,2)], nei[(E,3)], nei[(E,4)])
viz_elements(test_els, test_msh)

components = Inti.connected_components(test_els, nei)

BD = Inti.boundarynd(test_els, test_msh)
# bords = [Inti.nodes(test_msh)[abs(i)] for i in BD]

bords = Inti.LagrangeElement[]
for idxs in BD
    vtxs = Inti.nodes(Ω_msh)[idxs]
    bord = Inti.LagrangeLine(vtxs...)
    push!(bords, bord)
end

viz_elements_bords(nei[I][1], test_els, bords, test_msh)
