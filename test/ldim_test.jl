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
Ω, msh = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.2, order = 1)
# Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius=1.0, meshsize = 0.2)
Γ = Inti.external_boundary(Ω)

Γ_msh = msh[Γ]
Ω_msh = msh[Ω]
test_msh = Γ_msh
nei = Inti.topological_neighbors(test_msh) |> collect
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

function viz_elements_bords(els, bords, msh)
    fig, _, _ = viz(msh; color = 0, showsegments = false,alpha=0.5)
    for (E, i) in els
        el = Inti.elements(msh, E)[i]
        viz!(el; showsegments = false, alpha=0.7)
    end
    viz!(bords;color=4,showsegments = false)
    display(fig)
end

el_in_set(el, set) = any(x->sort(x) == sort(el), set)


I = 5
test_els = copy(nei[I][2])
push!(test_els, nei[I][1])

BD = Inti.boundary1d(test_els, test_msh)

bords = Inti.LagrangeElement[]
for idxs in BD
    vtxs = Inti.nodes(Ω_msh)[idxs]
    bord = Inti.LagrangeLine(vtxs...)
    push!(bords, bord)
end

viz_elements_bords(test_els, bords, test_msh)

for bord in bords
    viz!(bord;color=4)
end
viz(bords)
viz(first(bords))

##

quad = Inti.Quadrature(view(msh, Γ); qorder = 3)
σ = t == :interior ? 1 / 2 : -1 / 2
xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
T = Inti.default_density_eltype(pde)
c = rand(T)
u = (qnode) -> Inti.SingleLayerKernel(pde)(qnode, xs) * c
dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(pde)(qnode, xs) * c
γ₀u = map(u, quad)
γ₁u = map(dudn, quad)
γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
# single and double layer
G = Inti.SingleLayerKernel(pde)
S = Inti.IntegralOperator(G, quad)
Smat = Inti.assemble_matrix(S)
dG = Inti.DoubleLayerKernel(pde)
D = Inti.IntegralOperator(dG, quad)
Dmat = Inti.assemble_matrix(D)
e0 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm

green_multiplier = fill(-0.5, length(quad))
# δS, δD = Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
δS, δD = Inti.local_bdim_correction(pde, quad, quad; green_multiplier)
Sdim = Smat + δS
Ddim = Dmat + δD
# Sdim, Ddim = Inti.single_double_layer(;
#     pde,
#     target      = quad,
#     source      = quad,
#     compression = (method = :none,),
#     correction  = (method = :ldim,),
# )
e1 = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
@show norm(e0, Inf)
@show norm(e1, Inf)
