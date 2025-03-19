using Test
using Inti
using FMM3D
using Gmsh
using LinearAlgebra
using Random
using StaticArrays
using NearestNeighbors
using ForwardDiff

include("test_utils.jl")

# create a boundary and area meshes and quadrature only once
meshsize = .1
qorder = 2

Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = meshsize)
Γ = Inti.external_boundary(Ω)
Γ_msh = view(msh, Γ)

θ = LinRange(0, π, 50*Int(1/meshsize))
ϕ = LinRange(0, 2*π, 50*Int(1/meshsize))
# v = (θ, ϕ)
ψ = (v) -> [sin(v[1]) * cos(v[2]), sin(v[1]) * sin(v[2]), cos(v[1])]
param_disc = Array{SVector{3,Float64}}(undef, length(θ)*length(ϕ))
cart_idxs_θ = []
cart_idxs_ϕ = []
for i in eachindex(θ)
    for j in eachindex(ϕ)
        param_disc[(i-1)*length(ϕ) + j] = [k for k in ψ((θ[i], ϕ[j]))]
        push!(cart_idxs_θ, i)
        push!(cart_idxs_ϕ, j)
    end
end
kdt = KDTree(param_disc; reorder = false)

n2e = Inti.node2etags(msh)

nbdry_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}])[2]
bdry_node_idx = Vector{Int64}()
bdry_node_param_loc = Vector{Vector{Float64}}()

uniqueidx(v) = unique(i -> v[i], eachindex(v))

for elind = 1:nbdry_els
    node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}][:, elind]
    nodes = msh.nodes[node_indices]
    idxs, dists = nn(kdt, nodes)
    msh.nodes[node_indices[1]] = param_disc[idxs[1]]
    msh.nodes[node_indices[2]] = param_disc[idxs[2]]
    msh.nodes[node_indices[3]] = param_disc[idxs[3]]
    push!(bdry_node_idx, node_indices[1])
    push!(bdry_node_idx, node_indices[2])
    push!(bdry_node_idx, node_indices[3])
    # FIXME: θ and ϕ indexing is transposed
    push!(bdry_node_param_loc, Vector{Float64}([θ[cart_idxs_θ[idxs[1]]], ϕ[cart_idxs_ϕ[idxs[1]]]]))
    push!(bdry_node_param_loc, Vector{Float64}([θ[cart_idxs_θ[idxs[2]]], ϕ[cart_idxs_ϕ[idxs[2]]]]))
    push!(bdry_node_param_loc, Vector{Float64}([θ[cart_idxs_θ[idxs[3]]], ϕ[cart_idxs_ϕ[idxs[3]]]]))
end
I = uniqueidx(bdry_node_idx)
bdry_node_idx = bdry_node_idx[I]
bdry_node_param_loc = bdry_node_param_loc[I]
node_to_param = Dict(zip(bdry_node_idx, bdry_node_param_loc))

nvol_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}}])[2]
spharea = 0.0
els = []
#for elind = 1:nvol_els
#elind = 1 # 0 verts on bdry meshsize .1
elind = 20402 # 2 verts on bdry meshsize .1
#elind = 20386 # 3 verts on bdry meshsize .1
#elind = 20396
#elind = 152422 # 3 verts on bdry meshsize .1/2
    node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}}][:, elind]
    nodes = msh.nodes[node_indices]
    
    verts_on_bdry = findall(x -> x ∈ bdry_node_idx, node_indices)
    j = length(verts_on_bdry)
    if j > 1
        println(length(verts_on_bdry))
        println(elind)
        node_indices_on_bdry = node_indices[verts_on_bdry]
        α₁ = node_to_param[node_indices_on_bdry[1]]
        α₂ = node_to_param[node_indices_on_bdry[2]]
        if length(verts_on_bdry) > 2
            α₃ = node_to_param[node_indices_on_bdry[3]]
        else
            # Find missing node α₃ that (non-uniquely) defines the curved face simplex containing α₁, α₂
            candidate_els = Inti.elements_containing_nodes(n2e, node_indices_on_bdry)
            # Filter out volume elements
            candidate_els = candidate_els[length.(candidate_els).==3]
            α₃ = node_to_param[candidate_els[1][3]] #FIXME
        end
        a₁ = ψ(α₁)
        a₂ = ψ(α₂)
        a₃ = ψ(α₃)
        α₁hat = SVector{2,Float64}(0.0, 1.0)
        α₂hat = SVector{2,Float64}(0.0, 0.0)
        α₃hat = SVector{2,Float64}(1.0, 0.0)
        κ = Inti.LagrangeElement{Inti.ReferenceTriangle,3,SVector{2,Float64}}(SVector{3,SVector{2,Float64}}(α₁, α₂, α₃))
        f̂ₖ = (x) -> [(α₃[1] - α₂[1])*x[1] + (α₁[1] - α₂[1])*x[2] + α₂[1], (α₃[2] - α₂[2])*x[1] + (α₁[2] - α₂[2])*x[2] + α₂[2]]
        @assert (f̂ₖ(α₁hat) ≈ α₁) && (f̂ₖ(α₂hat) ≈ α₂) && (f̂ₖ(α₃hat) ≈ α₃)

        # l = 1
        πₖ¹_nodes = Inti.reference_nodes(typeof(κ))
        πₖ¹ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ¹_nodes))
        for i in eachindex(πₖ¹_nodes)
            πₖ¹ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ¹_nodes[i]))
            #πₖ¹ψ_reference_nodes[i] = ψ(f̂ₖ((πₖ¹_nodes[i][1], πₖ¹_nodes[i][2])))
            #πₖ¹ψ_reference_nodes[i] = ψ((f̂ₖ((πₖ¹_nodes[i][1], πₖ¹_nodes[i][2]))[1], f̂ₖ((πₖ¹_nodes[i][1], πₖ¹_nodes[i][2]))[2]))
        end
        πₖ¹ψ = (x) -> κ(x; f = πₖ¹ψ_reference_nodes)
        #l = 2
        # ...

        # l = 1
        # Nonlinear map
        # FIXME need to sum through j for j < d
        if j == 3
            f̂ₖ_comp = (x) -> f̂ₖ( (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat)/(x[1] + x[2] + x[3]) )
            Φₖ = (x) -> ( (x[1] + x[2] + x[3])^3 * (ψ(f̂ₖ_comp(x)) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])))) 
        else
            f̂ₖ_comp = (x) -> f̂ₖ( (x[1] * α₁hat + x[2] * α₂hat)/(x[1] + x[2]) )
            Φₖ = (x) -> ( (x[1] + x[2])^3 * (ψ(f̂ₖ_comp(x)) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])))) 
        end
        # Affine map
        # FIXME: set up aₖ ... so that they match x_1, x_2, x_3 in above map
        # should have bₖ be the node that doesn't match a₁, a₂, a₃.
        # works for j=3
        aₖ = nodes[2]
        bₖ = nodes[1]
        cₖ = nodes[3]
        dₖ = nodes[4]
        F̃ₖ = (x) -> [(aₖ[1] - bₖ[1])*x[1] + (cₖ[1] - bₖ[1])*x[2] + (dₖ[1] - bₖ[1])*x[3] + bₖ[1], (aₖ[2] - bₖ[2])*x[1] + (cₖ[2] - bₖ[2])*x[2] + (dₖ[2] - bₖ[2])*x[3] + bₖ[2], (aₖ[3] - bₖ[3])*x[1] + (cₖ[3] - bₖ[3])*x[2] + (dₖ[3] - bₖ[3])*x[3] + bₖ[3]]

        # Full transformation
        Fₖ = (x) -> F̃ₖ(x) + Φₖ(x)
        JF̃ₖ = (x) -> [aₖ[1] - bₖ[1]; cₖ[1] - bₖ[1]; dₖ[1] - bₖ[1];; aₖ[2] - bₖ[2]; cₖ[2] - bₖ[2]; dₖ[2] - bₖ[2];; aₖ[3] - bₖ[3]; cₖ[3] - bₖ[3]; dₖ[3] - bₖ[3]]
        Jₖ = (x) -> JF̃ₖ(x) + ForwardDiff.jacobian(Φₖ, x)
    else
        aₖ = nodes[1]
        bₖ = nodes[2]
        cₖ = nodes[3]
        dₖ = nodes[4]
        Fₖ = (x) -> [(aₖ[1] - bₖ[1])*x[1] + (cₖ[1] - bₖ[1])*x[2] + (dₖ[1] - bₖ[1])*x[3] + bₖ[1], (aₖ[2] - bₖ[2])*x[1] + (cₖ[2] - bₖ[2])*x[2] + (dₖ[2] - bₖ[2])*x[3] + bₖ[2], (aₖ[3] - bₖ[3])*x[1] + (cₖ[3] - bₖ[3])*x[2] + (dₖ[3] - bₖ[3])*x[3] + bₖ[3]]
        Jₖ = (x) -> [aₖ[1] - bₖ[1]; cₖ[1] - bₖ[1]; dₖ[1] - bₖ[1];; aₖ[2] - bₖ[2]; cₖ[2] - bₖ[2]; dₖ[2] - bₖ[2];; aₖ[3] - bₖ[3]; cₖ[3] - bₖ[3]; dₖ[3] - bₖ[3]]
    end
    
    Q = Inti.VioreanuRokhlin(; domain=Inti.ReferenceTetrahedron(), order=qorder)()
    nq = length(Q[2])
    for q in 1:nq
        global spharea
        spharea += Q[2][q] * abs(det(Jₖ(Q[1][q])))
    end
#end

#
#ugrid = Array{Float64}(undef, 200, 200)
#vgrid = Array{Float64}(undef, 200, 200)
#for i = 1:200
#    for j = 1:200
#        ugrid[i, j] = Fₖ((x1grid[i, j], y1grid[i, j]))[1]
#        vgrid[i, j] = Fₖ((x1grid[i, j], y1grid[i, j]))[2]
#    end
#end
#f = Figure()
#Makie.scatter(reduce(vcat, ugrid), reduce(vcat, vgrid), label="")
#Makie.scatter!(stack(nodes, dims=1)[:, 1], stack(nodes, dims=1)[:, 2], color = :magenta)
#Makie.lines!(stack(nodes, dims=1)[:, 1], stack(nodes, dims=1)[:, 2], color = :red)