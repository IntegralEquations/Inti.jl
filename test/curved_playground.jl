using Inti
using Meshes
using StaticArrays
using GLMakie
using Gmsh
using LinearAlgebra
using NearestNeighbors
using ForwardDiff

function domain_and_mesh(; meshsize, meshorder = 1)
    Inti.clear_entities!()
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(meshorder)
    msh = Inti.import_mesh(; dim = 2)
    Ω = Inti.Domain(Inti.entities(msh)) do ent
        return Inti.geometric_dimension(ent) == 2
    end
    gmsh.finalize()
    return Ω, msh
end

meshsize = 0.1
qorder = 2

tmesh = @elapsed begin
    Ω, msh = domain_and_mesh(; meshsize)
end
@info "Mesh generation time: $tmesh"

Γ = Inti.external_boundary(Ω)
Ωₕ = view(msh, Ω)
Γₕ = view(msh, Γ)

ψ = (t) -> [cos(2*π*t), sin(2*π*t)]
ψ_der = (t) -> [-2*π*sin(2*π*t), 2*π*cos(2*π*t)]

# Now sample from the patch
t = LinRange(0, 1, 500*Int(1/meshsize))

param_disc = ψ.(t)
kdt = KDTree(transpose(stack(param_disc, dims=1)))

nbdry_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceHyperCube{1}, 2, SVector{2, Float64}}])[2]
bdry_node_idx = Vector{Int64}()
bdry_node_param_loc = Vector{Float64}()

uniqueidx(v) = unique(i -> v[i], eachindex(v))

# Re-write nodes to lay on exact boundary
for elind = 1:nbdry_els
    local node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceHyperCube{1}, 2, SVector{2, Float64}}][:, elind]
    local nodes = msh.nodes[node_indices]
    idxs, dists = nn(kdt, nodes)
    msh.nodes[node_indices[1]] = param_disc[idxs[1]]
    msh.nodes[node_indices[2]] = param_disc[idxs[2]]
    push!(bdry_node_idx, node_indices[1])
    push!(bdry_node_idx, node_indices[2])
    push!(bdry_node_param_loc, t[idxs[1]])
    push!(bdry_node_param_loc, t[idxs[2]])
end
I = uniqueidx(bdry_node_idx)
bdry_node_idx = bdry_node_idx[I]
bdry_node_param_loc = bdry_node_param_loc[I]
node_to_param = Dict(zip(bdry_node_idx, bdry_node_param_loc))

nvol_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{2, Float64}}])[2]

# generate volume parametrizations
circarea = 0.0
#elind = 747
els = []
for elind = 1:nvol_els
    #elind = 6
    #elind = 749
    node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{2, Float64}}][:, elind]
    nodes = msh.nodes[node_indices]

    # First determine if straight or curved
    verts_on_bdry = findall(x -> x ∈ bdry_node_idx, node_indices)
    if length(verts_on_bdry) > 1
        node_indices_on_bdry = node_indices[verts_on_bdry]

        # Need parametric coordinates of curved mapping to be consistent with straight simplex nodes
        α₁ = min(node_to_param[node_indices_on_bdry[1]], node_to_param[node_indices_on_bdry[2]])
        α₂ = max(node_to_param[node_indices_on_bdry[1]], node_to_param[node_indices_on_bdry[2]])
        # HACK: handle wrap-around in parameter space when using a global parametrization
        if abs(α₁ - α₂) > 0.5
            α₁ = α₂
            α₂ = 1.0
        end
        a₁ = ψ(α₁)
        a₂ = ψ(α₂)
        
        ## Interpolant πₖʲ construction from Inti
        α₁hat = 0.0
        α₂hat = 1.0
        f̂ₖ = (t) -> α₁ .+ (α₂ - α₁)*t
        f̂ₖ_comp = (x) -> f̂ₖ( (x[1] * α₁hat + x[2] * α₂hat)/(x[1] + x[2]) )

        # l = 1 projection onto linear FE space
        πₖ¹_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceLine, 2, SVector{2,Float64}})
        πₖ¹ψ_reference_nodes = Vector{SVector{2,Float64}}(undef, length(πₖ¹_nodes))
        for i in eachindex(πₖ¹_nodes)
            πₖ¹ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ¹_nodes[i][1]))
        end
        πₖ¹ψ_reference_nodes = SVector{2}(πₖ¹ψ_reference_nodes)
        πₖ¹ψ = (x) -> Inti.LagrangeElement{Inti.ReferenceLine}(πₖ¹ψ_reference_nodes)(x)

        # l = 2 projection onto quadratic FE space
        πₖ²_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceLine, 3, SVector{3,Float64}})
        πₖ²ψ_reference_nodes = Vector{SVector{2,Float64}}(undef, length(πₖ²_nodes))
        for i in eachindex(πₖ²_nodes)
            πₖ²ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ²_nodes[i][1]))
        end
        πₖ²ψ_reference_nodes = SVector{3}(πₖ²ψ_reference_nodes)
        πₖ²ψ = (x) -> Inti.LagrangeElement{Inti.ReferenceLine}(πₖ²ψ_reference_nodes)(x)

        # l = 3 projection onto cubic FE space
        πₖ³_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceLine, 4, SVector{4,Float64}})
        πₖ³ψ_reference_nodes = Vector{SVector{2,Float64}}(undef, length(πₖ³_nodes))
        for i in eachindex(πₖ³_nodes)
            πₖ³ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ³_nodes[i][1]))
        end
        πₖ³ψ_reference_nodes = SVector{4}(πₖ³ψ_reference_nodes)
        πₖ³ψ = (x) -> Inti.LagrangeElement{Inti.ReferenceLine}(πₖ³ψ_reference_nodes)(x)

        # Nonlinear map

        # l = 1
        Φₖ_l1 = (x) -> (x[1] + x[2])^3 * (ψ(f̂ₖ_comp(x)) - πₖ¹ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])))

        # l = 2
        Φₖ_l2 = (x) -> (x[1] + x[2])^4 * (ψ(f̂ₖ_comp(x)) - πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))) + (x[1] + x[2])^2*(πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) - πₖ¹ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])))

        # l = 3
        Φₖ_l3 = (x) -> (x[1] + x[2])^5 * (ψ(f̂ₖ_comp(x)) - πₖ³ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))) + (x[1] + x[2])^2*(πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) - πₖ¹ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))) + (x[1] + x[2])^3*(πₖ³ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) - πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])))

        # Zlamal nonlinear map
        Φₖ_Z = (x) -> x[2]/(1 - x[1]) * (ψ(x[1] * α₁ + (1 - x[1]) * α₂) - x[1] * a₁ - (1 - x[1])*a₂)

        # Affine map
        aₖ = msh.nodes[node_indices_on_bdry[1]]
        bₖ = msh.nodes[setdiff(node_indices, node_indices[verts_on_bdry])[1]]
        cₖ = msh.nodes[node_indices_on_bdry[2]]
        F̃ₖ = (x) -> [(cₖ[1] - bₖ[1])*x[1] + (aₖ[1] - bₖ[1])*x[2] + bₖ[1], (cₖ[2] - bₖ[2])*x[1] + (aₖ[2] - bₖ[2])*x[2] + bₖ[2]]

        # Full transformation
        Fₖ = (x) -> F̃ₖ(x) + Φₖ(x)
        JF̃ₖ = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
        D = Inti.ReferenceTriangle
        T = SVector{2,Float64}
        el = Inti.ParametricElement{D,T}(x -> Fₖ(x))
        push!(els, el)
        Jₖ_l1 = (x) ->  JF̃ₖ(x) + transpose(ForwardDiff.jacobian(Φₖ_l1, x))
        Jₖ_l2 = (x) ->  JF̃ₖ(x) + transpose(ForwardDiff.jacobian(Φₖ_l2, x))
        Jₖ_l3 = (x) ->  JF̃ₖ(x) + transpose(ForwardDiff.jacobian(Φₖ_l3, x))

        Fₖ_Z = (x) -> F̃ₖ(x) + Φₖ_Z(x)
        Jₖ_Z = (x) -> [cₖ[1]-bₖ[1] + Φₖ_Z_der_x1(x)[1]; aₖ[1]-bₖ[1] + Φₖ_Z_der_x2(x)[1];; cₖ[2]-bₖ[2] + Φₖ_Z_der_x1(x)[2]; aₖ[2]-bₖ[2] + Φₖ_Z_der_x2(x)[2]]
    else
        # Full transformation: Affine map
        aₖ = nodes[1]
        bₖ = nodes[2]
        cₖ = nodes[3]
        Fₖ = (x) -> [(cₖ[1] - bₖ[1])*x[1] + (aₖ[1] - bₖ[1])*x[2] + bₖ[1], (cₖ[2] - bₖ[2])*x[1] + (aₖ[2] - bₖ[2])*x[2] + bₖ[2]]
        Jₖ = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
        Jₖ_l1 = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
        Jₖ_l2 = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
        Jₖ_l3 = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
        Jₖ_Z = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
    end

    Q = Inti.VioreanuRokhlin(; domain=Inti.ReferenceTriangle(), order=qorder)()
    nq = length(Q[2])
    for q in 1:nq
        global circarea
        circarea += Q[2][q] * abs(det(Jₖ_l3(Q[1][q])))
    end
end