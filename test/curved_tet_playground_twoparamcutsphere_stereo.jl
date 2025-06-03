using Test
using Inti
using FMM3D
using Gmsh
using LinearAlgebra
using Random
using StaticArrays
using NearestNeighbors
using ForwardDiff
using NonlinearSolve

include("test_utils.jl")

# create a boundary and area meshes and quadrature only once
meshsize = .1
qorder = 5

#Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = meshsize)
Ω, msh = gmsh_cut_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = meshsize)
Γ = Inti.external_boundary(Ω)
Γ_msh = view(msh, Γ)

ang = π/2
X = LinRange(-1.1, 1.1, 100*round(Int, 1/meshsize))
Y = LinRange(-1.1, 1.1, 100*round(Int, 1/meshsize))
#θ₂ = LinRange(-ang, π-ang, 50*round(Int, 1/meshsize))
#θ₂ = LinRange(0, π, 50*round(Int, 1/meshsize))
# v = (θ, ϕ)
#ang = 0.0
#M = [cos(ang); 0; sin(ang);; 0; 1; 0;; -1*sin(ang); 0; cos(ang)]
#M = [cos(ang) 0 sin(ang); 0 1 0; -1*sin(ang) 0 cos(ang)]
#ψ₁ = (v) -> [0.0, 0.0, 0.0]
#ψ₁ = (v) -> [sin(v[1]) * cos(v[2]), sin(v[1]) * sin(v[2]), cos(v[1])]
#ψ₂ = (v) -> M * ψ₁(v)
ψ₁ = (v) -> [-2*v[1]/(1 + v[1]^2 + v[2]^2), 2*v[2]/(1 + v[1]^2 + v[2]^2), (-1 + v[1]^2 + v[2]^2)/(1 + v[1]^2 + v[2]^2)]
ψ₂ = (v) -> [2*v[1]/(1 + v[1]^2 + v[2]^2), 2*v[2]/(1 + v[1]^2 + v[2]^2), (1 - v[1]^2 - v[2]^2)/(1 + v[1]^2 + v[2]^2)]
ψ₁⁻¹ =  (x) -> [-x[1]/(1 - x[3]), x[2]/(1 - x[3])]
ψ₂⁻¹ =  (x) -> [x[1]/(1 + x[3]), x[2]/(1 + x[3])]
#ψ₂ = (v) -> [sin(v[1] + ang) * cos(v[2]), sin(v[1] + ang) * sin(v[2]), cos(v[1] + ang)]

function chart_id(face_nodes)
    id = 1
    #if all([abs(q[3]) for q in face_nodes] .> 0.75)
    #    id = 2
    #end
    if all([q[3] for q in face_nodes] .> 0.0)
        id = 2
    end
    return id
end

#function ψ₁⁻¹(v0, p)
#    F₁ = (v, p) -> ψ₁(v) - p
#    prob₁= NonlinearProblem(F₁, v0, p)
#    ψ₁⁻¹ = NonlinearSolve.solve(prob₁, SimpleNewtonRaphson())
#end
#
#function ψ₂⁻¹(v0, p)
#    F₂ = (v, p) -> ψ₂(v) - p
#    prob₂= NonlinearProblem(F₂, v0, p)
#    ψ₂⁻¹ = NonlinearSolve.solve(prob₂, SimpleNewtonRaphson())
#end

chart_1 = Array{SVector{3,Float64}}(undef, length(X)*length(Y))
chart_1_cart_idxs_X = []
chart_1_cart_idxs_Y = []
for i in eachindex(X)
    for j in eachindex(Y)
        chart_1[(i-1)*length(Y) + j] = [k for k in ψ₁((X[i], Y[j]))]
        push!(chart_1_cart_idxs_X, i)
        push!(chart_1_cart_idxs_Y, j)
    end
end
chart_1_kdt = KDTree(chart_1; reorder = false)
# chart 2
chart_2 = Array{SVector{3,Float64}}(undef, length(X)*length(Y))
chart_2_cart_idxs_X = []
chart_2_cart_idxs_Y = []
for i in eachindex(X)
    for j in eachindex(Y)
        chart_2[(i-1)*length(Y) + j] = [k for k in ψ₂((X[i], Y[j]))]
        push!(chart_2_cart_idxs_X, i)
        push!(chart_2_cart_idxs_Y, j)
    end
end
chart_2_kdt = KDTree(chart_2; reorder = false)

n2e = Inti.node2etags(msh)

nbdry_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}])[2]
chart_1_bdry_node_idx = Vector{Int64}()
chart_1_bdry_node_param_loc = Vector{Vector{Float64}}()
chart_2_bdry_node_idx = Vector{Int64}()
chart_2_bdry_node_param_loc = Vector{Vector{Float64}}()

uniqueidx(v) = unique(i -> v[i], eachindex(v))

# Set up chart <-> node Dict
for elind = 1:nbdry_els
    node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}][:, elind]
    nodes = msh.nodes[node_indices]

    if chart_id(nodes) == 1
        if all(norm.(nodes) .≈ 1)
            idxs, dists = nn(chart_1_kdt, nodes)
            ψ⁻¹ = ψ₁⁻¹
            if node_indices[1] ∉ chart_1_bdry_node_idx && node_indices[1] ∉ chart_2_bdry_node_idx
                if abs(msh.nodes[node_indices[1]][3]) ≈ 0.8
                #if false #abs(msh.nodes[node_indices[1]][3]) ≈ 0.8
                    α = Vector{Float64}(ψ⁻¹(copy(msh.nodes[node_indices[1]])))
                    push!(chart_1_bdry_node_idx, node_indices[1])
                    push!(chart_1_bdry_node_param_loc, α)
                else
                    msh.nodes[node_indices[1]] = chart_1[idxs[1]]
                    push!(chart_1_bdry_node_param_loc, Vector{Float64}([X[chart_1_cart_idxs_X[idxs[1]]], Y[chart_1_cart_idxs_Y[idxs[1]]]]))
                    push!(chart_1_bdry_node_idx, node_indices[1])
                end
            end
            if node_indices[2] ∉ chart_1_bdry_node_idx && node_indices[2] ∉ chart_2_bdry_node_idx
                if abs(msh.nodes[node_indices[2]][3]) ≈ 0.8
                #if false #abs(msh.nodes[node_indices[2]][3]) ≈ 0.8
                    α = Vector{Float64}(ψ⁻¹(copy(msh.nodes[node_indices[2]])))
                    push!(chart_1_bdry_node_idx, node_indices[2])
                    push!(chart_1_bdry_node_param_loc, α)
                else
                    msh.nodes[node_indices[2]] = chart_1[idxs[2]]
                    push!(chart_1_bdry_node_idx, node_indices[2])
                    push!(chart_1_bdry_node_param_loc, Vector{Float64}([X[chart_1_cart_idxs_X[idxs[2]]], Y[chart_1_cart_idxs_Y[idxs[2]]]]))
                end
            end
            if node_indices[3] ∉ chart_1_bdry_node_idx && node_indices[3] ∉ chart_2_bdry_node_idx
                if abs(msh.nodes[node_indices[3]][3]) ≈ 0.8
                #if false #abs(msh.nodes[node_indices[3]][3]) ≈ 0.8
                    α = Vector{Float64}(ψ⁻¹(copy(msh.nodes[node_indices[3]])))
                    push!(chart_1_bdry_node_idx, node_indices[3])
                    push!(chart_1_bdry_node_param_loc, α)
                else
                    msh.nodes[node_indices[3]] = chart_1[idxs[3]]
                    push!(chart_1_bdry_node_idx, node_indices[3])
                    push!(chart_1_bdry_node_param_loc, Vector{Float64}([X[chart_1_cart_idxs_X[idxs[3]]], Y[chart_1_cart_idxs_Y[idxs[3]]]]))
                end
            end
        end
    else # chart_id == 2
        if all(norm.(nodes) .≈ 1)
            idxs, dists = nn(chart_2_kdt, nodes)
            ψ⁻¹ = ψ₂⁻¹
            if abs(nodes[2][1]) < 10^(-13) && abs(nodes[2][2]) < 10^(-13)  && nodes[2][3] ≈ -0.8
                println("KILGORE WAS HERE")
            end
            if node_indices[1] ∉ chart_1_bdry_node_idx && node_indices[1] ∉ chart_2_bdry_node_idx
                if abs(msh.nodes[node_indices[1]][3]) ≈ 0.8
                #if false #abs(msh.nodes[node_indices[1]][3]) ≈ 0.8
                    α = Vector{Float64}(ψ⁻¹(copy(msh.nodes[node_indices[1]])))
                    push!(chart_2_bdry_node_idx, node_indices[1])
                    push!(chart_2_bdry_node_param_loc, α)
                else
                    msh.nodes[node_indices[1]] = chart_2[idxs[1]]
                    push!(chart_2_bdry_node_param_loc, Vector{Float64}([X[chart_2_cart_idxs_X[idxs[1]]], Y[chart_2_cart_idxs_Y[idxs[1]]]]))
                    push!(chart_2_bdry_node_idx, node_indices[1])
                end
            end
            if node_indices[2] ∉ chart_1_bdry_node_idx && node_indices[2] ∉ chart_2_bdry_node_idx
                if abs(msh.nodes[node_indices[2]][3]) ≈ 0.8
                #if false #abs(msh.nodes[node_indices[2]][3]) ≈ 0.8
                    α = Vector{Float64}(ψ⁻¹(copy(msh.nodes[node_indices[2]])))
                    push!(chart_2_bdry_node_idx, node_indices[2])
                    push!(chart_2_bdry_node_param_loc, α)
                else
                    msh.nodes[node_indices[2]] = chart_2[idxs[2]]
                    push!(chart_2_bdry_node_idx, node_indices[2])
                    push!(chart_2_bdry_node_param_loc, Vector{Float64}([X[chart_2_cart_idxs_X[idxs[2]]], Y[chart_2_cart_idxs_Y[idxs[2]]]]))
                end
            end
            if node_indices[3] ∉ chart_1_bdry_node_idx && node_indices[3] ∉ chart_2_bdry_node_idx
                if abs(msh.nodes[node_indices[3]][3]) ≈ 0.8
                #if false #abs(msh.nodes[node_indices[3]][3]) ≈ 0.8
                    α = Vector{Float64}(ψ⁻¹(copy(msh.nodes[node_indices[3]])))
                    push!(chart_2_bdry_node_idx, node_indices[3])
                    push!(chart_2_bdry_node_param_loc, α)
                else
                    msh.nodes[node_indices[3]] = chart_2[idxs[3]]
                    push!(chart_2_bdry_node_idx, node_indices[3])
                    push!(chart_2_bdry_node_param_loc, Vector{Float64}([X[chart_2_cart_idxs_X[idxs[3]]], Y[chart_2_cart_idxs_Y[idxs[3]]]]))
                end
            end
        end
    end
end
chart_1_node_to_param = Dict(zip(chart_1_bdry_node_idx, chart_1_bdry_node_param_loc))
chart_2_node_to_param = Dict(zip(chart_2_bdry_node_idx, chart_2_bdry_node_param_loc))

nvol_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}}])[2]
ncurv_vols = 0
spharea = 0.0
nnewton = 0
elcount = 0
els = []
elvol = Vector{Float64}(undef, nvol_els)
for elind = 1:nvol_els
    # j = 2
    #elind = 18819

    # j = 3
    #elind = 18818
    #elind = 17793

    #elind = 5319
    #elind = 15315
    #elind = 17778
    node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}}][:, elind]
    nodes = msh.nodes[node_indices]
    
    j = 0
    verts_on_bdry_chart_1 = findall(x -> x ∈ chart_1_bdry_node_idx, node_indices)
    verts_on_bdry_chart_2 = findall(x -> x ∈ chart_2_bdry_node_idx, node_indices)
    if !isempty(verts_on_bdry_chart_1) || !isempty(verts_on_bdry_chart_2)
        nverts_in_chart = [length(verts_on_bdry_chart_1), length(verts_on_bdry_chart_2)]
        chart_num = argmax(nverts_in_chart)
        nverts_in_major_chart = nverts_in_chart[chart_num]
        j = sum(nverts_in_chart)
        # TODO Improve this; this is a tentative list of vertices on boundary.
        # Imperfect since you want *all* vertices to get j right but need to
        # determine the proper chart based on this number, through chart_num
        # that itself requires verts_on_bdry. Break the cycle
        verts_on_bdry = (chart_num == 1) ? verts_on_bdry_chart_1 : verts_on_bdry_chart_2
    end
    if j > 1
        global ncurv_vols += 1
        node_indices_on_bdry = deepcopy(node_indices[verts_on_bdry])
        #chart_num = chart_id(msh.nodes[node_indices_on_bdry])
        if chart_num == 1
            node_to_param = chart_1_node_to_param
            ψ = ψ₁
            ψ⁻¹ = ψ₁⁻¹
            #verts_on_bdry = verts_on_bdry_chart_1
            α₁ = deepcopy(node_to_param[node_indices_on_bdry[1]])
            #if nverts_in_major_chart == 1
            #    println(elind)
            #end
            if nverts_in_major_chart >= 2
                α₂ = deepcopy(node_to_param[node_indices_on_bdry[2]])
            else
                push!(node_indices_on_bdry, node_indices[verts_on_bdry_chart_2[1]])
                p = deepcopy(msh.nodes[node_indices[verts_on_bdry_chart_2[1]]])
                global nnewton += 1
                α₂ = ψ⁻¹(p)
                @assert norm(ψ(α₂) - p) < 10^(-14)
            end
            if nverts_in_major_chart >= 3
                α₃ = deepcopy(node_to_param[node_indices_on_bdry[3]])
            else
                #println(elind)
                # Find missing node α₃ that (non-uniquely) defines the curved face simplex containing α₁, α₂
                candidate_els = Inti.elements_containing_nodes(n2e, node_indices_on_bdry)
                # Filter out volume elements; should be at most two face simplices remaining
                candidate_els = candidate_els[length.(candidate_els).==3]
                # Take the first face simplex; while either would work if j=2,
                # if j=3 only one of the candidate face triangles will work, so
                # find that one
                if candidate_els[1][1] ∉ node_indices_on_bdry
                    p = deepcopy(msh.nodes[candidate_els[1][1]])
                elseif candidate_els[1][2] ∉ node_indices_on_bdry
                    p = deepcopy(msh.nodes[candidate_els[1][2]])
                elseif candidate_els[1][3] ∉ node_indices_on_bdry
                    p = deepcopy(msh.nodes[candidate_els[1][3]])
                else
                    @assert false
                end
                if j == 3 && p ∉ nodes
                    if candidate_els[2][1] ∉ node_indices_on_bdry
                        p = deepcopy(msh.nodes[candidate_els[2][1]])
                    elseif candidate_els[2][2] ∉ node_indices_on_bdry
                        p = deepcopy(msh.nodes[candidate_els[2][2]])
                    elseif candidate_els[2][3] ∉ node_indices_on_bdry
                        p = deepcopy(msh.nodes[candidate_els[2][3]])
                    else
                        @assert false
                    end
                end
                global nnewton += 1
                α₃ = ψ⁻¹(p)
            end
        else
            node_to_param = chart_2_node_to_param
            ψ = ψ₂
            ψ⁻¹ = ψ₂⁻¹
            #verts_on_bdry = verts_on_bdry_chart_2
            α₁ = deepcopy(node_to_param[node_indices_on_bdry[1]])
            if nverts_in_major_chart >= 2
                α₂ = deepcopy(node_to_param[node_indices_on_bdry[2]])
            else
                # minor chart is chart 1, can be improved and streamlined
                push!(node_indices_on_bdry, node_indices[verts_on_bdry_chart_1[1]])
                p = deepcopy(msh.nodes[node_indices[verts_on_bdry_chart_1[1]]])
                global nnewton += 1
                α₂ = ψ⁻¹(p)
            end
            if nverts_in_major_chart >= 3
                α₃ = deepcopy(node_to_param[node_indices_on_bdry[3]])
            else
                # Find missing node α₃ that (non-uniquely) defines the curved face simplex containing α₁, α₂
                candidate_els = Inti.elements_containing_nodes(n2e, node_indices_on_bdry)
                # Filter out volume elements; should be at most two face simplices remaining
                candidate_els = candidate_els[length.(candidate_els).==3]
                # Take the first face simplex; while either would work if j=2,
                # if j=3 only one of the candidate face triangles will work, so
                # find that one
                if candidate_els[1][1] ∉ node_indices_on_bdry
                    p = deepcopy(msh.nodes[candidate_els[1][1]])
                elseif candidate_els[1][2] ∉ node_indices_on_bdry
                    p = deepcopy(msh.nodes[candidate_els[1][2]])
                elseif candidate_els[1][3] ∉ node_indices_on_bdry
                    p = deepcopy(msh.nodes[candidate_els[1][3]])
                else
                    @assert false
                end
                if j == 3 && p ∉ nodes
                    if candidate_els[2][1] ∉ node_indices_on_bdry
                        p = deepcopy(msh.nodes[candidate_els[2][1]])
                    elseif candidate_els[2][2] ∉ node_indices_on_bdry
                        p = deepcopy(msh.nodes[candidate_els[2][2]])
                    elseif candidate_els[2][3] ∉ node_indices_on_bdry
                        p = deepcopy(msh.nodes[candidate_els[2][3]])
                    else
                        @assert false
                    end
                end
                global nnewton += 1
                α₃ = ψ⁻¹(p)
            end
        end
        atol = 10^(-4)
        a₁ = SVector{3,Float64}(ψ(α₁))
        a₂ = SVector{3,Float64}(ψ(α₂))
        a₃ = SVector{3,Float64}(ψ(α₃))
        α₁hat = SVector{2,Float64}(1.0, 0.0)
        α₂hat = SVector{2,Float64}(0.0, 1.0)
        α₃hat = SVector{2,Float64}(0.0, 0.0)

        πₖ¹_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, 3, SVector{2,Float64}})
        α_reference_nodes = Vector{SVector{2,Float64}}(undef, length(πₖ¹_nodes))
        α_reference_nodes[1] = SVector{2}(α₃)
        α_reference_nodes[2] = SVector{2}(α₁)
        α_reference_nodes[3] = SVector{2}(α₂)
        α_reference_nodes = SVector{3}(α_reference_nodes)
        f̂ₖ = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(α_reference_nodes)(x)

        @assert (f̂ₖ(α₁hat) ≈ α₁) && (f̂ₖ(α₂hat) ≈ α₂) && (f̂ₖ(α₃hat) ≈ α₃)
        @assert a₁ ≈ ψ(f̂ₖ(α₁hat))
        @assert a₂ ≈ ψ(f̂ₖ(α₂hat))
        @assert a₃ ≈ ψ(f̂ₖ(α₃hat))
        @assert a₁ ≈ nodes[1] || a₁ ≈ nodes[2] || a₁ ≈ nodes[3] || a₁ ≈ nodes[4]
        @assert a₂ ≈ nodes[1] || a₂ ≈ nodes[2] || a₂ ≈ nodes[3] || a₂ ≈ nodes[4]
        if j == 3
            @assert a₃ ≈ nodes[1] || a₃ ≈ nodes[2] || a₃ ≈ nodes[3] || a₃ ≈ nodes[4]
        end
        
        # Affine map
        # Vertices aₖ and bₖ always lay on surface. Vertex dₖ always lays in volume.
        aₖ = a₁
        bₖ = a₂
        cₖ = a₃
        atol = 10^-8
        facenodes = [a₁, a₂, a₃]
        skipnode = 0
        if all(norm.(Ref(nodes[1]) .- facenodes) .> atol)
            dₖ = nodes[1]
            skipnode = 1
        elseif all(norm.(Ref(nodes[2]) .- facenodes) .> atol)
            dₖ = nodes[2]
            skipnode = 2
        elseif all(norm.(Ref(nodes[3]) .- facenodes) .> atol)
            dₖ = nodes[3]
            skipnode = 3
        elseif all(norm.(Ref(nodes[4]) .- facenodes) .> atol)
            dₖ = nodes[4]
            skipnode = 4
        else
            error("Uhoh")
        end
        if j == 2
            if all(norm.(Ref(nodes[1]) .- facenodes) .> atol)
                (skipnode == 1) || (cₖ = copy(nodes[1]))
            end
            if all(norm.(Ref(nodes[2]) .- facenodes) .> atol)
                (skipnode == 2) || (cₖ = copy(nodes[2]))
            end
            if all(norm.(Ref(nodes[3]) .- facenodes) .> atol)
                (skipnode == 3) || (cₖ = copy(nodes[3]))
            end
            if all(norm.(Ref(nodes[4]) .- facenodes) .> atol)
                (skipnode == 4) || (cₖ = copy(nodes[4]))
            end
            atol = 10^-12
            @assert abs(norm(cₖ) - 1.0) > atol
            @assert norm(cₖ - a₃) > atol
            @assert norm(cₖ - dₖ) > atol
            @assert norm(dₖ - a₁) > atol
            @assert norm(dₖ - a₂) > atol
        end
        atol = 10^-12
        @assert abs(norm(dₖ) - 1) > atol
        @assert !all(norm.(Ref(a₁) .- nodes) .> atol)
        @assert !all(norm.(Ref(a₂) .- nodes) .> atol)
        if j == 3
            @assert !all(norm.(Ref(a₃) .- nodes) .> atol)
        end
        F̃ₖ = (x) -> [(aₖ[1] - dₖ[1])*x[1] + (bₖ[1] - dₖ[1])*x[2] + (cₖ[1] - dₖ[1])*x[3] + dₖ[1], (aₖ[2] - dₖ[2])*x[1] + (bₖ[2] - dₖ[2])*x[2] + (cₖ[2] - dₖ[2])*x[3] + dₖ[2], (aₖ[3] - dₖ[3])*x[1] + (bₖ[3] - dₖ[3])*x[2] + (cₖ[3] - dₖ[3])*x[3] + dₖ[3]]
        @assert aₖ ≈ a₁
        @assert bₖ ≈ a₂

        # l = 1
        πₖ¹_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, binomial(2+1,2), SVector{2,Float64}})
        πₖ¹ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ¹_nodes))
        for i in eachindex(πₖ¹_nodes)
            πₖ¹ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ¹_nodes[i]))
        end
        πₖ¹ψ_reference_nodes = SVector{binomial(2+1,2)}(πₖ¹ψ_reference_nodes)
        πₖ¹ψ = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ¹ψ_reference_nodes)(x)
        #l = 2
        πₖ²_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, binomial(2+2,2), SVector{2,Float64}})
        πₖ²ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ²_nodes))
        for i in eachindex(πₖ²_nodes)
            πₖ²ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ²_nodes[i]))
        end
        πₖ²ψ_reference_nodes = SVector{binomial(2+2,2)}(πₖ²ψ_reference_nodes)
        πₖ²ψ = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ²ψ_reference_nodes)(x)
        #l = 3
        πₖ³_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, binomial(2+3,2), SVector{2,Float64}})
        πₖ³ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ³_nodes))
        for i in eachindex(πₖ³_nodes)
            πₖ³ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ³_nodes[i]))
        end
        πₖ³ψ_reference_nodes = SVector{binomial(2+3,2)}(πₖ³ψ_reference_nodes)
        πₖ³ψ = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ³ψ_reference_nodes)(x)
        #l = 4
        πₖ⁴_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, binomial(2+4,2), SVector{2,Float64}})
        πₖ⁴ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ⁴_nodes))
        for i in eachindex(πₖ⁴_nodes)
            πₖ⁴ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁴_nodes[i]))
        end
        πₖ⁴ψ_reference_nodes = SVector{binomial(2+4,2)}(πₖ⁴ψ_reference_nodes)
        πₖ⁴ψ = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ⁴ψ_reference_nodes)(x)
        #l = 5
        πₖ⁵_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, binomial(2+5,2), SVector{2,Float64}})
        πₖ⁵ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ⁵_nodes))
        for i in eachindex(πₖ⁵_nodes)
            πₖ⁵ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁵_nodes[i]))
        end
        πₖ⁵ψ_reference_nodes = SVector{binomial(2+5,2)}(πₖ⁵ψ_reference_nodes)
        πₖ⁵ψ = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ⁵ψ_reference_nodes)(x)

        # Nonlinear map
        if j == 3
            f̂ₖ_comp = (x) -> f̂ₖ( (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat)/(x[1] + x[2] + x[3]) )
            Φₖ_θ1 = (x) -> ( (x[1] + x[2] + x[3])^3 * (ψ(f̂ₖ_comp(x)) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))))
            Φₖ_θ2 = (x) -> ( (x[1] + x[2] + x[3])^4 * (ψ(f̂ₖ_comp(x)) - πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^2 * (πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))))
            Φₖ_θ3 = (x) -> ( (x[1] + x[2] + x[3])^5 * (ψ(f̂ₖ_comp(x)) - πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^2 * (πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^3 * (πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) )
            Φₖ_θ4 = (x) -> ( (x[1] + x[2] + x[3])^6 * (ψ(f̂ₖ_comp(x)) - πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^2 * (πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^3 * (πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^4 * (πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) )
            Φₖ_θ5 = (x) -> ( (x[1] + x[2] + x[3])^7 * (ψ(f̂ₖ_comp(x)) - πₖ⁵ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^2 * (πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^3 * (πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^4 * (πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^5 * (πₖ⁵ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])) - πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]))))
        else
            f̂ₖ_comp = (x) -> f̂ₖ( (x[1] * α₁hat + x[2] * α₂hat)/(x[1] + x[2]) )
            Φₖ_θ1 = (x) -> ( (x[1] + x[2])^3 * (ψ(f̂ₖ_comp(x)) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))))
            Φₖ_θ2 = (x) -> ( (x[1] + x[2])^4 * (ψ(f̂ₖ_comp(x)) - πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^2 * (πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))))
            Φₖ_θ3 = (x) -> ( (x[1] + x[2])^5 * (ψ(f̂ₖ_comp(x)) - πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^2 * (πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^3 * (πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))))
            Φₖ_θ4 = (x) -> ( (x[1] + x[2])^6 * (ψ(f̂ₖ_comp(x)) - πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^2 * (πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^3 * (πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^4 * (πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))))
            Φₖ_θ5 = (x) -> ( (x[1] + x[2])^7 * (ψ(f̂ₖ_comp(x)) - πₖ⁵ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^2 * (πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^3 * (πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^4 * (πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) + (x[1] + x[2])^5 * (πₖ⁵ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) - πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))))
        end
        Φₖ = Φₖ_θ5

        # Full transformation
        atol = 10^(-12)
        Fₖ = (x) -> F̃ₖ(x) + Φₖ(x)
        @assert norm(Fₖ([1.0, 0.0, 0.0]) - a₁) < atol
        @assert norm(Fₖ([0.0, 1.0, 0.0]) - a₂) < atol
        @assert norm(Fₖ([1.0, 0.0, 0.0]) - aₖ) < atol
        @assert norm(Fₖ([0.0, 1.0, 0.0]) - bₖ) < atol
        @assert norm(Fₖ([0.0, 0.0000000000000001, 1.0]) - cₖ) < atol
        @assert norm(Fₖ([0.0, 0.0000000000000001, 0.0]) - dₖ) < atol
        if j == 3
            @assert norm(a₃ - cₖ) < atol
            @assert norm(Fₖ([0.0, 0.0, 1.0]) - cₖ) < atol
            @assert norm(Fₖ([0.0, 0.0, 1.0]) - a₃) < atol
            @assert abs(norm(Fₖ([0.2, 0.3, 0.5])) - 1.0) < atol
            @assert abs(norm(Fₖ([0.0, 0.0, 1.0])) - 1.0) < atol
            @assert norm(Φₖ([0.0, 0.0, 0.3])) < atol
            @assert norm(Φₖ([0.0, 0.3, 0.0])) < atol
            @assert norm(Φₖ([0.3, 0.0, 0.0])) < atol
            @assert norm(Φₖ([0.3, 0.45, 0.25]) - (ψ(f̂ₖ_comp([0.3, 0.45, 0.25])) - 0.3*a₁ - 0.45*a₂ - 0.25*a₃)) < atol
            @assert norm(Φₖ([0.55, 0.45, 0.0]) - (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂)) < atol
        end
        @assert abs(norm(Fₖ([0.6, 0.4, 0.0])) - 1.0) < atol
        @assert abs(norm(Fₖ([1.0, 0.0, 0.0])) - 1.0) < atol
        @assert abs(norm(Fₖ([0.0, 1.0, 0.0])) - 1.0) < atol
        @assert norm(Φₖ([0.0, 0.0000000000000001, 0.3])) < atol
        @assert norm(Φₖ([0.0, 0.3, 0.0])) < atol
        @assert norm(Φₖ([0.3, 0.0, 0.0])) < atol
        if j == 2
            @assert norm(Φₖ([0.6, 0.0, 0.4])) < atol
            @assert norm(Φₖ([0.0, 0.6, 0.4])) < atol
            @assert norm(Φₖ([0.55, 0.45, 0.0]) - (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂)) < atol
        end
    else
        aₖ = nodes[1]
        bₖ = nodes[2]
        cₖ = nodes[3]
        dₖ = nodes[4]
        Fₖ = (x) -> [(aₖ[1] - bₖ[1])*x[1] + (cₖ[1] - bₖ[1])*x[2] + (dₖ[1] - bₖ[1])*x[3] + bₖ[1], (aₖ[2] - bₖ[2])*x[1] + (cₖ[2] - bₖ[2])*x[2] + (dₖ[2] - bₖ[2])*x[3] + bₖ[2], (aₖ[3] - bₖ[3])*x[1] + (cₖ[3] - bₖ[3])*x[2] + (dₖ[3] - bₖ[3])*x[3] + bₖ[3]]
    end
    
    Jₖ = (x) -> transpose(ForwardDiff.jacobian(Fₖ, x))
    
    Q = Inti.Gauss(; domain=Inti.ReferenceTetrahedron(), order=qorder)()
    #Q = Inti.VioreanuRokhlin(; domain=Inti.ReferenceTetrahedron(), order=qorder)()
    nq = length(Q[2])
    elarea = 0.0
    for q in 1:nq
        tmp = Q[2][q] * abs(det(Jₖ(Q[1][q])))
        elarea += tmp
        #global elarea += tmp
    end
    global elvol[elind] = elarea
    global spharea += elarea

    ### Comparison below with using single parametrization 
    if j > 1 && chart_num == 2
        node_indices_on_bdry = deepcopy(node_indices[verts_on_bdry])
        #chart_num = chart_id(msh.nodes[node_indices_on_bdry])
        node_to_param = chart_1_node_to_param
        if chart_num == 2
            ψ_alt = ψ₁
            ψ⁻¹_alt = ψ₁⁻¹
        else
            ψ_alt = ψ₂
            ψ⁻¹_alt = ψ₂⁻¹
        end
        α₁_alt = ψ⁻¹_alt(a₁)
        α₂_alt = ψ⁻¹_alt(a₂)
        α₃_alt = ψ⁻¹_alt(a₃)
        a₁_alt = SVector{3,Float64}(ψ_alt(α₁_alt))
        a₂_alt = SVector{3,Float64}(ψ_alt(α₂_alt))
        a₃_alt = SVector{3,Float64}(ψ_alt(α₃_alt))
        α₁hat_alt = SVector{2,Float64}(1.0, 0.0)
        α₂hat_alt = SVector{2,Float64}(0.0, 1.0)
        α₃hat_alt = SVector{2,Float64}(0.0, 0.0)

        πₖ¹_nodes_alt = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, 3, SVector{2,Float64}})
        α_reference_nodes_alt = Vector{SVector{2,Float64}}(undef, length(πₖ¹_nodes_alt))
        α_reference_nodes_alt[1] = SVector{2}(α₃_alt)
        α_reference_nodes_alt[2] = SVector{2}(α₁_alt)
        α_reference_nodes_alt[3] = SVector{2}(α₂_alt)
        α_reference_nodes_alt = SVector{3}(α_reference_nodes_alt)
        f̂ₖ_alt = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(α_reference_nodes_alt)(x)

        @assert (f̂ₖ_alt(α₁hat_alt) ≈ α₁_alt) && (f̂ₖ_alt(α₂hat_alt) ≈ α₂_alt) && (f̂ₖ_alt(α₃hat_alt) ≈ α₃_alt)
        @assert a₁_alt ≈ ψ_alt(f̂ₖ_alt(α₁hat_alt))
        @assert a₂_alt ≈ ψ_alt(f̂ₖ_alt(α₂hat_alt))
        @assert a₃_alt ≈ ψ_alt(f̂ₖ_alt(α₃hat_alt))
        @assert a₁_alt ≈ nodes[1] || a₁_alt ≈ nodes[2] || a₁_alt ≈ nodes[3] || a₁_alt ≈ nodes[4]
        @assert a₂_alt ≈ nodes[1] || a₂_alt ≈ nodes[2] || a₂_alt ≈ nodes[3] || a₂_alt ≈ nodes[4]
        if j == 3
            @assert a₃_alt ≈ nodes[1] || a₃_alt ≈ nodes[2] || a₃_alt ≈ nodes[3] || a₃_alt ≈ nodes[4]
        end
        
        # Affine map
        # Vertices aₖ and bₖ always lay on surface. Vertex dₖ always lays in volume.
        aₖ_alt = a₁_alt
        bₖ_alt = a₂_alt
        cₖ_alt = a₃_alt
        atol = 10^-8
        facenodes_alt = [a₁_alt, a₂_alt, a₃_alt]
        skipnode = 0
        if all(norm.(Ref(nodes[1]) .- facenodes_alt) .> atol)
            dₖ_alt = nodes[1]
            skipnode = 1
        elseif all(norm.(Ref(nodes[2]) .- facenodes_alt) .> atol)
            dₖ_alt = nodes[2]
            skipnode = 2
        elseif all(norm.(Ref(nodes[3]) .- facenodes_alt) .> atol)
            dₖ_alt = nodes[3]
            skipnode = 3
        elseif all(norm.(Ref(nodes[4]) .- facenodes_alt) .> atol)
            dₖ_alt = nodes[4]
            skipnode = 4
        else
            error("Uhoh")
        end
        if j == 2
            if all(norm.(Ref(nodes[1]) .- facenodes_alt) .> atol)
                (skipnode == 1) || (cₖ_alt = copy(nodes[1]))
            end
            if all(norm.(Ref(nodes[2]) .- facenodes_alt) .> atol)
                (skipnode == 2) || (cₖ_alt = copy(nodes[2]))
            end
            if all(norm.(Ref(nodes[3]) .- facenodes_alt) .> atol)
                (skipnode == 3) || (cₖ_alt = copy(nodes[3]))
            end
            if all(norm.(Ref(nodes[4]) .- facenodes_alt) .> atol)
                (skipnode == 4) || (cₖ_alt = copy(nodes[4]))
            end
            atol = 10^-12
            @assert abs(norm(cₖ_alt) - 1.0) > atol
            @assert norm(cₖ_alt - a₃_alt) > atol
            @assert norm(cₖ_alt - dₖ_alt) > atol
            @assert norm(dₖ_alt - a₁_alt) > atol
            @assert norm(dₖ_alt - a₂_alt) > atol
        end
        atol = 10^-12
        @assert abs(norm(dₖ_alt) - 1) > atol
        @assert !all(norm.(Ref(a₁_alt) .- nodes) .> atol)
        @assert !all(norm.(Ref(a₂_alt) .- nodes) .> atol)
        if j == 3
            @assert !all(norm.(Ref(a₃_alt) .- nodes) .> atol)
        end
        F̃ₖ_alt = (x) -> [(aₖ_alt[1] - dₖ_alt[1])*x[1] + (bₖ_alt[1] - dₖ_alt[1])*x[2] + (cₖ_alt[1] - dₖ_alt[1])*x[3] + dₖ_alt[1], (aₖ_alt[2] - dₖ_alt[2])*x[1] + (bₖ_alt[2] - dₖ_alt[2])*x[2] + (cₖ_alt[2] - dₖ_alt[2])*x[3] + dₖ_alt[2], (aₖ_alt[3] - dₖ_alt[3])*x[1] + (bₖ_alt[3] - dₖ_alt[3])*x[2] + (cₖ_alt[3] - dₖ_alt[3])*x[3] + dₖ_alt[3]]
        @assert aₖ_alt ≈ a₁_alt
        @assert bₖ_alt ≈ a₂_alt

        # l = 1
        πₖ¹_nodes_alt = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, binomial(2+1,2), SVector{2,Float64}})
        πₖ¹ψ_reference_nodes_alt = Vector{SVector{3,Float64}}(undef, length(πₖ¹_nodes))
        for i in eachindex(πₖ¹_nodes_alt)
            πₖ¹ψ_reference_nodes_alt[i] = ψ_alt(f̂ₖ_alt(πₖ¹_nodes_alt[i]))
        end
        πₖ¹ψ_reference_nodes_alt = SVector{binomial(2+1,2)}(πₖ¹ψ_reference_nodes_alt)
        πₖ¹ψ_alt = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ¹ψ_reference_nodes_alt)(x)
        #l = 2
        πₖ²_nodes_alt = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, binomial(2+2,2), SVector{2,Float64}})
        πₖ²ψ_reference_nodes_alt = Vector{SVector{3,Float64}}(undef, length(πₖ²_nodes_alt))
        for i in eachindex(πₖ²_nodes_alt)
            πₖ²ψ_reference_nodes_alt[i] = ψ_alt(f̂ₖ_alt(πₖ²_nodes_alt[i]))
        end
        πₖ²ψ_reference_nodes_alt = SVector{binomial(2+2,2)}(πₖ²ψ_reference_nodes_alt)
        πₖ²ψ_alt = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ²ψ_reference_nodes_alt)(x)

        # Nonlinear map
        if j == 3
            f̂ₖ_comp_alt = (x) -> f̂ₖ_alt( (x[1] * α₁hat_alt + x[2] * α₂hat_alt + x[3] * α₃hat_alt)/(x[1] + x[2] + x[3]) )
            Φₖ_θ1_alt = (x) -> ( (x[1] + x[2] + x[3])^3 * (ψ_alt(f̂ₖ_comp_alt(x)) - πₖ¹ψ_alt((x[1]*α₁hat_alt + x[2]*α₂hat_alt + x[3]*α₃hat_alt) / (x[1] + x[2] + x[3]))))
            Φₖ_θ2_alt = (x) -> ( (x[1] + x[2] + x[3])^4 * (ψ_alt(f̂ₖ_comp_alt(x)) - πₖ²ψ_alt((x[1]*α₁hat_alt + x[2]*α₂hat_alt + x[3]*α₃hat_alt) / (x[1] + x[2] + x[3]))) + (x[1] + x[2] + x[3])^2 * (πₖ²ψ_alt((x[1]*α₁hat_alt + x[2]*α₂hat_alt + x[3]*α₃hat_alt) / (x[1] + x[2] + x[3])) - πₖ¹ψ_alt((x[1]*α₁hat_alt + x[2]*α₂hat_alt + x[3]*α₃hat_alt) / (x[1] + x[2] + x[3]))))
        else
            f̂ₖ_comp_alt = (x) -> f̂ₖ_alt( (x[1] * α₁hat_alt + x[2] * α₂hat_alt)/(x[1] + x[2]) )
            Φₖ_θ1_alt = (x) -> ( (x[1] + x[2])^3 * (ψ_alt(f̂ₖ_comp_alt(x)) - πₖ¹ψ_alt((x[1]*α₁hat_alt + x[2]*α₂hat_alt) / (x[1] + x[2]))))
            Φₖ_θ2_alt = (x) -> ( (x[1] + x[2])^4 * (ψ_alt(f̂ₖ_comp_alt(x)) - πₖ²ψ_alt((x[1]*α₁hat_alt + x[2]*α₂hat_alt) / (x[1] + x[2]))) + (x[1] + x[2])^2 * (πₖ²ψ_alt((x[1]*α₁hat_alt + x[2]*α₂hat_alt) / (x[1] + x[2])) - πₖ¹ψ_alt((x[1]*α₁hat_alt + x[2]*α₂hat_alt) / (x[1] + x[2]))))
        end
        Φₖ_alt = Φₖ_θ2_alt

        # Full transformation
        atol = 10^(-12)
        Fₖ_alt = (x) -> F̃ₖ_alt(x) + Φₖ_alt(x)
        @assert norm(Fₖ_alt([1.0, 0.0, 0.0]) - a₁_alt) < atol
        @assert norm(Fₖ_alt([0.0, 1.0, 0.0]) - a₂_alt) < atol
        @assert norm(Fₖ_alt([1.0, 0.0, 0.0]) - aₖ_alt) < atol
        @assert norm(Fₖ_alt([0.0, 1.0, 0.0]) - bₖ_alt) < atol
        @assert norm(Fₖ_alt([0.0, 0.0000000000000001, 1.0]) - cₖ_alt) < atol
        @assert norm(Fₖ_alt([0.0, 0.0000000000000001, 0.0]) - dₖ_alt) < atol
        if j == 3
            @assert norm(a₃_alt - cₖ_alt) < atol
            @assert norm(Fₖ_alt([0.0, 0.0, 1.0]) - cₖ_alt) < atol
            @assert norm(Fₖ_alt([0.0, 0.0, 1.0]) - a₃_alt) < atol
            @assert abs(norm(Fₖ_alt([0.2, 0.3, 0.5])) - 1.0) < atol
            @assert abs(norm(Fₖ_alt([0.0, 0.0, 1.0])) - 1.0) < atol
            @assert norm(Φₖ_alt([0.0, 0.0, 0.3])) < atol
            @assert norm(Φₖ_alt([0.0, 0.3, 0.0])) < atol
            @assert norm(Φₖ_alt([0.3, 0.0, 0.0])) < atol
            @assert norm(Φₖ_alt([0.3, 0.45, 0.25]) - (ψ_alt(f̂ₖ_comp_alt([0.3, 0.45, 0.25])) - 0.3*a₁_alt - 0.45*a₂_alt - 0.25*a₃_alt)) < atol
            @assert norm(Φₖ_alt([0.55, 0.45, 0.0]) - (ψ_alt(f̂ₖ_comp_alt([0.55, 0.45, 0.0])) - 0.55*a₁_alt - 0.45*a₂_alt)) < atol
        end
        @assert abs(norm(Fₖ_alt([0.6, 0.4, 0.0])) - 1.0) < atol
        @assert abs(norm(Fₖ_alt([1.0, 0.0, 0.0])) - 1.0) < atol
        @assert abs(norm(Fₖ_alt([0.0, 1.0, 0.0])) - 1.0) < atol
        @assert norm(Φₖ_alt([0.0, 0.0000000000000001, 0.3])) < atol
        @assert norm(Φₖ_alt([0.0, 0.3, 0.0])) < atol
        @assert norm(Φₖ_alt([0.3, 0.0, 0.0])) < atol
        if j == 2
            @assert norm(Φₖ_alt([0.6, 0.0, 0.4])) < atol
            @assert norm(Φₖ_alt([0.0, 0.6, 0.4])) < atol
            @assert norm(Φₖ_alt([0.55, 0.45, 0.0]) - (ψ_alt(f̂ₖ_comp_alt([0.55, 0.45, 0.0])) - 0.55*a₁_alt - 0.45*a₂_alt)) < atol
        end
    
        Jₖ_alt = (x) -> transpose(ForwardDiff.jacobian(Fₖ_alt, x))
    
        Q = Inti.Gauss(; domain=Inti.ReferenceTetrahedron(), order=qorder)()
        #Q = Inti.VioreanuRokhlin(; domain=Inti.ReferenceTetrahedron(), order=qorder)()
        nq = length(Q[2])
        elarea = 0.0
        for q in 1:nq
            tmp = Q[2][q] * abs(det(Jₖ_alt(Q[1][q])))
            elarea += tmp
            #global elarea += tmp
        end
        #@assert elarea ≈ elvol[elind]
        #if j > 1 && !(elarea ≈ elvol[elind])
        #    println(elarea)
        #    println(elvol[elind])
        #    println(elind)
        #end
        if j > 1 && abs(elarea - elvol[elind]) < 10^(-9)
            println(elind)
        end
        #global elvol[elind] = elarea
        #global spharea += elarea
    end
end

h = 0.001
truearea = 4 * π / 3 - 2 * 1/3 * π * h^2 * (3 - h)