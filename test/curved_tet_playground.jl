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
qorder = 2

Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = meshsize)
Γ = Inti.external_boundary(Ω)
Γ_msh = view(msh, Γ)

θ = LinRange(0, π, 50*Int(1/meshsize))
ϕ = LinRange(0, 2*π, 50*Int(1/meshsize))
# v = (θ, ϕ)
ang = π/2
#ang = 0.0
M = [cos(ang); 0; sin(ang);; 0; 1; 0;; -1*sin(ang); 0; cos(ang)]
ψ₁ = (v) -> [sin(v[1]) * cos(v[2]), sin(v[1]) * sin(v[2]), cos(v[1])]
ψ₂ = (v) -> M * [sin(v[1]) * cos(v[2]), sin(v[1]) * sin(v[2]), cos(v[1])]

function chart_id(face_nodes)
    chart_id = 1
    if all([abs(q[3]) for q in face_nodes] .> 0.75)
        chart_id = 2
    end
    return chart_id
end

function ψ₁inv(v0, p)
    F₁ = (v, p) -> ψ₁(v) - p
    prob₁= NonlinearProblem(F₁, v0, p)
    ψ₁inv = NonlinearSolve.solve(prob₁, SimpleNewtonRaphson())
end

function ψ₂inv(v0, p)
    F₂ = (v, p) -> ψ₂(v) - p
    prob₂= NonlinearProblem(F₂, v0, p)
    ψ₂inv = NonlinearSolve.solve(prob₂, SimpleNewtonRaphson())
end

chart_1 = Array{SVector{3,Float64}}(undef, length(θ)*length(ϕ))
chart_1_cart_idxs_θ = []
chart_1_cart_idxs_ϕ = []
for i in eachindex(θ)
    for j in eachindex(ϕ)
        chart_1[(i-1)*length(ϕ) + j] = [k for k in ψ₁((θ[i], ϕ[j]))]
        push!(chart_1_cart_idxs_θ, i)
        push!(chart_1_cart_idxs_ϕ, j)
    end
end
chart_1_kdt = KDTree(chart_1; reorder = false)
# chart 2
chart_2 = Array{SVector{3,Float64}}(undef, length(θ)*length(ϕ))
chart_2_cart_idxs_θ = []
chart_2_cart_idxs_ϕ = []
for i in eachindex(θ)
    for j in eachindex(ϕ)
        chart_2[(i-1)*length(ϕ) + j] = [k for k in ψ₂((θ[i], ϕ[j]))]
        push!(chart_2_cart_idxs_θ, i)
        push!(chart_2_cart_idxs_ϕ, j)
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

# Set up chart 1
for elind = 1:nbdry_els
    node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}][:, elind]
    nodes = msh.nodes[node_indices]

    if chart_id(nodes) == 1
        idxs, dists = nn(chart_1_kdt, nodes)
        if node_indices[1] ∉ chart_2_bdry_node_idx
            msh.nodes[node_indices[1]] = chart_1[idxs[1]]
            push!(chart_1_bdry_node_idx, node_indices[1])
            push!(chart_1_bdry_node_param_loc, Vector{Float64}([θ[chart_1_cart_idxs_θ[idxs[1]]], ϕ[chart_1_cart_idxs_ϕ[idxs[1]]]]))
        end
        if node_indices[2] ∉ chart_2_bdry_node_idx
            msh.nodes[node_indices[2]] = chart_1[idxs[2]]
            push!(chart_1_bdry_node_idx, node_indices[2])
            push!(chart_1_bdry_node_param_loc, Vector{Float64}([θ[chart_1_cart_idxs_θ[idxs[2]]], ϕ[chart_1_cart_idxs_ϕ[idxs[2]]]]))
        end
        if node_indices[3] ∉ chart_2_bdry_node_idx
            msh.nodes[node_indices[3]] = chart_1[idxs[3]]
            push!(chart_1_bdry_node_idx, node_indices[3])
            push!(chart_1_bdry_node_param_loc, Vector{Float64}([θ[chart_1_cart_idxs_θ[idxs[3]]], ϕ[chart_1_cart_idxs_ϕ[idxs[3]]]]))
        end
    else
        idxs, dists = nn(chart_2_kdt, nodes)
        if node_indices[1] ∉ chart_1_bdry_node_idx
            msh.nodes[node_indices[1]] = chart_2[idxs[1]]
            push!(chart_2_bdry_node_idx, node_indices[1])
            push!(chart_2_bdry_node_param_loc, Vector{Float64}([θ[chart_2_cart_idxs_θ[idxs[1]]], ϕ[chart_2_cart_idxs_ϕ[idxs[1]]]]))
        end
        if node_indices[2] ∉ chart_1_bdry_node_idx
            msh.nodes[node_indices[2]] = chart_2[idxs[2]]
            push!(chart_2_bdry_node_idx, node_indices[2])
            push!(chart_2_bdry_node_param_loc, Vector{Float64}([θ[chart_2_cart_idxs_θ[idxs[2]]], ϕ[chart_2_cart_idxs_ϕ[idxs[2]]]]))
        end
        if node_indices[3] ∉ chart_1_bdry_node_idx
            msh.nodes[node_indices[3]] = chart_2[idxs[3]]
            push!(chart_2_bdry_node_idx, node_indices[3])
            push!(chart_2_bdry_node_param_loc, Vector{Float64}([θ[chart_2_cart_idxs_θ[idxs[3]]], ϕ[chart_2_cart_idxs_ϕ[idxs[3]]]]))
        end
    end
end
I = uniqueidx(chart_1_bdry_node_idx)
chart_1_bdry_node_idx = chart_1_bdry_node_idx[I]
chart_1_bdry_node_param_loc = chart_1_bdry_node_param_loc[I]
chart_1_node_to_param = Dict(zip(chart_1_bdry_node_idx, chart_1_bdry_node_param_loc))
I = uniqueidx(chart_2_bdry_node_idx)
chart_2_bdry_node_idx = chart_2_bdry_node_idx[I]
chart_2_bdry_node_param_loc = chart_2_bdry_node_param_loc[I]
chart_2_node_to_param = Dict(zip(chart_2_bdry_node_idx, chart_2_bdry_node_param_loc))

nvol_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}}])[2]
spharea = 0.0
nnewton = 0
els = []
for elind = 1:nvol_els
#elind = 1 # 0 verts on bdry meshsize .1
#elind = 26
#elind = 148
#elind = 189
#elind = 267
#elind = 480
#elind = 737
#elind = 1356
#elind = 4162
#elind = 1737
#elind = 13022
#elind = 13783
#elind = 18300
#elind = 19507
#elind = 19660 # α straddling 2π
#elind = 19986
#elind = 20122
#elind = 20396
#elind = 20398
#elind = 20402 # 2 verts on bdry meshsize .1
#elind = 20386 # 3 verts on bdry meshsize .1
#elind = 20396
#elind = 152422 # 3 verts on bdry meshsize .1/2
#elind = 98351 # 2 verts on bdry meshsize .1/2
    node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}}][:, elind]
    nodes = msh.nodes[node_indices]
    
    verts_on_bdry_chart_1 = findall(x -> x ∈ chart_1_bdry_node_idx, node_indices)
    verts_on_bdry_chart_2 = findall(x -> x ∈ chart_2_bdry_node_idx, node_indices)
    j = 0
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
        node_indices_on_bdry = copy(node_indices[verts_on_bdry])
        #chart_num = chart_id(msh.nodes[node_indices_on_bdry])
        if chart_num == 1
            node_to_param = chart_1_node_to_param
            ψ = ψ₁
            verts_on_bdry = verts_on_bdry_chart_1
            α₁ = copy(node_to_param[node_indices_on_bdry[1]])
            if nverts_in_major_chart >= 2
                α₂ = copy(node_to_param[node_indices_on_bdry[2]])
            else
                push!(node_indices_on_bdry, node_indices[verts_on_bdry_chart_2[1]])
                p = copy(msh.nodes[node_indices[verts_on_bdry_chart_2[1]]])
                global nnewton += 1
                α₂ = ψ₁inv(α₁, p)
            end
            if nverts_in_major_chart >= 3
                α₃ = copy(node_to_param[node_indices_on_bdry[3]])
            else
                # Find missing node α₃ that (non-uniquely) defines the curved face simplex containing α₁, α₂
                candidate_els = Inti.elements_containing_nodes(n2e, node_indices_on_bdry)
                # Filter out volume elements; should be at most two face simplices remaining
                candidate_els = candidate_els[length.(candidate_els).==3]
                # Take the first face simplex; while either would work if j=2,
                # if j=3 only one of the candidate face triangles will work, so
                # find that one
                if candidate_els[1][1] ∉ node_indices_on_bdry
                    p = copy(msh.nodes[candidate_els[1][1]])
                elseif candidate_els[1][2] ∉ node_indices_on_bdry
                    p = copy(msh.nodes[candidate_els[1][2]])
                elseif candidate_els[1][3] ∉ node_indices_on_bdry
                    p = copy(msh.nodes[candidate_els[1][3]])
                else
                    @assert false
                end
                if j == 3 && p ∉ nodes
                    if candidate_els[2][1] ∉ node_indices_on_bdry
                        p = copy(msh.nodes[candidate_els[2][1]])
                    elseif candidate_els[2][2] ∉ node_indices_on_bdry
                        p = copy(msh.nodes[candidate_els[2][2]])
                    elseif candidate_els[2][3] ∉ node_indices_on_bdry
                        p = copy(msh.nodes[candidate_els[2][3]])
                    else
                        @assert false
                    end
                end
                global nnewton += 1
                α₃ = ψ₁inv(α₂, p)
            end
        else
            node_to_param = chart_2_node_to_param
            ψ = ψ₂
            verts_on_bdry = verts_on_bdry_chart_2
            α₁ = copy(node_to_param[node_indices_on_bdry[1]])
            if nverts_in_major_chart >= 2
                α₂ = copy(node_to_param[node_indices_on_bdry[2]])
            else
                # minor chart is chart 1, can be improved and streamlined
                push!(node_indices_on_bdry, node_indices[verts_on_bdry_chart_1[1]])
                p = copy(msh.nodes[node_indices[verts_on_bdry_chart_1[1]]])
                #global nnewton += 1
                α₂ = ψ₂inv(α₁, p)
            end
            if nverts_in_major_chart >= 3
                α₃ = copy(node_to_param[node_indices_on_bdry[3]])
            else
                # Find missing node α₃ that (non-uniquely) defines the curved face simplex containing α₁, α₂
                candidate_els = Inti.elements_containing_nodes(n2e, node_indices_on_bdry)
                # Filter out volume elements; should be at most two face simplices remaining
                candidate_els = candidate_els[length.(candidate_els).==3]
                # Take the first face simplex; while either would work if j=2,
                # if j=3 only one of the candidate face triangles will work, so
                # find that one
                if candidate_els[1][1] ∉ node_indices_on_bdry
                    p = copy(msh.nodes[candidate_els[1][1]])
                elseif candidate_els[1][2] ∉ node_indices_on_bdry
                    p = copy(msh.nodes[candidate_els[1][2]])
                elseif candidate_els[1][3] ∉ node_indices_on_bdry
                    p = copy(msh.nodes[candidate_els[1][3]])
                else
                    @assert false
                end
                if j == 3 && p ∉ nodes
                    if candidate_els[2][1] ∉ node_indices_on_bdry
                        p = copy(msh.nodes[candidate_els[2][1]])
                    elseif candidate_els[2][2] ∉ node_indices_on_bdry
                        p = copy(msh.nodes[candidate_els[2][2]])
                    elseif candidate_els[2][3] ∉ node_indices_on_bdry
                        p = copy(msh.nodes[candidate_els[2][3]])
                    else
                        @assert false
                    end
                end
                #global nnewton += 1
                α₃ = ψ₂inv(α₂, p)
            end
        end
        #println(chart_num)
        #println(node_indices_on_bdry)
        @assert (α₂ ≠ α₃) && (α₁ ≠ α₃) && (α₁ ≠ α₂)
        # Try to handle periodicity in ϕ
        if (α₂[2] ≈ 0.0) && (abs(α₂[2] - α₁[2]) > π || abs(α₂[2] - α₃[2]) > π)
            α₂[2] = 2*π
        end
        if (α₃[2] ≈ 0.0) && (abs(α₃[2] - α₁[2]) > π || abs(α₃[2] - α₂[2]) > π)
            α₃[2] = 2*π
        end
        if (α₁[2] ≈ 0.0) && (abs(α₁[2] - α₂[2]) > π || abs(α₁[2] - α₃[2]) > π)
            α₁[2] = 2*π
        end
        if (α₂[2] ≈ 2*π) && (abs(α₂[2] - α₁[2]) > π || abs(α₂[2] - α₃[2]) > π)
            α₂[2] = 0.0
        end
        if (α₃[2] ≈ 2*π) && (abs(α₃[2] - α₁[2]) > π || abs(α₃[2] - α₂[2]) > π)
            α₃[2] = 0.0
        end
        if (α₁[2] ≈ 2*π) && (abs(α₁[2] - α₂[2]) > π || abs(α₁[2] - α₃[2]) > π)
            α₁[2] = 0.0
        end

        # Try to handle periodicity in ϕ -- case of α straddling 2π
        if (abs(α₁[2] - α₂[2]) > π) || (abs(α₂[2] - α₃[2]) > π) || (abs(α₁[2] - α₃[2]) > π)
            if α₁[2] < π && α₂[2] < π && α₃[2] > 2*(2*π)/3
                α₃[2] -= 2*π
            end
            if α₂[2] < π && α₃[2] < π && α₁[2] > 2*(2*π)/3
                α₁[2] -= 2*π
            end
            if α₁[2] < π && α₃[2] < π && α₂[2] > 2*(2*π)/3
                α₂[2] -= 2*π
            end
            #println(elind)
            #println([α₁, α₂, α₃])
        end
        if (abs(α₁[2] - α₂[2]) > π) || (abs(α₂[2] - α₃[2]) > π) || (abs(α₁[2] - α₃[2]) > π)
            if α₁[2] > 2*(2*π)/3 && α₂[2] > 2*(2*π)/3 && α₃[2] < π
                α₃[2] += 2*π
            end
            if α₂[2] > 2*(2*π)/3 && α₃[2] > 2*(2*π)/3 && α₁[2] < π
                α₁[2] += 2*π
            end
            if α₁[2] > 2*(2*π)/3 && α₃[2] > 2*(2*π)/3 && α₂[2] < π
                α₂[2] += 2*π
            end
            #println(elind)
            #println([α₁, α₂, α₃])
        end
        if !((abs(α₁[2] - α₂[2]) < π/4) && (abs(α₂[2] - α₃[2]) < π/4) && (abs(α₁[2] - α₃[2]) < π/4))
            @warn "Chart parametrization warning at element #", elind, " with ", j, "verts on bdry, at θ ≈ ", max(α₁[1], α₂[1], α₃[1])
            @warn "Chart parametrization warning at element #", elind, " with ", j, "verts on bdry, at θ ≈ ", min(α₁[1], α₂[1], α₃[1])
        end
        @assert (α₂ ≠ α₃) && (α₁ ≠ α₃) && (α₁ ≠ α₂)
        a₁ = SVector{3,Float64}(ψ(α₁))
        a₂ = SVector{3,Float64}(ψ(α₂))
        a₃ = SVector{3,Float64}(ψ(α₃))
        α₁hat = SVector{2,Float64}(0.0, 0.0)
        α₂hat = SVector{2,Float64}(1.0, 0.0)
        α₃hat = SVector{2,Float64}(0.0, 1.0)
        f̂ₖ = (x) -> [(α₂[1] - α₁[1])*x[1] + (α₃[1] - α₁[1])*x[2] + α₁[1], (α₂[2] - α₁[2])*x[1] + (α₃[2] - α₁[2])*x[2] + α₁[2]]
        #println(elind)
        @assert (f̂ₖ(α₁hat) ≈ α₁) && (f̂ₖ(α₂hat) ≈ α₂) && (f̂ₖ(α₃hat) ≈ α₃)
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
                #(skipnode == 1) || (println("cₖ assigned", 1))
            end
            if all(norm.(Ref(nodes[2]) .- facenodes) .> atol)
                (skipnode == 2) || (cₖ = copy(nodes[2]))
                #(skipnode == 2) || (println("cₖ assigned", 2))
            end
            if all(norm.(Ref(nodes[3]) .- facenodes) .> atol)
                (skipnode == 3) || (cₖ = copy(nodes[3]))
                #(skipnode == 3) || (println("cₖ assigned", 3))
            end
            if all(norm.(Ref(nodes[4]) .- facenodes) .> atol)
                (skipnode == 4) || (cₖ = copy(nodes[4]))
                #(skipnode == 4) || (println("cₖ assigned", 4))
            end
            @assert !(cₖ ≈ a₃)
        end
        F̃ₖ = (x) -> [(aₖ[1] - dₖ[1])*x[1] + (bₖ[1] - dₖ[1])*x[2] + (cₖ[1] - dₖ[1])*x[3] + dₖ[1], (aₖ[2] - dₖ[2])*x[1] + (bₖ[2] - dₖ[2])*x[2] + (cₖ[2] - dₖ[2])*x[3] + dₖ[2], (aₖ[3] - dₖ[3])*x[1] + (bₖ[3] - dₖ[3])*x[2] + (cₖ[3] - dₖ[3])*x[3] + dₖ[3]]

        # l = 1
        πₖ¹_nodes = Inti.reference_nodes(Inti.LagrangeElement{Inti.ReferenceTriangle, 3, SVector{2,Float64}})
        πₖ¹ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ¹_nodes))
        for i in eachindex(πₖ¹_nodes)
            πₖ¹ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ¹_nodes[i]))
        end
        πₖ¹ψ_reference_nodes = SVector{3}(πₖ¹ψ_reference_nodes)
        πₖ¹ψ = (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ¹ψ_reference_nodes)(x)
        #l = 2
        # ...

        # l = 1
        # Nonlinear map
        if j == 3
            f̂ₖ_comp = (x) -> f̂ₖ( (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat)/(x[1] + x[2] + x[3]) )
            Φₖ = (x) -> ( (x[1] + x[2] + x[3])^3 * (ψ(f̂ₖ_comp(x)) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3])))) 
        else
            f̂ₖ_comp = (x) -> f̂ₖ( (x[1] * α₁hat + x[2] * α₂hat)/(x[1] + x[2]) )
            Φₖ = (x) -> ( (x[1] + x[2])^3 * (ψ(f̂ₖ_comp(x)) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))))
        end

        #if j == 3 && nverts_in_major_chart == 2
        #    println(elind)
        #    println(a₁)
        #    println(a₂)
        #    println(a₃)
        #end
        #println("==========")

        # Full transformation
        Fₖ = (x) -> F̃ₖ(x) + Φₖ(x)
        JF̃ₖ = (x) -> [aₖ[1] - bₖ[1]; cₖ[1] - bₖ[1]; dₖ[1] - bₖ[1];; aₖ[2] - bₖ[2]; cₖ[2] - bₖ[2]; dₖ[2] - bₖ[2];; aₖ[3] - bₖ[3]; cₖ[3] - bₖ[3]; dₖ[3] - bₖ[3]]
        Jₖ = (x) -> JF̃ₖ(x) + ForwardDiff.jacobian(Φₖ, x)
        #if (elind == 4162) || (elind == 10304) || (elind == 10304) || (elind == 13783) || (elind == 18300) || (elind == 19505) || (elind == 19507)
        #    Jₖ = (x) -> JF̃ₖ(x)
        #end
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
        tmp = Q[2][q] * abs(det(Jₖ(Q[1][q])))
        if tmp > .0004
            println(elind)
            println(q)
            println(tmp)
            #error("large integral value")
        end
        spharea += tmp
        #spharea += Q[2][q] * abs(det(Jₖ(Q[1][q])))
    end
end