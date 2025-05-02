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
meshsize = .1 # .1 meshsize mesh has 1585 face nodes
qorder = 5

Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = meshsize)
Γ = Inti.external_boundary(Ω)
Γ_msh = view(msh, Γ)
msh_OG = deepcopy(msh)

θ = LinRange(0, π, 50*round(Int, 1/meshsize))
ϕ = LinRange(0, 2*π, 50*round(Int, 1/meshsize))
# v = (θ, ϕ)
ang = π/2
#ang = 0.0
M = [cos(ang); 0; sin(ang);; 0; 1; 0;; -1*sin(ang); 0; cos(ang)]
ψ₁ = (v) -> [sin(v[1]) * cos(v[2]), sin(v[1]) * sin(v[2]), cos(v[1])]
ψ₂ = (v) -> M * [sin(v[1]) * cos(v[2]), sin(v[1]) * sin(v[2]), cos(v[1])]

function chart_id(face_nodes)
    id = 1
    if all([abs(q[3]) for q in face_nodes] .> 0.55)
        id = 2
    end
    return id
end

function ψ₁⁻¹(v0, p)
    F₁ = (v, p) -> ψ₁(v) - p
    prob₁= NonlinearProblem(F₁, v0, p)
    ψ₁⁻¹ = NonlinearSolve.solve(prob₁, SimpleNewtonRaphson())
end

function ψ₂⁻¹(v0, p)
    F₂ = (v, p) -> ψ₂(v) - p
    prob₂= NonlinearProblem(F₂, v0, p)
    ψ₂⁻¹ = NonlinearSolve.solve(prob₂, SimpleNewtonRaphson())
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
    #elind = 498
    node_indices = deepcopy(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}][:, elind])
    nodes = deepcopy(msh.nodes[node_indices])

    if chart_id(nodes) == 1
        idxs, dists = nn(chart_1_kdt, nodes)
        if node_indices[1] ∉ chart_2_bdry_node_idx && node_indices[1] ∉ chart_1_bdry_node_idx
            msh.nodes[node_indices[1]] = chart_1[idxs[1]]
            push!(chart_1_bdry_node_idx, node_indices[1])
            push!(chart_1_bdry_node_param_loc, Vector{Float64}([θ[chart_1_cart_idxs_θ[idxs[1]]], ϕ[chart_1_cart_idxs_ϕ[idxs[1]]]]))
        end
        if node_indices[2] ∉ chart_2_bdry_node_idx && node_indices[2] ∉ chart_1_bdry_node_idx
            msh.nodes[node_indices[2]] = chart_1[idxs[2]]
            push!(chart_1_bdry_node_idx, node_indices[2])
            push!(chart_1_bdry_node_param_loc, Vector{Float64}([θ[chart_1_cart_idxs_θ[idxs[2]]], ϕ[chart_1_cart_idxs_ϕ[idxs[2]]]]))
        end
        if node_indices[3] ∉ chart_2_bdry_node_idx && node_indices[3] ∉ chart_1_bdry_node_idx
            msh.nodes[node_indices[3]] = chart_1[idxs[3]]
            push!(chart_1_bdry_node_idx, node_indices[3])
            push!(chart_1_bdry_node_param_loc, Vector{Float64}([θ[chart_1_cart_idxs_θ[idxs[3]]], ϕ[chart_1_cart_idxs_ϕ[idxs[3]]]]))
        end
    else
        #println(chart_id)
        idxs, dists = nn(chart_2_kdt, nodes)
        if node_indices[1] ∉ chart_1_bdry_node_idx && node_indices[1] ∉ chart_2_bdry_node_idx
            msh.nodes[node_indices[1]] = chart_2[idxs[1]]
            push!(chart_2_bdry_node_idx, node_indices[1])
            push!(chart_2_bdry_node_param_loc, Vector{Float64}([θ[chart_2_cart_idxs_θ[idxs[1]]], ϕ[chart_2_cart_idxs_ϕ[idxs[1]]]]))
        end
        if node_indices[2] ∉ chart_1_bdry_node_idx && node_indices[2] ∉ chart_2_bdry_node_idx
            msh.nodes[node_indices[2]] = chart_2[idxs[2]]
            push!(chart_2_bdry_node_idx, node_indices[2])
            push!(chart_2_bdry_node_param_loc, Vector{Float64}([θ[chart_2_cart_idxs_θ[idxs[2]]], ϕ[chart_2_cart_idxs_ϕ[idxs[2]]]]))
        end
        if node_indices[3] ∉ chart_1_bdry_node_idx && node_indices[3] ∉ chart_2_bdry_node_idx
            msh.nodes[node_indices[3]] = chart_2[idxs[3]]
            push!(chart_2_bdry_node_idx, node_indices[3])
            push!(chart_2_bdry_node_param_loc, Vector{Float64}([θ[chart_2_cart_idxs_θ[idxs[3]]], ϕ[chart_2_cart_idxs_ϕ[idxs[3]]]]))
        end
    end
end
#I = uniqueidx(chart_1_bdry_node_idx)
#chart_1_bdry_node_idx = chart_1_bdry_node_idx[I]
#chart_1_bdry_node_param_loc = chart_1_bdry_node_param_loc[I]
chart_1_node_to_param = Dict(zip(chart_1_bdry_node_idx, chart_1_bdry_node_param_loc))
#I = uniqueidx(chart_2_bdry_node_idx)
#chart_2_bdry_node_idx = chart_2_bdry_node_idx[I]
#chart_2_bdry_node_param_loc = chart_2_bdry_node_param_loc[I]
chart_2_node_to_param = Dict(zip(chart_2_bdry_node_idx, chart_2_bdry_node_param_loc))

nvol_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}}])[2]
spharea = 0.0
sfcarea = 0.0
sfcint = 0.0
nnewton = 0
elcount = 0
els = []
ncurv_vols = 0
for elind = 1:nvol_els
    node_indices = deepcopy(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}}][:, elind])
    nodes = deepcopy(msh.nodes[node_indices])
    
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
                α₂ = Vector{Float64}(ψ⁻¹(α₁, p))
                @assert norm(ψ(α₂) - p) < 10^(-14)
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
                res = ψ⁻¹(α₂, p)
                if res.retcode == ReturnCode.MaxIters
                    @assert false
                else
                    α₃ = Vector{Float64}(res.u)
                end
                @assert norm(ψ(α₃) - p) < 10^(-14)
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
                α₂ = Vector{Float64}(ψ⁻¹(α₁, p))
                @assert norm(ψ(α₂) - p) < 10^(-14)
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
                res = ψ⁻¹(α₂, p)
                if res.retcode == ReturnCode.MaxIters
                    @assert false
                else
                    α₃ = Vector{Float64}(res.u)
                end
                @assert norm(ψ(α₃) - p) < 10^(-14)
            end
        end
        #println(chart_num)
        #println(node_indices_on_bdry)
        atol = 10^(-4)
        @assert (norm(α₂ - α₃) > atol) && (norm(α₁ - α₃) > atol) && (norm(α₁ - α₂) > atol)
        # Try to handle periodicity in ϕ
        if (abs(α₂[2]) < atol) && (abs(α₂[2] - α₁[2]) > π || abs(α₂[2] - α₃[2]) > π)
            α₂[2] = 2*π
        end
        if (abs(α₃[2]) < atol) && (abs(α₃[2] - α₁[2]) > π || abs(α₃[2] - α₂[2]) > π)
            α₃[2] = 2*π
        end
        if (abs(α₁[2]) < atol) && (abs(α₁[2] - α₂[2]) > π || abs(α₁[2] - α₃[2]) > π)
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
        @assert ((abs(α₁[1] - α₂[1]) < π/8) && (abs(α₂[1] - α₃[1]) < π/8) && (abs(α₁[1] - α₃[1]) < π/8))
        #if !((abs(α₁[2] - α₂[2]) < π/8) && (abs(α₂[2] - α₃[2]) < π/8) && (abs(α₁[2] - α₃[2]) < π/8))
        #    @warn "Chart parametrization warning at element #", elind, " with ", j, "verts on bdry, at θ ≈ ", max(α₁[1], α₂[1], α₃[1])
        #    @warn "Chart parametrization warning at element #", elind, " with ", j, "verts on bdry, at θ ≈ ", min(α₁[1], α₂[1], α₃[1])
        #end
        @assert (α₂ ≠ α₃) && (α₁ ≠ α₃) && (α₁ ≠ α₂)
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
    
    Q = Inti.VioreanuRokhlin(; domain=Inti.ReferenceTetrahedron(), order=qorder)()
    nq = length(Q[2])
    elarea = 0.0
    for q in 1:nq
        tmp = Q[2][q] * abs(det(Jₖ(Q[1][q])))
        elarea += tmp
    end
    global spharea += elarea
        
    #els = [faceel1, faceel2, faceel3, faceel4]
    if j == 3
        qrule = Inti.VioreanuRokhlin(; domain=Inti.ReferenceTriangle(), order=qorder)
        etype2qrule = Dict(E => qrule for E in Inti.element_types(msh))
        quad = Inti.Quadrature{3,Float64}(msh, etype2qrule, Inti.QuadratureNode{3,Float64}[], Dict{DataType,Matrix{Int}}(),)
    
        D = Inti.ReferenceTriangle
        T = SVector{2,Float64}
        faceel1 = Inti.ParametricElement{D,T}(x -> Fₖ([x[1], x[2], 1 - x[1] - x[2]]))
        Inti._build_quadrature!(quad, [faceel1], qrule)
        faceel2 = Inti.ParametricElement{D,T}(x -> Fₖ([x[1], 0.0, x[2]]))
        Inti._build_quadrature!(quad, [faceel2], qrule)
        faceel3 = Inti.ParametricElement{D,T}(x -> Fₖ([0.0, x[1], x[2]]))
        Inti._build_quadrature!(quad, [faceel3], qrule)
        faceel4 = Inti.ParametricElement{D,T}(x -> Fₖ([x[1], x[2], 0.0]))
        Inti._build_quadrature!(quad, [faceel4], qrule)
    
        sfcint1 = 0.0
        sfcint2 = 0.0
        vecfield = (x) -> x
        sgnvec1 = [1, 1, -1, -1]
        sgnvec2 = [-1, -1, 1, 1]
        for qind in eachindex(quad)
            nq = length(qrule()[1])
            sgn1 = sgnvec1[div(qind-1, nq) + 1]
            sgn2 = sgnvec2[div(qind-1, nq) + 1]
            sfcint1 += dot(vecfield(quad[qind].coords), sgn1*quad[qind].normal) * quad[qind].weight
            sfcint2 += dot(vecfield(quad[qind].coords), sgn2*quad[qind].normal) * quad[qind].weight
            # Only add contributions from faceel1, the face on the surface of sphere
            if qind <= nq
                global sfcarea += quad[qind].weight
            end
        end
        if sfcint1 > 0
            @assert sfcint1 > 0
            global sfcint += sfcint1
        else
            @assert sfcint2 > 0
            global sfcint += sfcint2
        end
    end

    #A = faceel1([0.000000000001, 0.0])
    #B = faceel1([1.0, 0.0])
    #C = faceel1([0.000000000000000001, 1.0])
    #norm(cross(B - A, C - A))/2 - (quad[1].weight + quad[2].weight + quad[3].weight)
    #A = faceel2([0.000000000001, 0.0])
    #B = faceel2([1.0, 0.0])
    #C = faceel2([0.000000000000000001, 1.0])
    #norm(cross(B - A, C - A))/2 - (quad[4].weight + quad[5].weight + quad[6].weight)
    #A = faceel3([0.000000000001, 0.0])
    #B = faceel3([1.0, 0.0])
    #C = faceel3([0.000000000000000001, 1.0])
    #norm(cross(B - A, C - A))/2 - (quad[7].weight + quad[8].weight + quad[9].weight)
    #A = faceel4([0.000000000001, 0.0])
    #B = faceel4([1.0, 0.0])
    #C = faceel4([0.000000000000000001, 1.0])
    #norm(cross(B - A, C - A))/2 - (quad[10].weight + quad[11].weight + quad[12].weight)
end