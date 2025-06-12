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
r1 = 1.0
r2 = 0.5

Ω, msh = gmsh_torus(; center = [0.0, 0.0, 0.0], r1 = r1, r2 = r2, meshsize = meshsize)
Γ = Inti.external_boundary(Ω)
Γ_msh = view(msh, Γ)

θ = LinRange(0, 2*π, 50*round(Int, 1/meshsize))
ϕ = LinRange(0, 2*π, 50*round(Int, 1/meshsize))
# v = (θ, ϕ)
ψ₁ = (v) -> [(r1 + r2*sin(v[1]))*cos(v[2]), (r1 + r2*sin(v[1]))*sin(v[2]), r2*cos(v[1])]

function ψ₁⁻¹(v0, p)
    F₁ = (v, p) -> ψ₁(v) - p
    prob₁= NonlinearProblem(F₁, v0, p)
    ψ₁⁻¹ = NonlinearSolve.solve(prob₁, SimpleNewtonRaphson())
end

face_element_on_torus(nodelist, R, r) = all([(sqrt(node[1]^2 + node[2]^2) - R^2)^2 + node[3]^2 ≈ r^2 for node in nodelist])

face_element_on_surface = (nodelist) -> face_element_on_torus(nodelist, r1, r2)

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

n2e = Inti.node2etags(msh)

nbdry_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}])[2]
chart_1_bdry_node_idx = Vector{Int64}()
chart_1_bdry_node_param_loc = Vector{Vector{Float64}}()

uniqueidx(v) = unique(i -> v[i], eachindex(v))

crvmsh = Inti.Mesh{3,Float64}()
(; nodes, etype2mat, etype2els, ent2etags) = crvmsh
foreach(k -> ent2etags[k] = Dict{DataType,Vector{Int}}(), Inti.entities(msh))
append!(nodes, msh.nodes)

# Set up chart <-> node Dict
for elind = 1:nbdry_els
    node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}][:, elind]
    straight_nodes = crvmsh.nodes[node_indices]

    if face_element_on_surface(straight_nodes)
        idxs, dists = nn(chart_1_kdt, straight_nodes)
        ψ⁻¹ = ψ₁⁻¹
        if node_indices[1] ∉ chart_1_bdry_node_idx
            crvmsh.nodes[node_indices[1]] = chart_1[idxs[1]]
            push!(chart_1_bdry_node_param_loc, Vector{Float64}([θ[chart_1_cart_idxs_θ[idxs[1]]], ϕ[chart_1_cart_idxs_ϕ[idxs[1]]]]))
            push!(chart_1_bdry_node_idx, node_indices[1])
        end
        if node_indices[2] ∉ chart_1_bdry_node_idx
            crvmsh.nodes[node_indices[2]] = chart_1[idxs[2]]
            push!(chart_1_bdry_node_param_loc, Vector{Float64}([θ[chart_1_cart_idxs_θ[idxs[2]]], ϕ[chart_1_cart_idxs_ϕ[idxs[2]]]]))
            push!(chart_1_bdry_node_idx, node_indices[2])
        end
        if node_indices[3] ∉ chart_1_bdry_node_idx
            crvmsh.nodes[node_indices[3]] = chart_1[idxs[3]]
            push!(chart_1_bdry_node_param_loc, Vector{Float64}([θ[chart_1_cart_idxs_θ[idxs[3]]], ϕ[chart_1_cart_idxs_ϕ[idxs[3]]]]))
            push!(chart_1_bdry_node_idx, node_indices[3])
        end
    end
end
chart_1_node_to_param = Dict(zip(chart_1_bdry_node_idx, chart_1_bdry_node_param_loc))

connect_straight = Int[]
connect_curve = Int[]
connect_curve_bdry = Int[]
# TODO Could use an ElementIterator for straight elements
els_straight = []
els_curve = []
els_curve_bdry = []

for E in Inti.element_types(msh)
    # The purpose of this check is to see if other element types are present in
    # the mesh, such as e.g. cubes; This code errors when encountering a cube,
    # but the method can be extended to transfer straight cubes to the new mesh,
    # similar to how straight simplices are transferred below.
    E <: Union{Inti.LagrangeElement{Inti.ReferenceSimplex{3}},Inti.LagrangeElement{Inti.ReferenceSimplex{2}},
    Inti.LagrangeElement{Inti.ReferenceHyperCube{1}}, SVector} || error()
    E <: SVector && continue
    E <: Inti.LagrangeElement{Inti.ReferenceHyperCube{1}} && continue
    E <: Inti.LagrangeElement{Inti.ReferenceHyperCube{2}} && continue
    E <: Inti.LagrangeElement{Inti.ReferenceSimplex{2}} && continue
    E <: Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}} || (println(E); error())
    E_straight_bdry = Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}
    els = Inti.elements(msh, E)
    for elind in eachindex(els)
        node_indices = msh.etype2mat[E][:, elind]
        straight_nodes = crvmsh.nodes[node_indices]
    
        verts_on_bdry = findall(x -> x ∈ chart_1_bdry_node_idx, node_indices)
        # j in C. Bernardi SINUM Sec. 6
        j = 0
        if !isempty(verts_on_bdry)
            nverts_in_chart = length(verts_on_bdry)
            j = nverts_in_chart
        end
        if j > 1
            append!(connect_curve, node_indices)
            node_indices_on_bdry = copy(node_indices[verts_on_bdry])
            append!(connect_curve_bdry, node_indices_on_bdry)
            node_to_param = chart_1_node_to_param
            ψ = ψ₁
            ψ⁻¹ = ψ₁⁻¹
            α₁ = copy(node_to_param[node_indices_on_bdry[1]])
            if nverts_in_chart >= 2
                α₂ = copy(node_to_param[node_indices_on_bdry[2]])
            else
                @assert false
            end
            if nverts_in_chart >= 3
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
                    p = copy(crvmsh.nodes[candidate_els[1][1]])
                elseif candidate_els[1][2] ∉ node_indices_on_bdry
                    p = copy(crvmsh.nodes[candidate_els[1][2]])
                elseif candidate_els[1][3] ∉ node_indices_on_bdry
                    p = copy(crvmsh.nodes[candidate_els[1][3]])
                else
                    @assert false
                end
                if j == 3 && p ∉ straight_nodes
                    if candidate_els[2][1] ∉ node_indices_on_bdry
                        p = copy(crvmsh.nodes[candidate_els[2][1]])
                    elseif candidate_els[2][2] ∉ node_indices_on_bdry
                        p = copy(crvmsh.nodes[candidate_els[2][2]])
                    elseif candidate_els[2][3] ∉ node_indices_on_bdry
                        p = copy(crvmsh.nodes[candidate_els[2][3]])
                    else
                        @assert false
                    end
                end
                res = ψ⁻¹(α₂, p)
                if  res.retcode == ReturnCode.MaxIters
                    α₃ = copy(α₂)
                    @assert j == 2
                else
                    α₃ = Vector{Float64}(ψ⁻¹(α₂, p))
                    @assert norm(ψ(α₃) - p) < 10^(-14)
                end
            end
            atol = 10^(-4)
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

            # Try to handle periodicity in ϕ
            if (abs(α₂[1]) < atol) && (abs(α₂[1] - α₁[1]) > π || abs(α₂[1] - α₃[1]) > π)
                α₂[1] = 2*π
            end
            if (abs(α₃[1]) < atol) && (abs(α₃[1] - α₁[1]) > π || abs(α₃[1] - α₂[1]) > π)
                α₃[1] = 2*π
            end
            if (abs(α₁[1]) < atol) && (abs(α₁[1] - α₂[1]) > π || abs(α₁[1] - α₃[1]) > π)
                α₁[1] = 2*π
            end
            if (α₂[1] ≈ 2*π) && (abs(α₂[1] - α₁[1]) > π || abs(α₂[1] - α₃[1]) > π)
                α₂[1] = 0.0
            end
            if (α₃[1] ≈ 2*π) && (abs(α₃[1] - α₁[1]) > π || abs(α₃[1] - α₂[1]) > π)
                α₃[1] = 0.0
            end
            if (α₁[1] ≈ 2*π) && (abs(α₁[1] - α₂[1]) > π || abs(α₁[1] - α₃[1]) > π)
                α₁[1] = 0.0
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
            end
            # Try to handle periodicity in ϕ -- case of α straddling 2π
            if (abs(α₁[1] - α₂[1]) > π) || (abs(α₂[1] - α₃[1]) > π) || (abs(α₁[1] - α₃[1]) > π)
                if α₁[1] < π && α₂[1] < π && α₃[1] > 2*(2*π)/3
                    α₃[1] -= 2*π
                end
                if α₂[1] < π && α₃[1] < π && α₁[1] > 2*(2*π)/3
                    α₁[1] -= 2*π
                end
                if α₁[1] < π && α₃[1] < π && α₂[1] > 2*(2*π)/3
                    α₂[1] -= 2*π
                end
            end
            if (abs(α₁[1] - α₂[1]) > π) || (abs(α₂[1] - α₃[1]) > π) || (abs(α₁[1] - α₃[1]) > π)
                if α₁[1] > 2*(2*π)/3 && α₂[1] > 2*(2*π)/3 && α₃[1] < π
                    α₃[1] += 2*π
                end
                if α₂[1] > 2*(2*π)/3 && α₃[1] > 2*(2*π)/3 && α₁[1] < π
                    α₁[1] += 2*π
                end
                if α₁[1] > 2*(2*π)/3 && α₃[1] > 2*(2*π)/3 && α₂[1] < π
                    α₂[1] += 2*π
                end
            end
            @assert ((abs(α₁[1] - α₂[1]) < π/8) && (abs(α₂[1] - α₃[1]) < π/8) && (abs(α₁[1] - α₃[1]) < π/8))
            if !((abs(α₁[2] - α₂[2]) < π/8) && (abs(α₂[2] - α₃[2]) < π/8) && (abs(α₁[2] - α₃[2]) < π/8))
                @warn "Chart parametrization warning at element #", elind, " with ", j, "verts on bdry, at θ ≈ ", max(α₁[1], α₂[1], α₃[1])
            end
            a₁ = SVector{3,Float64}(ψ(α₁))
            a₂ = SVector{3,Float64}(ψ(α₂))
            a₃ = SVector{3,Float64}(ψ(α₃))

            # Construction of the affine map with vertices (aₖ, bₖ, cₖ, dₖ).
            # Vertices aₖ and bₖ always lay on surface. Vertex dₖ always lays in volume.
            aₖ = a₁
            bₖ = a₂
            cₖ = a₃
            atol = 10^-12
            facenodes = [a₁, a₂, a₃]
            skipnode = 0
            if all(norm.(Ref(straight_nodes[1]) .- facenodes) .> atol)
                dₖ = straight_nodes[1]
                skipnode = 1
            elseif all(norm.(Ref(straight_nodes[2]) .- facenodes) .> atol)
                dₖ = straight_nodes[2]
                skipnode = 2
            elseif all(norm.(Ref(straight_nodes[3]) .- facenodes) .> atol)
                dₖ = straight_nodes[3]
                skipnode = 3
            elseif all(norm.(Ref(straight_nodes[4]) .- facenodes) .> atol)
                dₖ = straight_nodes[4]
                skipnode = 4
            else
                error("Uhoh")
            end
            if j == 2
                if all(norm.(Ref(straight_nodes[1]) .- facenodes) .> atol)
                    (skipnode == 1) || (cₖ = copy(straight_nodes[1]))
                end
                if all(norm.(Ref(straight_nodes[2]) .- facenodes) .> atol)
                    (skipnode == 2) || (cₖ = copy(straight_nodes[2]))
                end
                if all(norm.(Ref(straight_nodes[3]) .- facenodes) .> atol)
                    (skipnode == 3) || (cₖ = copy(straight_nodes[3]))
                end
                if all(norm.(Ref(straight_nodes[4]) .- facenodes) .> atol)
                    (skipnode == 4) || (cₖ = copy(straight_nodes[4]))
                end
                @assert norm(cₖ - a₃) > atol
                @assert norm(cₖ - dₖ) > atol
                @assert norm(dₖ - a₁) > atol
                @assert norm(dₖ - a₂) > atol
            end
            @assert !all(norm.(Ref(a₁) .- straight_nodes) .> atol)
            @assert !all(norm.(Ref(a₂) .- straight_nodes) .> atol)
            if j == 3
                @assert !all(norm.(Ref(a₃) .- straight_nodes) .> atol)
            end

            # The following ensures an ordering of the face nodes so that
            # the resulting normal vector is properly oriented.
            if det([aₖ-dₖ bₖ-dₖ cₖ-dₖ]) < 0
                tmp = deepcopy(α₁)
                α₁ = deepcopy(α₂)
                α₂ = tmp
                a₁ = SVector{3,Float64}(ψ(α₁))
                a₂ = SVector{3,Float64}(ψ(α₂))
                a₃ = SVector{3,Float64}(ψ(α₃))
                aₖ = a₁
                bₖ = a₂
            end

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
            @assert a₁ ≈ straight_nodes[1] || a₁ ≈ straight_nodes[2] || a₁ ≈ straight_nodes[3] || a₁ ≈ straight_nodes[4]
            @assert a₂ ≈ straight_nodes[1] || a₂ ≈ straight_nodes[2] || a₂ ≈ straight_nodes[3] || a₂ ≈ straight_nodes[4]
            if j == 3
                @assert a₃ ≈ straight_nodes[1] || a₃ ≈ straight_nodes[2] || a₃ ≈ straight_nodes[3] || a₃ ≈ straight_nodes[4]
            end
            @assert aₖ ≈ a₁
            @assert bₖ ≈ a₂
            F̃ₖ = (x) -> [(aₖ[1] - dₖ[1])*x[1] + (bₖ[1] - dₖ[1])*x[2] + (cₖ[1] - dₖ[1])*x[3] + dₖ[1], (aₖ[2] - dₖ[2])*x[1] + (bₖ[2] - dₖ[2])*x[2] + (cₖ[2] - dₖ[2])*x[3] + dₖ[2], (aₖ[3] - dₖ[3])*x[1] + (bₖ[3] - dₖ[3])*x[2] + (cₖ[3] - dₖ[3])*x[3] + dₖ[3]]

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
                @assert norm(Φₖ([0.0, 0.0, 0.3])) < atol
                @assert norm(Φₖ([0.0, 0.3, 0.0])) < atol
                @assert norm(Φₖ([0.3, 0.0, 0.0])) < atol
                @assert norm(Φₖ([0.3, 0.45, 0.25]) - (ψ(f̂ₖ_comp([0.3, 0.45, 0.25])) - 0.3*a₁ - 0.45*a₂ - 0.25*a₃)) < atol
                @assert norm(Φₖ([0.55, 0.45, 0.0]) - (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂)) < atol
            end
            @assert norm(Φₖ([0.0, 0.0000000000000001, 0.3])) < atol
            @assert norm(Φₖ([0.0, 0.3, 0.0])) < atol
            @assert norm(Φₖ([0.3, 0.0, 0.0])) < atol
            if j == 2
                @assert norm(Φₖ([0.6, 0.0, 0.4])) < atol
                @assert norm(Φₖ([0.0, 0.6, 0.4])) < atol
                @assert norm(Φₖ([0.55, 0.45, 0.0]) - (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂)) < atol
            end

            D = Inti.ReferenceTetrahedron
            T = SVector{3,Float64}
            el = Inti.ParametricElement{D,T}(x -> Fₖ(x))
            push!(els_curve, el)
            if j == 3
                ψₖ = (s) -> Fₖ([s[1], s[2], 1.0 - s[1] - s[2]])
                F = Inti.ReferenceTriangle
                bdry_el = Inti.ParametricElement{F,T}(s -> ψₖ(s))
                push!(els_curve_bdry, bdry_el)
                Ecurvebdry = typeof(first(els_curve_bdry))
            end

            Ecurve = typeof(first(els_curve))
            for k in Inti.entities(msh)
                # determine if the straight (LagrangeElement) mesh element
                # belongs to the entity and, if so, add the curved
                # (ParametricElement) element.
                if haskey(msh.ent2etags[k], E)
                    n_straight_vol_els = size(msh.etype2mat[E])[2]
                    if any((i) -> sort(node_indices) == sort(msh.etype2mat[E][:, i]), range(1,n_straight_vol_els))
                        haskey(ent2etags[k], Ecurve) || (ent2etags[k][Ecurve] = Vector{Int64}())
                        append!(ent2etags[k][Ecurve], length(els_curve))
                    end

                end
                # find entity that contains straight (LagrangeElement) face
                # element which is now being replaced by a curved
                # (ParametricElement) face element
                if (j == 3) && (haskey(msh.ent2etags[k], E_straight_bdry))
                    k.dim == 2 || continue
                    n_straight_bdry_els = size(msh.etype2mat[E_straight_bdry])[2]
                    if any((i) -> sort(node_indices_on_bdry) == sort(msh.etype2mat[E_straight_bdry][:, i]), range(1, n_straight_bdry_els))
                        haskey(ent2etags[k], Ecurvebdry) || (ent2etags[k][Ecurvebdry] = Vector{Int64}())
                        append!(ent2etags[k][Ecurvebdry], length(els_curve_bdry))
                    end
                end
            end
        else
            aₖ = straight_nodes[1]
            bₖ = straight_nodes[2]
            cₖ = straight_nodes[3]
            dₖ = straight_nodes[4]
            Fₖ = (x) -> [(aₖ[1] - bₖ[1])*x[1] + (cₖ[1] - bₖ[1])*x[2] + (dₖ[1] - bₖ[1])*x[3] + bₖ[1], (aₖ[2] - bₖ[2])*x[1] + (cₖ[2] - bₖ[2])*x[2] + (dₖ[2] - bₖ[2])*x[3] + bₖ[2], (aₖ[3] - bₖ[3])*x[1] + (cₖ[3] - bₖ[3])*x[2] + (dₖ[3] - bₖ[3])*x[3] + bₖ[3]]
            F̃ₖ = (x) -> Fₖ(x)
            
            append!(connect_straight, node_indices)
            el = Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}}(straight_nodes)
            push!(els_straight, el)

            for k in Inti.entities(msh)
                # determine if the straight mesh element belongs to the entity and, if so, add.
                if haskey(msh.ent2etags[k], E)
                    n_straight_vol_els = size(msh.etype2mat[E])[2]
                    if any((i) -> node_indices == msh.etype2mat[E][:, i], range(1,n_straight_vol_els))
                        haskey(ent2etags[k], E) || (ent2etags[k][E] = Vector{Int64}())
                        append!(ent2etags[k][E], length(els_straight))
                    end
                end
                # Note: This code does not consider the possibility of boundary
                # entities that are the boundary of straight simplices.  This is
                # because of the assumption above that if j > 1 the triangle is
                # curved.
            end
        end
    end
end

nv = 4 # Number of vertices for connectivity information in the volume
nv_bdry = 3 # Number of vertices for connectivity information on the boundary
Ecurve = typeof(first(els_curve))
Ecurvebdry = typeof(first(els_curve_bdry))
Estraight = Inti.LagrangeElement{Inti.ReferenceSimplex{3}, 4, SVector{3, Float64}} # TODO fix this to auto be a P1 element type

crvmsh.etype2mat[Ecurve] = reshape(connect_curve, nv, :)
crvmsh.etype2els[Ecurve] = convert(Vector{Ecurve}, els_curve)
crvmsh.etype2orientation[Ecurve] = ones(length(els_curve))

crvmsh.etype2mat[Estraight] = reshape(connect_straight, nv, :)
crvmsh.etype2els[Estraight] = convert(Vector{Estraight}, els_straight)
crvmsh.etype2orientation[Estraight] = ones(length(els_straight))

crvmsh.etype2mat[Ecurvebdry] = reshape(connect_curve_bdry, nv_bdry, :)
crvmsh.etype2els[Ecurvebdry] = convert(Vector{Ecurvebdry}, els_curve_bdry)
crvmsh.etype2orientation[Ecurvebdry] = ones(length(els_curve_bdry))

truevol = 2 * π^2 * r2^2 * r1
truesfcarea = 4 * π^2 * r1 * r2

Γₕ = crvmsh[Γ]
Ωₕ = crvmsh[Ω]

qorder = 2
Ωₕ_quad = Inti.Quadrature(Ωₕ, qorder = qorder)
Γₕ_quad = Inti.Quadrature(Γₕ, qorder = qorder)
@assert isapprox(Inti.integrate(x -> 1, Ωₕ_quad), truevol, rtol = 1e-6)
@assert isapprox(Inti.integrate(x -> 1, Γₕ_quad), truesfcarea, rtol = 1e-6)

qorder = 5
Ωₕ_quad = Inti.Quadrature(Ωₕ, qorder = qorder)
Γₕ_quad = Inti.Quadrature(Γₕ, qorder = qorder)
@assert isapprox(Inti.integrate(x -> 1, Ωₕ_quad), truevol, rtol = 1e-11)
@assert isapprox(Inti.integrate(x -> 1, Γₕ_quad), truesfcarea, rtol = 1e-11)

qorder = 8
Ωₕ_quad = Inti.Quadrature(Ωₕ, qorder = qorder)
Γₕ_quad = Inti.Quadrature(Γₕ, qorder = qorder)
@assert isapprox(Inti.integrate(x -> 1, Ωₕ_quad), truevol, rtol = 1e-12)
@assert isapprox(Inti.integrate(x -> 1, Γₕ_quad), truesfcarea, rtol = 1e-14)

divF = (x) -> x[3] + x[3]^2 + x[2]^3
F = (x) -> [x[1]*x[3], x[2]*x[3]^2, x[2]^3*x[3]]
#divF = (x) -> 1.0
#F = (x) -> 1/3*[x[1], x[2], x[3]]
Inti.integrate(q -> divF(q.coords), Ωₕ_quad)
Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad)