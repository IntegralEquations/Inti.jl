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
meshsize = 0.1
qorder = 5
cutelevation = 0.2

Ω, msh = gmsh_cut_ball(;
    center = [0.0, 0.0, 0.0],
    radius = 1.0,
    meshsize = meshsize,
    cutelevation = cutelevation,
)
Γ = Inti.external_boundary(Ω)
Γ_msh = view(msh, Γ)

θ = LinRange(0, π, 50*round(Int, 1/meshsize))
ϕ = LinRange(0, 2*π, 50*round(Int, 1/meshsize))
# v = (θ, ϕ)
ψ₁ = (v) -> [sin(v[1]) * cos(v[2]), sin(v[1]) * sin(v[2]), cos(v[1])]

function ψ₁⁻¹(v0, p)
    F₁ = (v, p) -> ψ₁(v) - p
    prob₁ = NonlinearProblem(F₁, v0, p)
    return ψ₁⁻¹ = NonlinearSolve.solve(prob₁, SimpleNewtonRaphson())
end

face_element_on_sphere(nodelist) = all(norm.(nodelist) .≈ 1)
face_element_on_surface = (nodelist) -> face_element_on_sphere(nodelist)

chart_1 = Array{SVector{3,Float64}}(undef, length(θ)*length(ϕ))
chart_1_cart_idxs_θ = []
chart_1_cart_idxs_ϕ = []
for i in eachindex(θ)
    for j in eachindex(ϕ)
        chart_1[(i-1)*length(ϕ)+j] = [k for k in ψ₁((θ[i], ϕ[j]))]
        push!(chart_1_cart_idxs_θ, i)
        push!(chart_1_cart_idxs_ϕ, j)
    end
end
chart_1_kdt = KDTree(chart_1; reorder = false)

n2e = Inti.node2etags(msh)

nbdry_els = size(
    msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2},3,SVector{3,Float64}}],
)[2]
chart_1_bdry_node_idx = Vector{Int64}()
chart_1_bdry_node_param_loc = Vector{Vector{Float64}}()

uniqueidx(v) = unique(i -> v[i], eachindex(v))

# Set up chart <-> node Dict
for elind in 1:nbdry_els
    node_indices =
        msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2},3,SVector{3,Float64}}][
            :,
            elind,
        ]
    nodes = msh.nodes[node_indices]

    if face_element_on_surface(nodes)
        idxs, dists = nn(chart_1_kdt, nodes)
        ψ⁻¹ = ψ₁⁻¹
        if node_indices[1] ∉ chart_1_bdry_node_idx
            if abs(msh.nodes[node_indices[1]][3]) ≈ (1 - cutelevation)
                guess = SVector{2,Float64}((
                    θ[chart_1_cart_idxs_θ[idxs[1]]],
                    ϕ[chart_1_cart_idxs_ϕ[idxs[1]]],
                ))
                α = Vector{Float64}(ψ⁻¹(guess, copy(msh.nodes[node_indices[1]])))
                push!(chart_1_bdry_node_param_loc, α)
            else
                msh.nodes[node_indices[1]] = chart_1[idxs[1]]
                push!(
                    chart_1_bdry_node_param_loc,
                    Vector{Float64}([
                        θ[chart_1_cart_idxs_θ[idxs[1]]],
                        ϕ[chart_1_cart_idxs_ϕ[idxs[1]]],
                    ]),
                )
            end
            push!(chart_1_bdry_node_idx, node_indices[1])
        end
        if node_indices[2] ∉ chart_1_bdry_node_idx
            if abs(msh.nodes[node_indices[2]][3]) ≈ (1 - cutelevation)
                guess = SVector{2,Float64}((
                    θ[chart_1_cart_idxs_θ[idxs[1]]],
                    ϕ[chart_1_cart_idxs_ϕ[idxs[1]]],
                ))
                α = Vector{Float64}(ψ⁻¹(guess, copy(msh.nodes[node_indices[2]])))
                push!(chart_1_bdry_node_param_loc, α)
            else
                msh.nodes[node_indices[2]] = chart_1[idxs[2]]
                push!(
                    chart_1_bdry_node_param_loc,
                    Vector{Float64}([
                        θ[chart_1_cart_idxs_θ[idxs[2]]],
                        ϕ[chart_1_cart_idxs_ϕ[idxs[2]]],
                    ]),
                )
            end
            push!(chart_1_bdry_node_idx, node_indices[2])
        end
        if node_indices[3] ∉ chart_1_bdry_node_idx
            if abs(msh.nodes[node_indices[3]][3]) ≈ (1 - cutelevation)
                guess = SVector{2,Float64}((
                    θ[chart_1_cart_idxs_θ[idxs[1]]],
                    ϕ[chart_1_cart_idxs_ϕ[idxs[1]]],
                ))
                α = Vector{Float64}(ψ⁻¹(guess, copy(msh.nodes[node_indices[3]])))
                push!(chart_1_bdry_node_param_loc, α)
            else
                msh.nodes[node_indices[3]] = chart_1[idxs[3]]
                push!(
                    chart_1_bdry_node_param_loc,
                    Vector{Float64}([
                        θ[chart_1_cart_idxs_θ[idxs[3]]],
                        ϕ[chart_1_cart_idxs_ϕ[idxs[3]]],
                    ]),
                )
            end
            push!(chart_1_bdry_node_idx, node_indices[3])
        end
    end
end
chart_1_node_to_param = Dict(zip(chart_1_bdry_node_idx, chart_1_bdry_node_param_loc))

nvol_els = size(
    msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3},4,SVector{3,Float64}}],
)[2]
spharea = 0.0
nnewton = 0
elcount = 0
els = []
for elind in 1:nvol_els
    node_indices =
        msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{3},4,SVector{3,Float64}}][
            :,
            elind,
        ]
    nodes = msh.nodes[node_indices]

    verts_on_bdry = findall(x -> x ∈ chart_1_bdry_node_idx, node_indices)
    j = 0
    if !isempty(verts_on_bdry)
        nverts_in_chart = length(verts_on_bdry)
        j = nverts_in_chart
    end
    if j > 1
        node_indices_on_bdry = copy(node_indices[verts_on_bdry])
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
            candidate_els = candidate_els[length.(candidate_els) .== 3]
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
            res = ψ⁻¹(α₂, p)
            if res.retcode == ReturnCode.MaxIters
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
        if j == 3
            @assert (
                (abs(α₁[1] - α₂[1]) < π/8) &&
                (abs(α₂[1] - α₃[1]) < π/8) &&
                (abs(α₁[1] - α₃[1]) < π/8)
            )
        end
        if !(
            (abs(α₁[2] - α₂[2]) < π/8) &&
            (abs(α₂[2] - α₃[2]) < π/8) &&
            (abs(α₁[2] - α₃[2]) < π/8)
        )
            @warn "Chart parametrization warning at element #",
            elind,
            " with ",
            j,
            "verts on bdry, at θ ≈ ",
            max(α₁[1], α₂[1], α₃[1])
            @warn "Chart parametrization warning at element #",
            elind,
            " with ",
            j,
            "verts on bdry, at θ ≈ ",
            min(α₁[1], α₂[1], α₃[1])
        end
        a₁ = SVector{3,Float64}(ψ(α₁))
        a₂ = SVector{3,Float64}(ψ(α₂))
        a₃ = SVector{3,Float64}(ψ(α₃))
        α₁hat = SVector{2,Float64}(1.0, 0.0)
        α₂hat = SVector{2,Float64}(0.0, 1.0)
        α₃hat = SVector{2,Float64}(0.0, 0.0)

        πₖ¹_nodes = Inti.reference_nodes(
            Inti.LagrangeElement{Inti.ReferenceTriangle,3,SVector{2,Float64}},
        )
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
        F̃ₖ =
            (x) -> [
                (aₖ[1] - dₖ[1])*x[1] + (bₖ[1] - dₖ[1])*x[2] + (cₖ[1] - dₖ[1])*x[3] + dₖ[1],
                (aₖ[2] - dₖ[2])*x[1] + (bₖ[2] - dₖ[2])*x[2] + (cₖ[2] - dₖ[2])*x[3] + dₖ[2],
                (aₖ[3] - dₖ[3])*x[1] + (bₖ[3] - dₖ[3])*x[2] + (cₖ[3] - dₖ[3])*x[3] + dₖ[3],
            ]
        @assert aₖ ≈ a₁
        @assert bₖ ≈ a₂

        # l = 1
        πₖ¹_nodes = Inti.reference_nodes(
            Inti.LagrangeElement{
                Inti.ReferenceTriangle,
                binomial(2+1, 2),
                SVector{2,Float64},
            },
        )
        πₖ¹ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ¹_nodes))
        for i in eachindex(πₖ¹_nodes)
            πₖ¹ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ¹_nodes[i]))
        end
        πₖ¹ψ_reference_nodes = SVector{binomial(2+1, 2)}(πₖ¹ψ_reference_nodes)
        πₖ¹ψ =
            (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ¹ψ_reference_nodes)(x)
        #l = 2
        πₖ²_nodes = Inti.reference_nodes(
            Inti.LagrangeElement{
                Inti.ReferenceTriangle,
                binomial(2+2, 2),
                SVector{2,Float64},
            },
        )
        πₖ²ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ²_nodes))
        for i in eachindex(πₖ²_nodes)
            πₖ²ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ²_nodes[i]))
        end
        πₖ²ψ_reference_nodes = SVector{binomial(2+2, 2)}(πₖ²ψ_reference_nodes)
        πₖ²ψ =
            (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ²ψ_reference_nodes)(x)
        #l = 3
        πₖ³_nodes = Inti.reference_nodes(
            Inti.LagrangeElement{
                Inti.ReferenceTriangle,
                binomial(2+3, 2),
                SVector{2,Float64},
            },
        )
        πₖ³ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ³_nodes))
        for i in eachindex(πₖ³_nodes)
            πₖ³ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ³_nodes[i]))
        end
        πₖ³ψ_reference_nodes = SVector{binomial(2+3, 2)}(πₖ³ψ_reference_nodes)
        πₖ³ψ =
            (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ³ψ_reference_nodes)(x)
        #l = 4
        πₖ⁴_nodes = Inti.reference_nodes(
            Inti.LagrangeElement{
                Inti.ReferenceTriangle,
                binomial(2+4, 2),
                SVector{2,Float64},
            },
        )
        πₖ⁴ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ⁴_nodes))
        for i in eachindex(πₖ⁴_nodes)
            πₖ⁴ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁴_nodes[i]))
        end
        πₖ⁴ψ_reference_nodes = SVector{binomial(2+4, 2)}(πₖ⁴ψ_reference_nodes)
        πₖ⁴ψ =
            (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ⁴ψ_reference_nodes)(x)
        #l = 5
        πₖ⁵_nodes = Inti.reference_nodes(
            Inti.LagrangeElement{
                Inti.ReferenceTriangle,
                binomial(2+5, 2),
                SVector{2,Float64},
            },
        )
        πₖ⁵ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ⁵_nodes))
        for i in eachindex(πₖ⁵_nodes)
            πₖ⁵ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁵_nodes[i]))
        end
        πₖ⁵ψ_reference_nodes = SVector{binomial(2+5, 2)}(πₖ⁵ψ_reference_nodes)
        πₖ⁵ψ =
            (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ⁵ψ_reference_nodes)(x)

        # Nonlinear map
        if j == 3
            f̂ₖ_comp =
                (x) ->
                    f̂ₖ((x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat)/(x[1] + x[2] + x[3]))
            Φₖ_θ1 =
                (x) -> (
                    (x[1] + x[2] + x[3])^3 * (
                        ψ(f̂ₖ_comp(x)) - πₖ¹ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    )
                )
            Φₖ_θ2 =
                (x) -> (
                    (x[1] + x[2] + x[3])^4 * (
                        ψ(f̂ₖ_comp(x)) - πₖ²ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^2 * (
                        πₖ²ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ¹ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    )
                )
            Φₖ_θ3 =
                (x) -> (
                    (x[1] + x[2] + x[3])^5 * (
                        ψ(f̂ₖ_comp(x)) - πₖ³ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^2 * (
                        πₖ²ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ¹ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^3 * (
                        πₖ³ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ²ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    )
                )
            Φₖ_θ4 =
                (x) -> (
                    (x[1] + x[2] + x[3])^6 * (
                        ψ(f̂ₖ_comp(x)) - πₖ⁴ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^2 * (
                        πₖ²ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ¹ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^3 * (
                        πₖ³ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ²ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^4 * (
                        πₖ⁴ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ³ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    )
                )
            Φₖ_θ5 =
                (x) -> (
                    (x[1] + x[2] + x[3])^7 * (
                        ψ(f̂ₖ_comp(x)) - πₖ⁵ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^2 * (
                        πₖ²ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ¹ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^3 * (
                        πₖ³ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ²ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^4 * (
                        πₖ⁴ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ³ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    ) +
                    (x[1] + x[2] + x[3])^5 * (
                        πₖ⁵ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        ) - πₖ⁴ψ(
                            (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) / (x[1] + x[2] + x[3]),
                        )
                    )
                )
        else
            f̂ₖ_comp = (x) -> f̂ₖ((x[1] * α₁hat + x[2] * α₂hat)/(x[1] + x[2]))
            Φₖ_θ1 =
                (x) -> (
                    (x[1] + x[2])^3 *
                    (ψ(f̂ₖ_comp(x)) - πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])))
                )
            Φₖ_θ2 =
                (x) -> (
                    (x[1] + x[2])^4 *
                    (ψ(f̂ₖ_comp(x)) - πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) +
                    (x[1] + x[2])^2 * (
                        πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    )
                )
            Φₖ_θ3 =
                (x) -> (
                    (x[1] + x[2])^5 *
                    (ψ(f̂ₖ_comp(x)) - πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) +
                    (x[1] + x[2])^2 * (
                        πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    ) +
                    (x[1] + x[2])^3 * (
                        πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    )
                )
            Φₖ_θ4 =
                (x) -> (
                    (x[1] + x[2])^6 *
                    (ψ(f̂ₖ_comp(x)) - πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) +
                    (x[1] + x[2])^2 * (
                        πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    ) +
                    (x[1] + x[2])^3 * (
                        πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    ) +
                    (x[1] + x[2])^4 * (
                        πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    )
                )
            Φₖ_θ5 =
                (x) -> (
                    (x[1] + x[2])^7 *
                    (ψ(f̂ₖ_comp(x)) - πₖ⁵ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))) +
                    (x[1] + x[2])^2 * (
                        πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    ) +
                    (x[1] + x[2])^3 * (
                        πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    ) +
                    (x[1] + x[2])^4 * (
                        πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    ) +
                    (x[1] + x[2])^5 * (
                        πₖ⁵ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                        πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                    )
                )
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
            @assert norm(
                Φₖ([0.3, 0.45, 0.25]) -
                (ψ(f̂ₖ_comp([0.3, 0.45, 0.25])) - 0.3*a₁ - 0.45*a₂ - 0.25*a₃),
            ) < atol
            @assert norm(
                Φₖ([0.55, 0.45, 0.0]) -
                (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂),
            ) < atol
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
            @assert norm(
                Φₖ([0.55, 0.45, 0.0]) -
                (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂),
            ) < atol
        end
    else
        aₖ = nodes[1]
        bₖ = nodes[2]
        cₖ = nodes[3]
        dₖ = nodes[4]
        Fₖ =
            (x) -> [
                (aₖ[1] - bₖ[1])*x[1] + (cₖ[1] - bₖ[1])*x[2] + (dₖ[1] - bₖ[1])*x[3] + bₖ[1],
                (aₖ[2] - bₖ[2])*x[1] + (cₖ[2] - bₖ[2])*x[2] + (dₖ[2] - bₖ[2])*x[3] + bₖ[2],
                (aₖ[3] - bₖ[3])*x[1] + (cₖ[3] - bₖ[3])*x[2] + (dₖ[3] - bₖ[3])*x[3] + bₖ[3],
            ]
    end

    Jₖ = (x) -> transpose(ForwardDiff.jacobian(Fₖ, x))

    Q = Inti.Gauss(; domain = Inti.ReferenceTetrahedron(), order = qorder)()
    #Q = Inti.VioreanuRokhlin(; domain=Inti.ReferenceTetrahedron(), order=qorder)()
    nq = length(Q[2])
    elarea = 0.0
    for q in 1:nq
        tmp = Q[2][q] * abs(det(Jₖ(Q[1][q])))
        elarea += tmp
    end
    global spharea += elarea
end

truearea = 4 * π / 3 - 2 * 1/3 * π * cutelevation^2 * (3 - cutelevation)
