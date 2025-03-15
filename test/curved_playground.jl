using Inti
using Meshes
using StaticArrays
using GLMakie
using Gmsh
using LinearAlgebra
using NearestNeighbors

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
    global circarea
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
            print(elind)
        end
        a₁ = ψ(α₁)
        a₂ = ψ(α₂)
        α₁hat = (α₁ - 0)/(1 - 0)
        α₂hat = (α₂ - 0)/(1 - 0)
        f̂ₖ = (t) -> 0 + t
        f̂ₖ_comp = (x) -> f̂ₖ( (x[1] * α₁hat + x[2] * α₂hat)/(x[1] + x[2]) )

        # l = 1 projection onto linear FE space
        πₖ¹ψ = (α) -> ψ(α₁) + (α - α₁)/(α₂ - α₁) * (ψ(α₂) - ψ(α₁))
        πₖ¹ψ_der = (α) -> (ψ(α₂) - ψ(α₁))/(α₂ - α₁)

        # l = 2 projection onto quadratic FE space
        β₀_l2 = α₁
        β₁_l2 = (α₁ + α₂)/2
        β₂_l2 = α₂
        psi_div0_0_l2 = ψ(β₀_l2)
        psi_div1_10_l2 = (ψ(β₁_l2) - ψ(β₀_l2))/(β₁_l2 - β₀_l2)
        psi_div1_21_l2 = (ψ(β₂_l2) - ψ(β₁_l2))/(β₂_l2 - β₁_l2)
        psi_div2_210_l2 = (psi_div1_21_l2 - psi_div1_10_l2)/(β₂_l2 - β₀_l2)
        πₖ²ψ = (α) -> psi_div0_0_l2 + (α - β₀_l2)*psi_div1_10_l2 + (α - β₀_l2)*(α - β₁_l2) * psi_div2_210_l2
        πₖ²ψ_der = (α) -> psi_div1_10_l2 + (α - β₁_l2) * psi_div2_210_l2 + (α - β₀_l2) * psi_div2_210_l2

        # l = 3 projection onto cubic FE space
        β₀_l3 = α₁
        β₁_l3 = α₁ + 1/3*(α₂ - α₁)
        β₂_l3 = α₁ + 2/3*(α₂ - α₁)
        β₃_l3 = α₂
        psi_div0_0_l3 = ψ(β₀_l3) # ψ[x₀]
        psi_div1_10_l3 = (ψ(β₁_l3) - ψ(β₀_l3))/(β₁_l3 - β₀_l3) # ψ[x₁,x₀]
        psi_div1_21_l3 = (ψ(β₂_l3) - ψ(β₁_l3))/(β₂_l3 - β₁_l3) # ψ[x₂,x₁]
        psi_div1_32_l3 = (ψ(β₃_l3) - ψ(β₂_l3))/(β₃_l3 - β₂_l3) # ψ[x₃,x₂]
        psi_div2_210_l3 = (psi_div1_21_l3 - psi_div1_10_l3)/(β₂_l3 - β₀_l3) # ψ[x₂,x₁,x₀]
        psi_div2_321_l3 = (psi_div1_32_l3 - psi_div1_21_l3)/(β₃_l3 - β₁_l3) # ψ[x₃,x₂,x₁]
        psi_div3_3210_l3 = (psi_div2_321_l3 - psi_div2_210_l3)/(β₃_l3 - β₀_l3) # ψ[x₃,x₂,x₁,x₀]
        πₖ³ψ = (α) -> psi_div0_0_l3 + (α - β₀_l3)*psi_div1_10_l3 + (α - β₀_l3)*(α - β₁_l3) * psi_div2_210_l3 + (α - β₀_l3)*(α - β₁_l3)*(α - β₂_l3)*psi_div3_3210_l3
        πₖ³ψ_der = (α) -> psi_div1_10_l3 + (α - β₁_l3) * psi_div2_210_l3 + (α - β₀_l3) * psi_div2_210_l3 + (α - β₁_l3)*(α - β₂_l3)*psi_div3_3210_l3 + (α - β₀_l3)*(α - β₂_l3)*psi_div3_3210_l3 + (α - β₀_l3)*(α - β₁_l3)*psi_div3_3210_l3

        # Nonlinear map
        Φₖ_l2 = (x) -> (x[1] + x[2])^4 * (ψ(f̂ₖ_comp(x)) - πₖ²ψ(f̂ₖ_comp(x))) + (x[1] + x[2])^2*(πₖ²ψ(f̂ₖ_comp(x)) - πₖ¹ψ(f̂ₖ_comp(x)))
        Φₖ_l2_der_x1 = (x) -> (4*(x[1] + x[2])^3 * (ψ(f̂ₖ_comp(x)) - πₖ²ψ(f̂ₖ_comp(x))) + 2*(x[1] + x[2]) * ( πₖ²ψ(f̂ₖ_comp(x)) - πₖ¹ψ(f̂ₖ_comp(x)))) + ((x[1] + x[2])^4 * (ψ_der(f̂ₖ_comp(x)) - πₖ²ψ_der(f̂ₖ_comp(x))) + (x[1] + x[2])^2*(πₖ²ψ_der(f̂ₖ_comp(x)) - πₖ¹ψ_der(f̂ₖ_comp(x)))) * ( α₁/(x[1] + x[2]) - (x[1]*α₁ + x[2]*α₂)/(x[1] + x[2])^2 )
        Φₖ_l2_der_x2 = (x) -> (4*(x[1] + x[2])^3 * (ψ(f̂ₖ_comp(x)) - πₖ²ψ(f̂ₖ_comp(x))) + 2*(x[1] + x[2]) * ( πₖ²ψ(f̂ₖ_comp(x)) - πₖ¹ψ(f̂ₖ_comp(x)))) + ((x[1] + x[2])^4 * (ψ_der(f̂ₖ_comp(x)) - πₖ²ψ_der(f̂ₖ_comp(x))) + (x[1] + x[2])^2*(πₖ²ψ_der(f̂ₖ_comp(x)) - πₖ¹ψ_der(f̂ₖ_comp(x)))) * ( α₂/(x[1] + x[2]) - (x[1]*α₁ + x[2]*α₂)/(x[1] + x[2])^2 )

        # l = 3
        Φₖ = (x) -> (x[1] + x[2])^5 * (ψ(f̂ₖ_comp(x)) - πₖ³ψ(f̂ₖ_comp(x))) + (x[1] + x[2])^2*(πₖ²ψ(f̂ₖ_comp(x)) - πₖ¹ψ(f̂ₖ_comp(x))) + (x[1] + x[2])^3*(πₖ³ψ(f̂ₖ_comp(x)) - πₖ²ψ(f̂ₖ_comp(x)))
        Φₖ_der_x1 = (x) -> (5*(x[1] + x[2])^4 * (ψ(f̂ₖ_comp(x)) - πₖ³ψ(f̂ₖ_comp(x))) + 2*(x[1] + x[2])*(πₖ²ψ(f̂ₖ_comp(x)) - πₖ¹ψ(f̂ₖ_comp(x))) + 3*(x[1] + x[2])^2*(πₖ³ψ(f̂ₖ_comp(x)) - πₖ²ψ(f̂ₖ_comp(x)))) + ((x[1] + x[2])^5 * (ψ_der(f̂ₖ_comp(x)) - πₖ³ψ_der(f̂ₖ_comp(x))) + (x[1] + x[2])^2*(πₖ²ψ_der(f̂ₖ_comp(x)) - πₖ¹ψ_der(f̂ₖ_comp(x))) + (x[1] + x[2])^3*(πₖ³ψ_der(f̂ₖ_comp(x)) - πₖ²ψ_der(f̂ₖ_comp(x)))) * ( α₁/(x[1] + x[2]) - (x[1]*α₁ + x[2]*α₂)/(x[1] + x[2])^2 )
        Φₖ_der_x2 = (x) -> (5*(x[1] + x[2])^4 * (ψ(f̂ₖ_comp(x)) - πₖ³ψ(f̂ₖ_comp(x))) + 2*(x[1] + x[2])*(πₖ²ψ(f̂ₖ_comp(x)) - πₖ¹ψ(f̂ₖ_comp(x))) + 3*(x[1] + x[2])^2*(πₖ³ψ(f̂ₖ_comp(x)) - πₖ²ψ(f̂ₖ_comp(x)))) + ((x[1] + x[2])^5 * (ψ_der(f̂ₖ_comp(x)) - πₖ³ψ_der(f̂ₖ_comp(x))) + (x[1] + x[2])^2*(πₖ²ψ_der(f̂ₖ_comp(x)) - πₖ¹ψ_der(f̂ₖ_comp(x))) + (x[1] + x[2])^3*(πₖ³ψ_der(f̂ₖ_comp(x)) - πₖ²ψ_der(f̂ₖ_comp(x)))) * ( α₂/(x[1] + x[2]) - (x[1]*α₁ + x[2]*α₂)/(x[1] + x[2])^2 )

        # Zlamal nonlinear map
        Φₖ_Z = (x) -> x[2]/(1 - x[1]) * (ψ(x[1] * α₁ + (1 - x[1]) * α₂) - x[1] * a₁ - (1 - x[1])*a₂)
        Φₖ_Z_der_x1 = (x) -> x[2]/(1 - x[1])^2 * (ψ(x[1] * α₁ + (1 - x[1]) * α₂) - x[1] * a₁ - (1 - x[1])*a₂) + x[2]/(1 - x[1]) * (ψ_der(x[1] * α₁ + (1 - x[1])*α₂)*(α₁ - α₂) - a₁ + a₂)
        Φₖ_Z_der_x2 = (x) -> 1/(1 - x[1]) * (ψ(x[1] * α₁ + (1 - x[1]) * α₂) - x[1] * a₁ - (1 - x[1])*a₂)

        # Affine map
        aₖ = msh.nodes[node_indices_on_bdry[1]]
        bₖ = msh.nodes[setdiff(node_indices, node_indices[verts_on_bdry])[1]]
        cₖ = msh.nodes[node_indices_on_bdry[2]]
        F̃ₖ = (x) -> [(cₖ[1] - bₖ[1])*x[1] + (aₖ[1] - bₖ[1])*x[2] + bₖ[1], (cₖ[2] - bₖ[2])*x[1] + (aₖ[2] - bₖ[2])*x[2] + bₖ[2]]

        # Full transformation
        Fₖ = (x) -> F̃ₖ(x) + Φₖ(x)
        D = Inti.ReferenceTriangle
        T = SVector{2,Float64}
        el = Inti.ParametricElement{D,T}(x -> Fₖ(x))
        push!(els, el)
        Jₖ = (x) -> [cₖ[1]-bₖ[1] + Φₖ_der_x1(x)[1]; aₖ[1]-bₖ[1] + Φₖ_der_x2(x)[1];; cₖ[2]-bₖ[2] + Φₖ_der_x1(x)[2]; aₖ[2]-bₖ[2] + Φₖ_der_x2(x)[2]]
        Jₖ_l2 = (x) -> [cₖ[1]-bₖ[1] + Φₖ_l2_der_x1(x)[1]; aₖ[1]-bₖ[1] + Φₖ_l2_der_x2(x)[1];; cₖ[2]-bₖ[2] + Φₖ_l2_der_x1(x)[2]; aₖ[2]-bₖ[2] + Φₖ_l2_der_x2(x)[2]]
        Fₖ_Z = (x) -> F̃ₖ(x) + Φₖ_Z(x)
        Jₖ_Z = (x) -> [cₖ[1]-bₖ[1] + Φₖ_Z_der_x1(x)[1]; aₖ[1]-bₖ[1] + Φₖ_Z_der_x2(x)[1];; cₖ[2]-bₖ[2] + Φₖ_Z_der_x1(x)[2]; aₖ[2]-bₖ[2] + Φₖ_Z_der_x2(x)[2]]
    else
        # Full transformation: Affine map
        aₖ = nodes[1]
        bₖ = nodes[2]
        cₖ = nodes[3]
        Fₖ = (x) -> [(cₖ[1] - bₖ[1])*x[1] + (aₖ[1] - bₖ[1])*x[2] + bₖ[1], (cₖ[2] - bₖ[2])*x[1] + (aₖ[2] - bₖ[2])*x[2] + bₖ[2]]
        Jₖ = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
        Jₖ_l2 = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
        Jₖ_Z = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
    end

    Q = Inti.VioreanuRokhlin(; domain=Inti.ReferenceTriangle(), order=qorder)()
    nq = length(Q[2])
    for q in 1:nq
        global circarea
        circarea += Q[2][q] * abs(det(Jₖ(Q[1][q])))
    end
end

#onedgrid = LinRange(0, 1, 200)
#x1grid = Array{Float64}(undef, 200, 200)
#y1grid = Array{Float64}(undef, 200, 200)
#for i=1:200
#    x1grid[i, :] = 1 .- onedgrid
#    y1grid[i, :] = LinRange(0, onedgrid[i], 200)
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