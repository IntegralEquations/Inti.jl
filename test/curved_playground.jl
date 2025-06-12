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

tmesh = @elapsed begin
    Ω, msh = domain_and_mesh(; meshsize)
end
@info "Mesh generation time: $tmesh"

Γ = Inti.external_boundary(Ω)
Ωₕ = view(msh, Ω)
Γₕ = view(msh, Γ)

ψ = (t) -> [cos(2*π*t), sin(2*π*t)]

# Now sample from the patch
t = LinRange(0, 1, 500*Int(1/meshsize))

param_disc = ψ.(t)
kdt = KDTree(transpose(stack(param_disc, dims=1)))

nbdry_els = size(msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceHyperCube{1}, 2, SVector{2, Float64}}])[2]
bdry_node_idx = Vector{Int64}()
bdry_node_param_loc = Vector{Float64}()

uniqueidx(v) = unique(i -> v[i], eachindex(v))

crvmsh = Inti.Mesh{2,Float64}()
(; nodes, etype2mat, etype2els, ent2etags) = crvmsh
foreach(k -> ent2etags[k] = Dict{DataType,Vector{Int}}(), Inti.entities(msh))
append!(nodes, msh.nodes)

# Re-write nodes to lay on exact boundary
for elind = 1:nbdry_els
    local node_indices = msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceHyperCube{1}, 2, SVector{2, Float64}}][:, elind]
    local straight_nodes = crvmsh.nodes[node_indices]
    idxs, dists = nn(kdt, straight_nodes)
    crvmsh.nodes[node_indices[1]] = param_disc[idxs[1]]
    crvmsh.nodes[node_indices[2]] = param_disc[idxs[2]]
    push!(bdry_node_idx, node_indices[1])
    push!(bdry_node_idx, node_indices[2])
    push!(bdry_node_param_loc, t[idxs[1]])
    push!(bdry_node_param_loc, t[idxs[2]])
end
I = uniqueidx(bdry_node_idx)
bdry_node_idx = bdry_node_idx[I]
bdry_node_param_loc = bdry_node_param_loc[I]
node_to_param = Dict(zip(bdry_node_idx, bdry_node_param_loc))

# generate volume parametrizations
circarea = 0.0

connect_straight = Int[]
connect_curve = Int[]
connect_curve_bdry = Int[]
# TODO Could use an ElementIterator for straight elements
els_straight = []
els_curve = []
els_curve_bdry = []

for E in Inti.element_types(msh)
    # The purpose of this check is to see if other element types are present in
    # the mesh, such as e.g. quads; This code errors when encountering a quad,
    # but the method can be extended to transfer straight quads to the new mesh,
    # similar to how straight simplices are transferred below.
    E <: Union{Inti.LagrangeElement{Inti.ReferenceSimplex{2}},
    Inti.LagrangeElement{Inti.ReferenceHyperCube{1}}, SVector} || error()
    E <: SVector && continue
    E <: Inti.LagrangeElement{Inti.ReferenceHyperCube{1}} && continue
    E_straight_bdry = Inti.LagrangeElement{Inti.ReferenceHyperCube{1}, 2, SVector{2, Float64}}
    els = Inti.elements(msh, E)
    for elind in eachindex(els)
        node_indices = msh.etype2mat[E][:, elind]
        straight_nodes = crvmsh.nodes[node_indices]

        # First determine if straight or curved
        verts_on_bdry = findall(x -> x ∈ bdry_node_idx, node_indices)
        j = length(verts_on_bdry) # j in C. Bernardi SINUM Sec. 6
        if j > 1
            append!(connect_curve, node_indices)
            node_indices_on_bdry = node_indices[verts_on_bdry]
            append!(connect_curve_bdry, node_indices_on_bdry)

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
            aₖ = crvmsh.nodes[node_indices_on_bdry[1]]
            bₖ = crvmsh.nodes[setdiff(node_indices, node_indices[verts_on_bdry])[1]]
            cₖ = crvmsh.nodes[node_indices_on_bdry[2]]
            F̃ₖ = (x) -> [(cₖ[1] - bₖ[1])*x[1] + (aₖ[1] - bₖ[1])*x[2] + bₖ[1], (cₖ[2] - bₖ[2])*x[1] + (aₖ[2] - bₖ[2])*x[2] + bₖ[2]]

            # Full transformation
            Fₖ = (x) -> F̃ₖ(x) + Φₖ_l3(x)
            JF̃ₖ = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
            D = Inti.ReferenceTriangle
            T = SVector{2,Float64}
            el = Inti.ParametricElement{D,T}(x -> Fₖ(x))
            push!(els_curve, el)
            ψₖ = (s) -> Fₖ([1.0 - s[1], s[1]])
            L = Inti.ReferenceHyperCube{1}
            bdry_el = Inti.ParametricElement{L,T}(s -> ψₖ(s))
            push!(els_curve_bdry, bdry_el)

            Jₖ_l1 = (x) ->  JF̃ₖ(x) + transpose(ForwardDiff.jacobian(Φₖ_l1, x))
            Jₖ_l2 = (x) ->  JF̃ₖ(x) + transpose(ForwardDiff.jacobian(Φₖ_l2, x))
            Jₖ_l3 = (x) ->  JF̃ₖ(x) + transpose(ForwardDiff.jacobian(Φₖ_l3, x))

            Fₖ_Z = (x) -> F̃ₖ(x) + Φₖ_Z(x)
            Jₖ_Z = (x) -> [cₖ[1]-bₖ[1] + Φₖ_Z_der_x1(x)[1]; aₖ[1]-bₖ[1] + Φₖ_Z_der_x2(x)[1];; cₖ[2]-bₖ[2] + Φₖ_Z_der_x1(x)[2]; aₖ[2]-bₖ[2] + Φₖ_Z_der_x2(x)[2]]
            # loop over entities
            Ecurve = typeof(first(els_curve))
            Ecurvebdry = typeof(first(els_curve_bdry))
            for k in Inti.entities(msh)
                # determine if the straight (LagrangeElement) mesh element
                # belongs to the entity and, if so, add the curved
                # (ParametricElement) element.
                if haskey(msh.ent2etags[k], E)
                    n_straight_vol_els = size(msh.etype2mat[E])[2]
                    if any((i) -> node_indices == msh.etype2mat[E][:, i], range(1,n_straight_vol_els))
                        haskey(ent2etags[k], Ecurve) || (ent2etags[k][Ecurve] = Vector{Int64}())
                        append!(ent2etags[k][Ecurve], length(els_curve))
                    end

                end
                # find entity that contains straight (LagrangeElement) face
                # element which is now being replaced by a curved
                # (ParametricElement) face element
                if haskey(msh.ent2etags[k], E_straight_bdry)
                    k.dim == 1 || continue
                    n_straight_bdry_els = size(msh.etype2mat[E_straight_bdry])[2]
                    straight_entity_elementind = findall((i) -> sort(node_indices_on_bdry) == sort(msh.etype2mat[E_straight_bdry][:, i]), range(1, n_straight_bdry_els))
                    if !isempty(straight_entity_elementind)
                        haskey(ent2etags[k], Ecurvebdry) || (ent2etags[k][Ecurvebdry] = Vector{Int64}())
                        append!(ent2etags[k][Ecurvebdry], length(els_curve_bdry))
                    end
                end
            end

        else
            # Full transformation: Affine map
            aₖ = straight_nodes[1]
            bₖ = straight_nodes[2]
            cₖ = straight_nodes[3]
            Fₖ = (x) -> [(cₖ[1] - bₖ[1])*x[1] + (aₖ[1] - bₖ[1])*x[2] + bₖ[1], (cₖ[2] - bₖ[2])*x[1] + (aₖ[2] - bₖ[2])*x[2] + bₖ[2]]
            Jₖ = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
            Jₖ_l1 = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
            Jₖ_l2 = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
            Jₖ_l3 = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]
            Jₖ_Z  = (x) -> [cₖ[1]-bₖ[1]; aₖ[1]-bₖ[1];; cₖ[2]-bₖ[2]; aₖ[2]-bₖ[2]]

            append!(connect_straight, node_indices)
            el = Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{2, Float64}}(straight_nodes)
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

nv = 3 # Number of vertices for connectivity information in the volume
nv_bdry = 2 # Number of vertices for connectivity information on the boundary
Ecurve = typeof(first(els_curve))
Ecurvebdry = typeof(first(els_curve_bdry))
Estraight = Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{2, Float64}} # TODO fix this to auto be a P1 element type

crvmsh.etype2mat[Ecurve] = reshape(connect_curve, nv, :)
crvmsh.etype2els[Ecurve] = convert(Vector{Ecurve}, els_curve)
crvmsh.etype2orientation[Ecurve] = ones(length(els_curve))

crvmsh.etype2mat[Estraight] = reshape(connect_straight, nv, :)
crvmsh.etype2els[Estraight] = convert(Vector{Estraight}, els_straight)
crvmsh.etype2orientation[Estraight] = ones(length(els_straight))

crvmsh.etype2mat[Ecurvebdry] = reshape(connect_curve_bdry, nv_bdry, :)
crvmsh.etype2els[Ecurvebdry] = convert(Vector{Ecurvebdry}, els_curve_bdry)
crvmsh.etype2orientation[Ecurvebdry] = ones(length(els_curve_bdry))

Γₕ = crvmsh[Γ]
Ωₕ = crvmsh[Ω]

qorder = 2
Ωₕ_quad = Inti.Quadrature(Ωₕ, qorder = qorder)
Γₕ_quad = Inti.Quadrature(Γₕ, qorder = qorder)
@assert isapprox(Inti.integrate(x -> 1, Ωₕ_quad), π, rtol = 1e-7)
@assert isapprox(Inti.integrate(q -> q.coords[1]^4, Ωₕ_quad), π/8, rtol = 1e-5)

qorder = 5
Ωₕ_quad = Inti.Quadrature(Ωₕ, qorder = qorder)
Γₕ_quad = Inti.Quadrature(Γₕ, qorder = qorder)
@assert isapprox(Inti.integrate(x -> 1, Ωₕ_quad), π, rtol = 1e-11)
@assert isapprox(Inti.integrate(q -> q.coords[1]^4, Ωₕ_quad), π/8, rtol = 1e-10)

qorder = 8
Ωₕ_quad = Inti.Quadrature(Ωₕ, qorder = qorder)
Γₕ_quad = Inti.Quadrature(Γₕ, qorder = qorder)
@assert isapprox(Inti.integrate(x -> 1, Ωₕ_quad), π, rtol = 1e-14)
@assert isapprox(Inti.integrate(q -> q.coords[1]^4, Ωₕ_quad), π/8, rtol = 1e-14)

Fvol = (x) -> x[2]^2 - 2*x[2]*x[1]^3
F = (x) -> [x[1]*x[2]^2, x[1]^3*x[2]^2]
#Fvol = (x) -> 1.0
#F = (x) -> [1/2*x[1], 1/2*x[2]]
greenvol = Inti.integrate(q -> Fvol(q.coords), Ωₕ_quad)
greenline = Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad)
@assert isapprox(greenline, greenvol, rtol = 1e-13)