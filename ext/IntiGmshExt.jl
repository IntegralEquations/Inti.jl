module IntiGmshExt

using Gmsh
import Inti
using StaticArrays

function __init__()
    @info "Loading Inti.jl Gmsh extension"
end

function Inti.import_mesh_from_gmsh_model(; dim = 3)
    @assert dim ∈ (2, 3) "only 2d and 3d meshes are supported"
    gmsh.isInitialized() == 1 ||
        error("gmsh is not initialized. Try gmsh.initialize() first.")
    msh = Inti.LagrangeMesh{3,Float64}()
    _import_mesh!(msh)
    dim == 2 && (msh = Inti._convert_to_2d(msh))
    # create a Domain with the entities of dimension `dim`
    Ω = Inti.Domain(Inti.entities(msh)) do ent
        return Inti.geometric_dimension(ent) == dim
    end
    return Ω, msh
end

function Inti.import_mesh_from_gmsh_file(fname; dim = 3)
    initialized = gmsh.isInitialized() == 1
    try
        initialized || gmsh.initialize()
        gmsh.open(fname)
    catch
        @error "could not open $fname"
    end
    Ω, msh = Inti.import_mesh_from_gmsh_model(; dim)
    initialized || gmsh.finalize()
    return Ω, msh
end

"""
    _import_mesh!(msh)

Import the mesh from gmsh into `msh` as a [`LagrangeMesh`](@ref
Inti.LagrangeMesh).
"""
function _import_mesh!(msh)
    # NOTE: when importing the nodes, we will renumber them so that the first node has
    # tag 1, the second node has tag 2, etc. This is not always the case in
    # gmsh, where the global node tags are not necessarily consecutive (AFAIU
    # they the tags need not even be a permutation of 1:N). Below we use a Dict
    # to map from the gmsh tags to the consecutive tags, but it would be
    # probably better to force gmsh to use consecutive tags in the first place.
    tags, coords, _ = gmsh.model.mesh.getNodes()
    tags_dict = Dict(zip(tags, collect(1:length(tags))))
    gmsh_nodes = reinterpret(SVector{3,Float64}, coords) |> collect
    shift = length(msh.nodes) # gmsh node tags need to be shifted in case msh was not empty
    append!(msh.nodes, gmsh_nodes)
    gmsh_dim_tags = gmsh.model.getEntities()
    for (dim, tag) in gmsh_dim_tags
        pgroups = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        labels = map(t -> gmsh.model.getPhysicalName(dim, t), pgroups)
        combined, oriented, recursive = true, true, false
        bnd_dim_tags = gmsh.model.getBoundary((dim, tag), combined, oriented, recursive)
        bnd = map(t -> Inti.EntityKey(t[1], t[2]), bnd_dim_tags)
        # add entity to global dictionary. The sign of tag is ignored,
        # orientation information is stored in the key. The underlying
        # parametrizatio of the entity is not (easily) available in gmsh, so we
        # set it to nothing.
        push_forward = nothing
        Inti.GeometricEntity(dim, abs(tag), bnd, labels, push_forward)
        key = Inti.EntityKey(dim, tag) # key for the entity
        _ent_to_mesh!(msh.etype2mat, msh.ent2etags, key, shift, tags_dict)
    end
    return msh
end

"""
    _ent_to_mesh!(etype2mat, ent2etags, key, shift)

For each element type used to mesh the entity with `key`:
- push into `etype2mat::Dict` the pair `etype=>connectivity`;
- push into `ent2etags::Dict` the pair `etype=>etags`;

where:
- `etype::DataType` determines the type of the element (see
    [`_type_tag_to_etype`](@ref));
- `connectivity::Matrix{Int}` gives the connectity matrix of the elements of
    type `etype`;
- `etags::Vector{Int}` gives the tags of the elements of type `etype` used to
  mesh the entity with the given `key`.
"""
function _ent_to_mesh!(etype2mat, ent2etags, key, shift, tags_dict)
    d, t = key.dim, key.tag
    haskey(ent2etags, key) && error("entity $key already in ent2etags")
    etype2etags = ent2etags[key] = Dict{DataType,Vector{Int}}()
    # Loop on GMSH element types (integer)
    type_tags, _, ntagss = gmsh.model.mesh.getElements(d, t)
    for (type_tag, ntags) in zip(type_tags, ntagss)
        _, _, _, Np, _ = gmsh.model.mesh.getElementProperties(type_tag)
        ntags = map(i -> tags_dict[i], reshape(ntags, Int(Np), :))
        etype = _type_tag_to_etype(type_tag)
        if etype in keys(etype2mat)
            etag = size(etype2mat[etype], 2) .+ collect(1:size(ntags, 2))
            ntags = hcat(etype2mat[etype], ntags .+ shift)
        else
            etag = collect(1:size(ntags, 2))
        end
        push!(etype2mat, etype => ntags)
        push!(etype2etags, etype => etag)
    end
    return nothing
end

"""
    _type_tag_to_etype(tag)

Mapping of `gmsh` element types, encoded as an integer, to the internal
equivalent of those.
"""
function _type_tag_to_etype(tag)
    T = SVector{3,Float64} # point type
    name, dim, order, num_nodes, ref_nodes, num_primary_nodes =
        gmsh.model.mesh.getElementProperties(tag)
    num_nodes = Int(num_nodes) #convert to Int64
    if occursin("Point", name)
        etype = SVector{3,Float64}
    elseif occursin("Line", name)
        etype = Inti.LagrangeLine{num_nodes,T}
    elseif occursin("Triangle", name)
        etype = Inti.LagrangeTriangle{num_nodes,T}
    elseif occursin("Quadrilateral", name)
        etype = Inti.LagrangeSquare{num_nodes,T}
    elseif occursin("Tetrahedron", name)
        etype = Inti.LagrangeTetrahedron{num_nodes,T}
    else
        error("unable to parse gmsh element of family $name")
    end
    return etype
end

"""
    _etype_to_type_tag(E::DataType)

The inverse of [`_type_tag_to_etype`](@ref).
"""
function _etype_to_type_tag(E::DataType)
    family_name = if E <: SVector{3,Float64}
        "Point"
    elseif E <: Inti.LagrangeLine
        "Line"
    elseif E <: Inti.LagrangeTriangle
        "Triangle"
    elseif E <: Inti.LagrangeSquare
        "Quadrilateral"
    elseif E <: Inti.LagrangeTetrahedron
        "Tetrahedron"
    else
        error("unable to parse element type $E")
    end
    order = Inti.order(E)
    return gmsh.model.mesh.getElementType(family_name, order)
end

function Inti.write_gmsh_model(msh::Inti.LagrangeMesh{N,Float64}; name = "") where {N}
    @assert N ∈ (2, 3)
    # lift the nodes to 3d if N == 2
    nodes = N == 3 ? Inti.nodes(msh) : [SVector(x[1], x[2], 0) for x in Inti.nodes(msh)]
    gmsh.isInitialized() == 1 || (@info "initializing gmsh"; gmsh.initialize())
    gmsh.model.add(name)
    # AbstractEntities in Inti ==> DiscreteEntities in gmsh. Bottom up approach.
    ents = collect(Inti.entities(msh))
    sort!(ents; lt = (x, y) -> (Inti.geometric_dimension(x) < Inti.geometric_dimension(y)))
    for ent in ents
        dim, tag = Inti.geometric_dimension(ent), Inti.tag(ent)
        bnd = [Inti.tag(b) for b in Inti.boundary(ent)]
        gmsh.model.addDiscreteEntity(dim, tag, bnd)
    end
    # add the nodes to each entity.
    # FIXME: this creates many duplicate nodes
    for ent in Inti.entities(msh)
        dim, tag  = Inti.geometric_dimension(ent), Inti.tag(ent)
        node_tags = Inti.ent2nodetags(msh, ent)
        coords    = reinterpret(Float64, [nodes[i] for i in node_tags]) |> collect
        gmsh.model.mesh.addNodes(dim, tag, node_tags, coords)
    end
    # add the elements for each entity
    for ent in Inti.entities(msh)
        dim, tag = Inti.geometric_dimension(ent), Inti.tag(ent)
        for (E, etags) in Inti.ent2etags(msh)[ent]
            elementType = E <: SVector ? 15 : _etype_to_type_tag(E)
            node_tags = vec(msh.etype2mat[E][:, etags])
            gmsh.model.mesh.addElementsByType(tag, elementType, [], node_tags)
        end
    end
    return nothing
end

function Inti.write_gmsh_view!(msh, data; name = "", model = gmsh.model.getCurrent())
    msg = "data must be of the same lenght as the number of nodes in the mesh"
    @assert length(data) == length(Inti.nodes(msh)) msg
    # tags, coords, _ = gmsh.model.mesh.getNodes()
    # dict = Dict(zip(collect(1:length(tags)), tags))
    # node_tags = map(i->dict[i],Inti.nodetags(msh))
    node_tags = Inti.nodetags(msh)
    view_tag = gmsh.view.add(name)
    # gmsh requires a vector of vectors format for data
    if data isa Vector{<:Number}
        data = [[x] for x in data]
    elseif data isa Vector{<:SVector}
        data = [collect(x) for x in data]
    end
    data_type = "NodeData"
    gmsh.view.addModelData(view_tag, 0, model, data_type, node_tags, data)
    return nothing
end

function Inti.write_gmsh_view!(msh, f::Function; kwargs...)
    data = f.(Inti.nodes(msh))
    return Inti.write_gmsh_view!(msh, data; kwargs...)
end

function Inti.write_gmsh_view(msh::Inti.SubMesh, data; viewname = "")
    @assert length(data) == length(Inti.nodes(msh))
    Inti.write_gmsh_model(msh.parent; name = modelname)
    Inti.write_gmsh_view!(msh, data; name = viewname)
    return nothing
end

function Inti.gmsh_curve(f, a, b; npts = 100, tag = 1)
    isclosed = all(f(a) .≈ f(b))
    is2d = length(f(a)) == 2
    pt_tags = Int32[]
    nmax = isclosed ? npts - 2 : npts - 1
    for i in 0:nmax
        s = i / (npts - 1)
        coords = f(a + s * (b - a))
        x, y = coords[1], coords[2]
        z = is2d ? 0 : coords[3]
        t = gmsh.model.occ.addPoint(x, y, z)
        push!(pt_tags, t)
    end
    # close the curve by adding the first point again
    isclosed && push!(pt_tags, pt_tags[1])
    t = gmsh.model.occ.addSpline(pt_tags, tag)
    gmsh.model.occ.synchronize()
    return t
end

end # module
