module IntiGmshExt

using Gmsh
import Inti
using Printf
using StaticArrays

function __init__()
    @info "Loading Inti.jl Gmsh extension"
end

"""
    struct GmshEntity <: AbstractEntity

Concrete type of [`AbstractEntity`](@ref Inti.AbstractEntity) generated using the `gmsh` API.
"""
struct GmshEntity <: Inti.AbstractEntity
    dim::Int
    gmshtag::Int
    tag::Int
    boundary::Vector{GmshEntity}
    model::String
    function GmshEntity(d::Integer, gmshtag::Integer, model, boundary = GmshEntity[])
        msg = "an elementary entities in the boundary has the wrong dimension"
        for b in boundary
            @assert geometric_dimension(b) == d - 1 msg
        end
        tag = Inti.new_tag(d)
        ent = new(d, gmshtag, tag, boundary, model)
        # every entity gets added to a global variable ENTITIES so that we can
        # ensure the (d,t) pair is a UUID for an entity, and to easily retrieve
        # different entities.
        Inti.global_add_entity!(ent)
        return ent
    end
end

gmshtag(e::GmshEntity) = e.gmshtag
gmshmodel(e::GmshEntity) = e.model
Inti.geometric_dimension(e::GmshEntity) = e.dim
Inti.tag(e::GmshEntity) = e.tag
Inti.boundary(e::GmshEntity) = e.boundary

function Inti.gmsh_import_domain(model = gmsh.model.getCurrent(); dim = 3, verbosity = 2)
    gmsh.isInitialized() == 1 ||
        error("gmsh is not initialized. Try `gmsh.initialize` first.")
    gmsh.option.setNumber("General.Verbosity", verbosity)
    Ω = Inti.Domain() # Create empty domain
    _import_domain!(Ω, model; dim)
    return Ω
end

"""
    _import_domain!(Ω::Domain,[model;dim=3])

Like [`Inti.gmsh_import_domain`](@ref), but appends entities to `Ω` instead of
creating a new domain.
"""
function _import_domain!(Ω::Inti.Domain, model = gmsh.model.getCurrent(); dim = 3)
    old_model = gmsh.model.getCurrent()
    gmsh.model.setCurrent(model)
    dim_tags = gmsh.model.getEntities(dim)
    for (_, tag) in dim_tags
        ent = GmshEntity(dim, tag, model)
        _fill_entity_boundary!(ent, model)
        push!(Ω.entities, ent)
    end
    gmsh.model.setCurrent(old_model)
    return Ω
end

"""
    _fill_entity_boundary!

Use the `gmsh` API to add the boundary of an `ElementaryEntity`.

This is a helper function, and should not be called by itself.
"""
function _fill_entity_boundary!(ent, model)
    combine  = true # FIXME: what should we use here?
    oriented = false
    dim_tags = gmsh.model.getBoundary((Inti.geometric_dimension(ent), gmshtag(ent)), combine, oriented)
    for (d, t) in dim_tags
        if haskey(Inti.ENTITIES,(d,t))
            bnd = Inti.ENTITIES[(d,t)]
        else
            bnd = GmshEntity(d, t, model)
            _fill_entity_boundary!(bnd, model)
        end
        push!(ent.boundary, bnd)
    end
    return ent
end

"""
    model_summary([model])

Print a summary of the `gmsh` `model` to the console; by defaul the current
model is used.
"""
function model_summary(model = gmsh.model.getCurrent())
    gmsh.model.setCurrent(model)
    @printf("List of entities in model %s: \n", model)
    @printf("|%10s|%10s|%10s|\n", "name", "dimension", "tag")
    ents = gmsh.model.getEntities()
    for ent in ents
        name = gmsh.model.getEntityName(ent...)
        dim, tag = ent
        @printf("|%10s|%10d|%10d|\n", name, dim, tag)
    end
    return println()
end

function Inti.gmsh_import_mesh(Ω::Inti.Domain; dim = 3)
    gmsh.isInitialized() == 1 ||
        error("gmsh is not initialized. Try `gmsh.initialize` first.")
    msh = Inti.LagrangeMesh{3,Float64}()
    _gmsh_import_mesh(msh, Ω)
    if dim == 3
        return msh
    elseif dim == 2
        return Inti._convert_to_2d(msh)
    else
        error("`dim` value must be `2` or `3`")
    end
end

"""
    _gmsh_import_mesh(msh,Ω)

Similar to [`Inti.gmsh_import_mesh`](@ref), but append information to `msh` instead of
creating a new mesh.

!!! danger
    This function assumes that `gmsh` has been initialized, and does not handle its
    finalization.
"""
function _gmsh_import_mesh(msh::Inti.LagrangeMesh, Ω::Inti.Domain)
    _, coord, _ = gmsh.model.mesh.getNodes()
    gmsh_nodes = collect(reinterpret(SVector{3,Float64}, coord))
    shift = length(msh.nodes) # gmsh node tags need to be shifted
    append!(msh.nodes, gmsh_nodes)
    # Recursively populate the dictionaries
    _domain_to_mesh!(msh, Ω, shift)
    return msh
end

"""
    _domain_to_mesh!(msh, Ω::Domain)

Recursively populate the dictionaries `etype2mat` and `ent2tag` in `msh` for the
entities in `Ω`. After all entities have been processed, the function recurses
on the [`skeleton`](@ref Inti.skeleton) of `Ω`.
"""
function _domain_to_mesh!(msh, Ω::Inti.Domain, shift)
    isempty(Ω) && (return msh)
    for ω in Ω.entities
        _ent_to_mesh!(msh.etype2mat, msh.ent2tags, ω, shift)
    end
    # recurse on the boundary of Ω
    Γ = Inti.skeleton(Ω)
    return _domain_to_mesh!(msh, Γ, shift)
end

"""
    _ent_to_mesh!(elements, ent2tag, ω::ElementaryEntity)

For each element type used to mesh `ω`:
- push into `elements::Dict` the pair `etype=>ntags`;
- push into `ent2tag::Dict` the pair `etype=>etags`;

where:
- `etype::DataType` determines the type of the element (see
    [`_type_tag_to_etype`](@ref));
- `ntags::Matrix{Int}` gives the indices of the nodes defining those
    elements;
- `etags::Vector{Int}` gives the indices of those elements in `elements`.
"""
function _ent_to_mesh!(elements, ent2tag, ω::GmshEntity, shift)
    ω in keys(ent2tag) && (return elements, ent2tag)
    etypes_to_etags = Dict{DataType,Vector{Int}}()
    # Loop on GMSH element types (integer)
    d = Inti.geometric_dimension(ω)
    type_tags, _, ntagss = gmsh.model.mesh.getElements(d, gmshtag(ω))
    for (type_tag, ntags) in zip(type_tags, ntagss)
        _, _, _, Np, _ = gmsh.model.mesh.getElementProperties(type_tag)
        ntags = reshape(ntags, Int(Np), :)
        etype = _type_tag_to_etype(type_tag)
        if etype in keys(elements)
            etag = size(elements[etype], 2) .+ collect(1:size(ntags, 2))
            ntags = hcat(elements[etype], ntags .+ shift)
        else
            etag = collect(1:size(ntags, 2))
        end
        push!(elements, etype => Int.(ntags))
        push!(etypes_to_etags, etype => etag)
    end
    push!(ent2tag, ω => etypes_to_etags)
    return elements, ent2tag
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

function Inti.gmsh_read_geo(fname; dim = 3)
    try
        gmsh.open(fname)
    catch
        @error "could not open $fname"
    end
    Ω = Inti.gmsh_import_domain(; dim)
    return Ω
end

"""
    gmsh_read_msh(fname::String; dim=3)

Read `fname` and create a `Domain` and a `GenericMesh` structure with all
entities in `Ω` of dimension `dim`.

!!! danger
    This function assumes that `gmsh` has been initialized, and does not handle its
    finalization.
"""
function Inti.gmsh_read_msh(fname; dim = 3)
    initialized = gmsh.isInitialized() == 1
    try
        initialized || gmsh.initialize()
        gmsh.open(fname)
    catch
        @error "could not open $fname"
    end
    Ω = Inti.gmsh_import_domain(; dim)
    msh = Inti.gmsh_import_mesh(Ω; dim)
    initialized || gmsh.finalize()
    return Ω, msh
end

end # module
