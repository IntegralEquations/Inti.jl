"""
    EntityKey

Used to represent the key of a [`GeometricEntity`](@ref), comprised of a `dim`
and a `tag` field, where `dim` is the geometrical dimension of the entity, and
`tag` is a unique integer identifying the entity.

The sign of the `tag` field is used to distinguish the orientation of the
entity, and is ignored when comparing two [`EntityKey`](@ref)s for equality.
"""
struct EntityKey
    dim::Int
    tag::Int
end

geometric_dimension(k::EntityKey) = k.dim
tag(k::EntityKey) = k.tag

Base.hash(ent::EntityKey, h::UInt) = hash((ent.dim, abs(ent.tag)), h)
Base.:(==)(e1::EntityKey, e2::EntityKey) = e1.dim == e2.dim && abs(e1.tag) == abs(e2.tag)

"""
    struct GeometricEntity

Used to represent geometrical objects such as lines, surfaces, and volumes.

Geometrical entities are stored in a global [`ENTITIES`](@ref) dictionary
mapping [`EntityKey`](@ref) to the corresponding `GeometricEntity`.

A `GeometricEntity` can also contain a `pushforward` field used to
parametrically represent the entry as the image of a reference domain
(`pushforward.domain`) under some function (`pushforward.parametrization`).
"""
struct GeometricEntity
    dim::Integer
    tag::Integer
    boundary::Vector{EntityKey}
    labels::Vector{String}
    pushforward
    function GeometricEntity(d::Integer, tag::Integer, boundary, labels, par)
        msg = "an elementary entities in the boundary has the wrong dimension"
        for b in boundary
            @assert geometric_dimension(b) == d - 1 msg
        end
        ent = new(d, tag, boundary, labels, par)
        # every entity gets added to a global variable ENTITIES so that we can
        # ensure the (d,t) pair is a UUID for an entity, and to easily retrieve
        # different entities.
        global_add_entity!(ent)
        return ent
    end
end

function GeometricEntity(;
    domain::HyperRectangle{N,T},
    parametrization,
    boundary = EntityKey[],
    labels = String[],
    tag = nothing,
) where {N,T}
    d = geometric_dimension(domain)
    t = isnothing(tag) ? new_tag(d) : tag
    V = return_type(parametrization, SVector{N,T})
    isbitstype(V) || (@warn "nonbits type $V returned by parametrization")
    return GeometricEntity(d, t, boundary, labels, (; domain, parametrization))
end

geometric_dimension(e::GeometricEntity) = e.dim
tag(e::GeometricEntity)                 = e.tag
boundary(e::GeometricEntity)            = e.boundary
labels(e::GeometricEntity)              = e.labels
push_forward(e::GeometricEntity)        = e.push_forward
domain(e::GeometricEntity)              = e.pushforward.domain
parametrization(e::GeometricEntity)     = e.pushforward.parametrization
hasparametrization(e::GeometricEntity)  = !isnothing(e.pushforward)
key(e::GeometricEntity)                 = EntityKey(geometric_dimension(e), tag(e))

function ambient_dimension(e::GeometricEntity)
    hasparametrization(e) || error("entity $(key(e)) has no parametrization")
    d = domain(e)
    x = center(d)
    f = parametrization(e)
    return length(f(x))
end

function Base.show(io::IO, ent::GeometricEntity)
    T = typeof(ent)
    d = geometric_dimension(ent)
    t = tag(ent)
    l = labels(ent)
    return print(io, "$T with (dim,tag)=($d,$t) and labels $l")
end

function Base.show(io::IO, k::EntityKey)
    return print(io, "EntityKey($(k.dim),$(k.tag))")
end

"""
    const ENTITIES

Dictionary mapping [`EntityKey`](@ref) to [`GeometricEntity`](@ref). Contains
all entities created in a given session.
"""
const ENTITIES = Dict{EntityKey,GeometricEntity}()

clear_entities!() = empty!(ENTITIES)

function global_add_entity!(ent::GeometricEntity)
    d, t = geometric_dimension(ent), tag(ent)
    k = EntityKey(d, t)
    msg = "overwriting an existing entity with the same (dim,tag)=($d,$t)"
    haskey(ENTITIES, k) && (@warn msg)
    return ENTITIES[k] = ent
end

function global_get_entity(k::EntityKey)
    return ENTITIES[k]
end

"""
    new_tag(dim)

Return a new tag for an entity of dimension `dim` so that `EntityKey(dim, tag)`
is not already in `ENTITIES`.
"""
function new_tag(dim::Int)
    tag = 1
    while haskey(ENTITIES, EntityKey(dim, tag))
        tag += 1
    end
    return tag
end
