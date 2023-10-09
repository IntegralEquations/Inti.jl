"""
    abstract type AbstractEntity

Entity of geometrical nature. Identifiable throught its `(dim,tag)` key.
"""
abstract type AbstractEntity end

"""
    key(e::AbstractEntity)

The `(dim,tag)` pair used as a key to identify geometrical entities.
"""
function key(e::AbstractEntity)
    interface_method(e)
end

"""
    tag(::AbstractEntity)

Integer tag used to idetify geometrical entities.
"""
function tag(e::AbstractEntity)
    interface_method(e)
end

"""
    boundary(e::AbstractEntity)

A vector of entities of dimension `dim-1` that form the boundary of `e`.
"""
function boundary(e::AbstractEntity)
    interface_method(e)
end

"""
    ==(e1::AbstractEntity,e2::AbstractEntity)

Two entities are considered equal
`geometric_dimension(e1)==geometric_dimension(e2)` and `tag(e1)=tag(e2)`.

Notice that this implies `dim` and `tag` of an entity should uniquely define it,
and therefore global variables like [`TAGS`](@ref) are needed to make sure newly
created [`AbstractEntity`](@ref) have a new `(dim,tag)` identifier.
"""
function Base.:(==)(e1::AbstractEntity, e2::AbstractEntity)
    d1, t1 = geometric_dimension(e1), tag(e1)
    d2, t2 = geometric_dimension(e2), tag(e2)
    d1 == d2 || (return false)
    t1 == t2 || (return false)
    return true
end
Base.hash(ent::AbstractEntity, h::UInt) = hash((geometric_dimension(ent), tag(ent)), h)

function Base.show(io::IO, ent::AbstractEntity)
    T = typeof(ent)
    d = geometric_dimension(ent)
    t = tag(ent)
    return print(io, "$T with (dim,tag)=($d,$t)")
end

#####################################################################

# Variables and functions to globally keep track of entities

#####################################################################

"""
    const TAGS::Dict{Int,Vector{Int}}

Global dictionary storing the used entity tags (the value) for a given dimension
(the key).
"""
const TAGS = Dict{Int,Vector{Int}}()

"""
    const ENTITIES

Global dictionary storing the used entity tags (the value) for a given dimension
(the key).
"""
const ENTITIES = Dict{Tuple{Int,Int},AbstractEntity}()

"""
    global_add_entity!(ent::AbstractEntity)

Add `ent` to the global dictionary [`ENTITIES`](@ref) and update [`TAGS`](@ref)
with its `(dim,tag)` key. This function should be called by the inner
constructor of *every* [`AbstractEntity`](@ref).
"""
function global_add_entity!(ent::AbstractEntity)
    d, t = geometric_dimension(ent), tag(ent)
    _add_tag!(d, t) # add this tag to global list to make sure it is not used again
    msg = "overwriting ENTITIES: value in key ($d,$t) will be replaced"
    haskey(ENTITIES, (d, t)) && (@warn msg)
    ENTITIES[(d, t)] = ent
    return d, t
end

"""
    new_tag(dim)

Generate a unique tag for an `AbstractEntity` of dimension `dim`.

The implementation consists of adding one to the maximum value of `TAGS[dim]`

# See also: [`TAGS`](@ref).
"""
function new_tag(dim::Integer)
    if !haskey(TAGS, dim)
        return 1
    else
        tnew = maximum(TAGS[dim]) + 1
        return tnew
    end
end

function _add_tag!(dim, tag)
    if is_new_tag(dim, tag)
        # now add key
        if haskey(TAGS, dim)
            push!(TAGS[dim], tag)
        else
            TAGS[dim] = [tag]
        end
    else
        # print warning but don't add duplicate tag
        msg = "entity of dimension $dim and tag $tag already exists in TAGS.
       Creating a possibly duplicate entity."
        @warn msg
    end
    return TAGS
end

function is_new_tag(dim, tag)
    if haskey(TAGS, dim)
        existing_tags = TAGS[dim]
        if in(tag, existing_tags)
            return false
        end
    end
    return true
end

"""
    clear_entities!()

Empty the global variables used to keep track of the various entities
created.

# See also: [`ENTITIES`](@ref), [`TAGS`](@ref)
"""
function clear_entities!()
    empty!(TAGS)
    empty!(ENTITIES)
    return nothing
end
