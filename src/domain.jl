"""
    struct Domain

Representation of a geometrical domain formed by a set of entities with the same
geometric dimension. For basic set operations on domains are supported (union,
intersection, difference, etc), and they all return a new `Domain` object.

Calling `keys(Ω)` returns the set of [`EntityKey`](@ref)s that make up the
domain; given a key, the underlying entities can be accessed with
[`global_get_entity(key)`](@ref).
"""
struct Domain
    keys::Set{EntityKey}
    function Domain(ents::Set{EntityKey})
        @assert allequal(geometric_dimension(ent) for ent in ents) "entities in a domain have different dimensions"
        return new(ents)
    end
end

Domain() = Domain(Set{EntityKey}())
Domain(ents::Vararg{EntityKey}) = Domain(Set(ents))
Domain(ents) = Domain(Set(ents))

"""
    Domain([f::Function,] keys)

Create a domain from a set of [`EntityKey`](@ref)s. Optionally, a filter
function `f` can be passed to filter the entities.

Note that all entities in a domain must have the same geometric dimension.
"""
function Domain(f::Function, ents)
    return Domain(filter(f, ents))
end
Domain(f::Function, Ω::Domain) = Domain(f, keys(Ω))
Domain(ents::AbstractVector{EntityKey}) = Domain(Set(ents))

"""
    entities(Ω::Domain)

Return all entities making up a domain (as a set of [`EntityKey`](@ref)s).
"""
entities(Ω::Domain) = Ω.keys

Base.keys(Ω::Domain) = Ω.keys

# helper function to get all keys in a domain recursively
function all_keys(Ω::Domain)
    k = Set{EntityKey}()
    return _all_keys!(k, Ω)
end
function _all_keys!(k, Ω::Domain)
    union!(k, Ω.keys)
    sk = skeleton(Ω)
    isempty(sk) && return k
    _all_keys!(k, skeleton(Ω))
    return k
end

function Base.show(io::IO, d::Domain)
    kk = keys(d)
    n  = length(entities(d))
    n == print(io, "Domain with $n ", n == 1 ? "entity" : "entities")
    for k in kk
        ent = global_get_entity(k)
        print(io, "\n $(k)")
    end
    return io
end

"""
    skeleton(Ω::Domain)

Return all the boundaries of the domain, i.e. the domain's skeleton.
"""
function skeleton(Ω::Domain)
    ents = Set{EntityKey}()
    for ent in entities(Ω)
        union!(ents, boundary(ent))
    end
    return Domain(ents)
end

"""
    internal_boundary(Ω::Domain)

Return the internal boundaries of a `Domain`. These are entities in
`skeleton(Ω)` which appear at least twice as a boundary of entities in `Ω`.
"""
function internal_boundary(Ω::Domain)
    seen     = Set{EntityKey}()
    repeated = Set{EntityKey}()
    for ω in entities(Ω)
        for γ in boundary(ω)
            in(γ, seen) ? push!(repeated, γ) : push!(seen, γ)
        end
    end
    return Domain(repeated)
end

"""
    external_boundary(Ω::Domain)

Return the external boundaries inside a domain. These are entities in the
skeleton of Ω which are not in the internal boundaries of Ω.

See also: [`internal_boundary`](@ref), [`skeleton`](@ref).
"""
function external_boundary(Ω::Domain)
    return setdiff(skeleton(Ω), internal_boundary(Ω))
end

"""
    boundary(Ω::Domain)

Return the external boundaries of a domain.

See also: [`external_boundary`](@ref), [`internal_boundary`](@ref), [`skeleton`](@ref).
"""
boundary(Ω::Domain) = external_boundary(Ω)

function Base.setdiff(Ω1::Domain, Ω2::Domain)
    return Domain(setdiff(keys(Ω1), keys(Ω2)))
end

function Base.intersect(Ω1::Domain, Ω2::Domain)
    if isempty(Ω1) || isempty(Ω2)
        return Domain()
    end
    d1, d2 = map(geometric_dimension, (Ω1, Ω2))
    d1 == d2 || error("domains have different dimensions: $d1 and $d2")
    # try intersection at highest dimension, if empty, try on skeleton
    Ωinter = Domain(intersect(keys(Ω1), keys(Ω2)))
    return isempty(Ωinter) ? intersect(skeleton(Ω1), skeleton(Ω2)) : Ωinter
end

Base.:(==)(Ω1::Domain, Ω2::Domain) = (keys(Ω1) == keys(Ω2))

function geometric_dimension(Ω::Domain)
    l, u = extrema(geometric_dimension(ent) for ent in entities(Ω))
    @assert l == u "geometric dimension of entities in a domain not equal"
    return u
end

"""
    iterate(Ω::Domain)

Iterating over a domain means iterating over its entities.
"""
Base.iterate(Ω::Domain, state = 1) = iterate(entities(Ω), state)

Base.isempty(Ω::Domain) = isempty(entities(Ω))

Base.in(ent::EntityKey, Ω::Domain) = in(ent, entities(Ω))
Base.in(Ω1::Domain, Ω2::Domain) = all(ent in Ω2 for ent in entities(Ω1))

Base.union(Ω1::Domain, Ωs...) = Domain(union(Ω1.keys, map(ω -> keys(ω), Ωs)...))
Base.union(e1::EntityKey, e2::EntityKey) = Domain(e1, e2)
Base.union(e1::EntityKey, Ω::Domain) = Domain(e1, keys(Ω)...)
Base.union(Ω::Domain, e::EntityKey) = Domain(keys(Ω)..., e)
