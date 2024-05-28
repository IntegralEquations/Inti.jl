"""
    struct Domain

Representation of a geometrical domain formed by a set of entities with the same
geometric dimension. For basic set operations on domains are supported (union,
intersection, difference, etc), and they all return a new `Domain` object.

The unlerlying entities are stored in a [`Set`](@ref) of [`EntityKey`](@ref)s;
the underlying entities can be accessed with [`global_get_entity(key)`](@ref).
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

"""
    Domain([f::Function,] keys)

Create a domain from a set of [`EntityKey`](@ref)s. Optionally, a filter
function `f` can be passed to filter the entities.

Note that all entities in a domain must have the same geometric dimension.
"""
function Domain(f::Function, ents)
    return Domain(filter(f, ents))
end
Domain(ents::AbstractVector{EntityKey}) = Domain(Set(ents))

Base.keys(Ω::Domain) = Ω.keys

"""
    entities(Ω::Domain)

Return all entities making up a domain.
"""
entities(Ω::Domain) = (global_get_entity(k) for k in Ω.keys)

function Base.show(io::IO, d::Domain)
    kk = keys(d)
    n  = length(entities(d))
    n == print(io, "Domain with $n ", n == 1 ? "entity" : "entities")
    for k in kk
        ent = global_get_entity(k)
        print(io, "\n $(k) --> $ent")
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
Base.in(Ω1::Domain, Ω2::Domain) = all(ent ∈ Ω2 for ent in entities(Ω1))
