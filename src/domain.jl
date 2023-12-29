"""
    struct Domain

Represents a physical domain as a union of [`AbstractEntity`](@ref) objects.
"""
struct Domain
    entities::Set{AbstractEntity}
end

Domain() = Domain(Set{AbstractEntity}())
Domain(ent::AbstractEntity) = Domain(Set(ent))

"""
    entities(Ω::Domain)

Return all entities making up a domain.
"""
entities(Ω::Domain) = Ω.entities

function Base.show(io::IO, d::Domain)
    ents = entities(d)
    n = length(entities(d))
    n == 1 ? print(io, "Domain with $n entity:") : print(io, "Domain with $n entities:")
    for ent in ents
        print(io, "\n\t $(ent)")
    end
    return io
end

function ambient_dimension(Ω::Domain)
    l, u = extrema(ambient_dimension(ent) for ent in entities(Ω))
    @assert l == u "ambient dimension of entities in a domain not equal"
    return u
end

"""
    skeleton(Ω::Domain)

Return all the boundaries of the domain, i.e. the domain's skeleton.
"""
function skeleton(Ω::Domain)
    ents = Set{AbstractEntity}()
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
    seen     = Set{AbstractEntity}()
    repeated = Set{AbstractEntity}()
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
    return Domain(setdiff(entities(Ω1), entities(Ω2)))
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

Base.in(ent::AbstractEntity, Ω::Domain) = in(ent, entities(Ω))
Base.in(Ω1::Domain, Ω2::Domain) = all(ent ∈ Ω2 for ent in entities(Ω1))
