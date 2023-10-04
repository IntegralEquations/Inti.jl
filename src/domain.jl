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
    n == 1 ? print(io, "Domain with $n entity:\n") : print(io, "Domain with $n entities:")
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
Base.iterate(Ω::Domain, state=1) = iterate(entities(Ω), state)

Base.isempty(Ω::Domain) = isempty(entities(Ω))

############################################################################################
# Implementation of a concrete type of AbstractEntity that is generated using
# gmsh. The functionality using the gmsh API is implemented in the extension
# IntiGmshExt.jl
############################################################################################

"""
    gmsh_import_domain([model;dim=3])

Construct a [`Domain`](@ref) from the `gmsh` `model` with all entities of
dimension `dim`; by defaul the current `gmsh` model is used.

!!! note
    This function assumes that `gmsh` has been initialized, and
    does not handle its finalization.
"""
function gmsh_import_domain end

"""
    gmsh_model_summary([model])

Print a summary of the `gmsh` `model` to the console; by defaul the current
model is used.
"""
function gmsh_model_summary end
