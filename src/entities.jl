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

labels(e::EntityKey) = labels(global_get_entity(e))
boundary(e::EntityKey) = boundary(global_get_entity(e))

function Base.show(io::IO, k::EntityKey)
    e = global_get_entity(k)
    print(io, "EntityKey: ($(k.dim), $(k.tag)) => $e")
    return io
end

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
    # the parametrization should maps SVector to SVector
    if !(V <: SVector)
        msg = """return_type of parametrization was $V (expected an SVector).
        This is usually due to a type instability in the parametrization
        function. Consider fixing this, as it may lead to serious performance issues."""
        @warn msg
        f = parametrization
        parametrization = x -> SVector(f(x))
    end
    return GeometricEntity(d, t, boundary, labels, (; domain, parametrization))
end

geometric_dimension(e::GeometricEntity) = e.dim
tag(e::GeometricEntity)                 = e.tag
boundary(e::GeometricEntity)            = e.boundary
labels(e::GeometricEntity)              = e.labels
push_forward(e::GeometricEntity)        = e.push_forward
domain(e::GeometricEntity)              = e.pushforward.domain
function parametrization(e::GeometricEntity)
    hasparametrization(e) || error("entity $(key(e)) has no parametrization")
    return e.pushforward.parametrization
end
hasparametrization(e::GeometricEntity) = !isnothing(e.pushforward)
key(e::GeometricEntity)                = EntityKey(geometric_dimension(e), tag(e))

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
    return print(io, "$T with labels $l")
end

"""
    measure(k::EntityKey, rtol)

Compute the length/area/volume of the entity `k` using an adaptive quadrature
with a relative tolerance `rtol`. Assumes that the entity has an explicit
parametrization.
"""
function measure(k::EntityKey, rtol = 1e-2)
    ent = global_get_entity(k)
    geometric_dimension(ent) == 1 || error("measure only supports 1d entities")
    hasparametrization(ent) || error("$k has no parametrization")
    d = domain(ent)
    τ̂ = domain(d)::ReferenceLine # FIXME: getters of reference domains should be renamed to avoid confusion
    a, b = low_corner(d), high_corner(d)
    l = norm(b - a)
    f = parametrization(ent) # map from [a,b] to the curve
    I, E = adaptive_integration(τ̂; rtol) do s
        x = d(s)
        μ = integration_measure(f, x)
        return l * μ
    end
    return I
end

"""
    line(a,b)

Create a [`GeometricEntity`] representing a straight line connecting points `a`
and `b`. The points `a` and `b` can be either `SVector`s or a `Tuple`.

The parametrization of the line is given by `f(u) = a + u(b - a)`, where `0 ≤ u
≤ 1`.
"""
function line(a, b)
    a, b = SVector(a), SVector(b)
    f = (u) -> a + u[1] * (b - a)
    d = HyperRectangle(SVector(0.0), SVector(1.0))
    ent = GeometricEntity(; domain = d, parametrization = f)
    return key(ent)
end

"""
    parametric_curve(f, a::Real, b::Real)

Create a [`GeometricEntity`] representing a parametric curve defined by the
`{f(t) | a ≤ t ≤ b}`.
"""
function parametric_curve(f, a::Real, b::Real)
    d = HyperRectangle(SVector(float(a)), SVector(float(b)))
    ent = GeometricEntity(; domain = d, parametrization = x -> f(x[1]))
    return key(ent)
end

# https://www.ljll.fr/perronnet/transfini/transfini.html
function transfinite_square(k1::T, k2::T, k3::T, k4::T) where {T<:EntityKey}
    c1, c2, c3, c4 = map((k1, k2, k3, k4)) do k
        l = global_get_entity(k)
        hasparametrization(l) || error("entity $(key(l)) has no parametrization")
        # chech that l is a curve in 2d
        @assert geometric_dimension(l) == 1
        @assert ambient_dimension(l) == 2
        # renormalize the parametrization to go from 0 to 1, flipping
        # orientation if needed
        f̂ = let f = parametrization(l), d = domain(l), flip = tag(k) < 0
            if flip
                x̂ -> f(d(1 - x̂))
            else
                x̂ -> f(d(x̂))
            end
        end
        return f̂
    end
    @assert c1(1) ≈ c2(0) && c2(1) ≈ c3(0) && c3(1) ≈ c4(0) && c4(1) ≈ c1(0)
    # create a closure and compute the parametrization
    f2d = _transfinite_square(c1, c2, c3, c4)
    d = HyperRectangle(SVector(0.0, 0.0), SVector(1.0, 1.0))
    ent = GeometricEntity(; domain = d, parametrization = f2d, boundary = [k1, k2, k3, k4])
    return key(ent)
end

function _transfinite_square(c1, c2, c3, c4)
    return x -> begin
        u, v = x[1], x[2]
        (1 - u) * c4(1 - v) + u * c2(v) + (1 - v) * c1(u) + v * c3(1 - u) - (
            (1 - u) * (1 - v) * c1(0) +
            u * (1 - v) * c2(0) +
            u * v * c3(0) +
            (1 - u) * v * c4(0)
        )
    end
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

"""
    global_get_entity(k::EntityKey)

Retrieve the [`GeometricEntity`](@ref) corresponding to the [`EntityKey`](@ref)
`k` from the global `ENTITIES` dictionary.
"""
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
