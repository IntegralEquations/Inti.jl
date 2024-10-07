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

# defer some functions on EntityKey to the corresponding GeometricEntity
for f in (
    :labels,
    :boundary,
    :pushforward,
    :ambient_dimension,
    :hasparametrization,
    :parametrization,
)
    @eval $f(k::EntityKey) = $f(global_get_entity(k))
end

function (k::EntityKey)(x)
    hasparametrization(k) || error("$k has no parametrization")
    f = parametrization(k)
    s = tag(k) < 0 ? 1 - x : x
    return f(s)
end

function Base.show(io::IO, k::EntityKey)
    e = global_get_entity(k)
    print(io, "EntityKey: ($(k.dim), $(k.tag)) => $e")
    return io
end

"""
    struct GeometricEntity

Geometrical objects such as lines, surfaces, and volumes.

Geometrical entities are stored in a global [`ENTITIES`](@ref) dictionary
mapping [`EntityKey`](@ref) to the corresponding `GeometricEntity`, and usually
entities are manipulated through their keys.

A `GeometricEntity` can also contain a `pushforward` field used to
parametrically represent the entry as the image of a reference domain
(`pushforward.domain`) under some function (`pushforward.parametrization`).

Note that entities are manipulated through their keys, and the `GeometricEntity`
constructor returns the key of the created entity; to retrieve the entity, use
the [`global_get_entity`](@ref) function.
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
        return key(ent)
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
    # try to evaluate to see if that errors
    try
        parametrization(center(domain))
    catch
        error("evaluating parametrization at the center of the domain failed")
    end
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
pushforward(e::GeometricEntity)         = e.pushforward
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
    return GeometricEntity(; domain = d, parametrization = f)
end

"""
    parametric_curve(f, a::Real, b::Real)

Create a [`GeometricEntity`] representing a parametric curve defined by the
`{f(t) | a ≤ t ≤ b}`. The function `f` should map a scalar to an `SVector`.

Flipping the orientation is supported by passing `a > b`.
"""
function parametric_curve(f::F, a::Real, b::Real; kwargs...) where {F}
    if a > b # flip parametrization to restore order in the universe
        d = HyperRectangle(SVector(float(b)), SVector(float(a)))
        parametrization = x -> f(b + a - x[1])
    else
        d = HyperRectangle(SVector(float(a)), SVector(float(b)))
        parametrization = x -> f(x[1])
    end
    return GeometricEntity(; domain = d, parametrization, kwargs...)
end

# https://www.ljll.fr/perronnet/transfini/transfini.html
function transfinite_square(k1::T, k2::T, k3::T, k4::T; kwargs...) where {T<:EntityKey}
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
    same = (x, y) -> isapprox(x, y; atol = 1e-8)
    if !(
        same(c1(1), c2(0)) &&
        same(c2(1), c3(0)) &&
        same(c3(1), c4(0)) &&
        same(c4(1), c1(0))
    )
        error("the curves do not form a square")
    end
    # create a closure and compute the parametrization
    f2d = _transfinite_square(c1, c2, c3, c4)
    d = HyperRectangle(SVector(0.0, 0.0), SVector(1.0, 1.0))
    return GeometricEntity(;
        domain = d,
        parametrization = f2d,
        boundary = [k1, k2, k3, k4],
        kwargs...,
    )
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
        parametric_surface(f, lc, hc, boundary = nothing; kwargs...)

Create a parametric surface defined by the function `f` over the rectangular domain
defined by the lower corner `lc` and the upper corner `hc`. The optional `boundary`
argument can be used to specify the boundary curves of the surface.

## Arguments
- `f`: A function that takes two arguments `x` and `y` and returns a tuple `(u, v)`
    representing the parametric coordinates of the surface at `(x, y)`.
- `lc`: A 2-element array representing the lower corner of the rectangular domain.
- `hc`: A 2-element array representing the upper corner of the rectangular domain.
- `boundary`: An optional array of boundary curves that define the surface.

## Keyword Arguments
- `kwargs`: Additional keyword arguments that can be passed to the `GeometricEntity`
    constructor.

## Returns
- The key of the created `GeometricEntity`.

"""
function parametric_surface(f, lc, hc, boundary = nothing; kwargs...)
    @assert length(lc) == length(hc) == 2 "a and b should have length 2"
    l1 = parametric_curve(x -> f(x, lc[2]), lc[1], hc[1])
    l2 = parametric_curve(x -> f(hc[1], x), lc[2], hc[2])
    l3 = parametric_curve(x -> f(x, hc[2]), hc[1], lc[1])
    l4 = parametric_curve(x -> f(lc[1], x), hc[2], lc[2])
    d = HyperRectangle(SVector(lc), SVector(hc))
    return GeometricEntity(;
        domain = d,
        parametrization = x -> f(x[1], x[2]),
        boundary = isnothing(boundary) ? [l1, l2, l3, l4] : boundary,
        kwargs...,
    )
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
function new_tag(dim::Integer)
    tag = 1
    while haskey(ENTITIES, EntityKey(dim, tag))
        tag += 1
    end
    return tag
end
