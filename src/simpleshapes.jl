const PREDEFINED_SHAPES = ["ellipsoid", "torus", "bean", "cushion", "acorn"]

"""
    GeometricEntity(shape::String [; translation, rotation, scaling, kwargs...])

Constructs a geometric entity with the specified shape and optional parameters,
and returns its `key`.

## Arguments
- `shape::String`: The shape of the geometric entity.
- `translation`: The translation vector of the geometric entity. Default is `SVector(0, 0, 0)`.
- `rotation`: The rotation vector of the geometric entity. Default is `SVector(0, 0, 0)`.
- `scaling`: The scaling vector of the geometric entity. Default is `SVector(1, 1, 1)`.
- `kwargs...`: Additional keyword arguments to be passed to the shape constructor.

## Supported shapes
- [`ellipsoid`](@ref)
- [`torus`](@ref)
- [`bean`](@ref)
- [`acorn`](@ref)
- [`cushion`](@ref)

"""
function GeometricEntity(
        shape::String;
        translation = SVector(0, 0, 0),
        rotation = SVector(0, 0, 0),
        scaling = SVector(1, 1, 1),
        kwargs...,
    )
    shape ∈ PREDEFINED_SHAPES ||
        throw(ArgumentError("shape must be one of $PREDEFINED_SHAPES"))
    f = getfield(Inti, Symbol(shape))
    return f(; translation, rotation, scaling, kwargs...)
end

"""
    ellipsoid(; translation, rotation, scaling, labels)

Create an ellipsoid entity in 3D, and apply optional transformations. Returns the key
of the created entity.
"""
function ellipsoid(;
        translation = SVector(0, 0, 0),
        rotation = SVector(0, 0, 0),
        scaling = SVector(1, 1, 1),
        labels = String[],
    )
    lc = SVector(-1.0, -1.0)
    hc = SVector(1.0, 1.0)
    bnd = EntityKey[]
    rot = rotation_matrix(rotation)
    for i in 1:6
        patch = parametric_surface(
            (u, v) -> _ellipsoid_parametrization(u, v, i, translation, rot, scaling),
            lc,
            hc,
        )
        push!(bnd, patch)
    end
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    return GeometricEntity(dim, tag, bnd, labels, pushforward)
end

function _ellipsoid_parametrization(u, v, id, trans, rot, scal)
    x = _unit_sphere_parametrization(u, v, id)
    return rot * (scal .* x) .+ trans
end

function _unit_sphere_parametrization(u, v, id)
    # parametrization of 6 patches. First gets a point on the cube [-1,1] ×
    # [-1,1] × [-1,1], the maps it onto the sphere
    if id == 1
        x = SVector(1.0, u, v)
    elseif id == 2
        x = SVector(-u, 1.0, v)
    elseif id == 3
        x = SVector(u, v, 1.0)
    elseif id == 4
        x = SVector(-1.0, -u, v)
    elseif id == 5
        x = SVector(u, -1.0, v)
    elseif id == 6
        x = SVector(-u, v, -1.0)
    end
    return x ./ sqrt(u^2 + v^2 + 1)
end

"""
    torus(; r, R, translation, rotation, scaling, labels)

Create a torus entity in 3D, and apply optional transformations. Returns the
key. The parameters `r` and `R` are the minor and major radii of the torus.
"""
function torus(;
        r = 0.5,
        R = 1,
        translation = SVector(0, 0, 0),
        rotation = SVector(0, 0, 0),
        scaling = SVector(1, 1, 1),
        labels = String[],
    )
    lc = π * SVector(-1.0, -1.0)
    hc = π * SVector(1.0, 1.0)
    bnd = EntityKey[]
    rot = rotation_matrix(rotation)
    patch = parametric_surface(lc, hc) do u, v
        # parametrization of a torus
        x = SVector((R + r * cos(v)) * cos(u), (R + r * cos(v)) * sin(u), r * sin(v))
        return rot * (scaling .* x) .+ translation
    end
    push!(bnd, patch)
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    return GeometricEntity(dim, tag, bnd, labels, pushforward)
end

"""
    bean(; translation, rotation, scaling, labels)

Create a bean entity in 3D, and apply optional transformations. Returns the key.
"""
function bean(;
        translation = SVector(0, 0, 0),
        rotation = SVector(0, 0, 0),
        scaling = SVector(1, 1, 1),
        labels = String[],
    )
    lc = SVector(-1.0, -1.0)
    hc = SVector(1.0, 1.0)
    bnd = EntityKey[]
    rot = rotation_matrix(rotation)
    for i in 1:6
        patch = parametric_surface(
            (u, v) -> _bean_parametrization(u, v, i, translation, rot, scaling),
            lc,
            hc,
        )
        push!(bnd, patch)
    end
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    return GeometricEntity(dim, tag, bnd, labels, pushforward)
end

function _bean_parametrization(u, v, id, trans, rot, scal)
    x̂ = _unit_sphere_parametrization(u, v, id)
    return rot * (scal .* _bean_deformation(x̂)) .+ trans
end

const _BEAN_A = 0.8
const _BEAN_B = 0.8
const _BEAN_ALPHA1 = 0.3
const _BEAN_ALPHA2 = 0.4
const _BEAN_ALPHA3 = 0.1

"""
    _bean_deformation(x̂)

Deformation carrying the unit sphere onto the bean surface. It leaves the third
coordinate alone and rescales the first two by functions of it, so it is a
diffeomorphism of the slab `|x̂[3]| < 1` and is inverted in closed form by
[`_bean_deformation_inverse`](@ref).
"""
function _bean_deformation(x̂)
    return SVector(
        _BEAN_A * sqrt(1.0 - _BEAN_ALPHA3 * cospi(x̂[3])) * x̂[1],
        -_BEAN_ALPHA1 * cospi(x̂[3]) +
            _BEAN_B * sqrt(1.0 - _BEAN_ALPHA2 * cospi(x̂[3])) * x̂[2],
        x̂[3],
    )
end

"""
    _bean_deformation_inverse(y)

Inverse of [`_bean_deformation`](@ref).
"""
function _bean_deformation_inverse(y)
    return SVector(
        y[1] / (_BEAN_A * sqrt(1.0 - _BEAN_ALPHA3 * cospi(y[3]))),
        (y[2] + _BEAN_ALPHA1 * cospi(y[3])) /
            (_BEAN_B * sqrt(1.0 - _BEAN_ALPHA2 * cospi(y[3]))),
        y[3],
    )
end

"""
    acorn(; translation, rotation, scaling, labels)

Create an acorn entity in 3D, and apply optional transformations. Returns the
key.
"""
function acorn(;
        translation = SVector(0, 0, 0),
        rotation = SVector(0, 0, 0),
        scaling = SVector(1, 1, 1),
        labels = String[],
    )
    lc = SVector(-1.0, -1.0)
    hc = SVector(1.0, 1.0)
    bnd = EntityKey[]
    rot = rotation_matrix(rotation)
    for i in 1:6
        patch = parametric_surface(
            (u, v) -> _acorn_parametrization(u, v, i, translation, rot, scaling),
            lc,
            hc,
        )
        push!(bnd, patch)
    end
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    return GeometricEntity(dim, tag, bnd, labels, pushforward)
end

function _acorn_parametrization(u, v, id, trans, rot, scal)
    x̂ = _unit_sphere_parametrization(u, v, id)
    th, phi, _ = cart2sph(x̂...)
    r = 0.6 + sqrt(4.25 + 2 * cos(3 * (phi + pi / 2)))
    x1 = r .* cos(th) .* cos(phi)
    x2 = r .* sin(th) .* cos(phi)
    x3 = r .* sin(phi)
    x = SVector(x1, x2, x3)
    return rot * (scal .* x) .+ trans
end

"""
    cushion(; translation, rotation, scaling, labels)

Create a cushion entity in 3D, and apply optional transformations. Returns the key.
"""
function cushion(; translation, rotation, scaling, labels = String[])
    lc = SVector(-1.0, -1.0)
    hc = SVector(1.0, 1.0)
    bnd = EntityKey[]
    rot = rotation_matrix(rotation)
    for i in 1:6
        patch = parametric_surface(
            (u, v) -> _cushion_parametrization(u, v, i, translation, rot, scaling),
            lc,
            hc,
        )
        push!(bnd, patch)
    end
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    return GeometricEntity(dim, tag, bnd, labels, pushforward)
end

function _cushion_parametrization(u, v, id, trans, rot, scal)
    x̂ = _unit_sphere_parametrization(u, v, id)
    th, phi, _ = cart2sph(x̂...)
    r = sqrt(0.8 + 0.5 * (cos(2 * th) - 1) .* (cos(4 * phi) - 1))
    x = r * SVector(cos(th) .* cos(phi), sin(th) .* cos(phi), sin(phi))
    return rot * (scal .* x) .+ trans
end
