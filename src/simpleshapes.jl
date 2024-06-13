"""
    const SIMPLE_SHAPES

Available predefined shapes:

- [`ball`](@ref)
- [`torus`](@ref)
- [`ellipsoid`](@ref)
- [`bean`](@ref)
- [`acorn`](@ref)
- [`cushion`](@ref)
"""
const SIMPLE_SHAPES = ["ball", "torus", "ellipsoid", "bean", "acorn", "cushion"]

"""
    ball(; center = SVector(0, 0, 0), radius = 1, labels = String[])

Create an `GeometricEntity` representing the ball.
"""
function ball(; center = SVector(0, 0, 0), radius = 1, labels = String[])
    lc = SVector(-1.0, -1.0)
    hc = SVector(1.0, 1.0)
    bnd = EntityKey[]
    for i in 1:6
        # TODO: the curves shared by patches should be created only once.
        patch = parametric_surface(
            (u, v) -> _sphere_parametrization(u, v, i, radius, center),
            lc,
            hc,
        )
        push!(bnd, patch)
    end
    dim = 3
    tag = new_tag(dim)
    # TODO: add transfinite volume parametrization
    pushforward = nothing # no volume parametrization for now
    sph = GeometricEntity(dim, tag, bnd, labels, pushforward)
    return key(sph)
end

function _sphere_parametrization(u, v, id, rad = 1, center = SVector(0, 0, 0))
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
    return center .+ rad .* x ./ sqrt(u^2 + v^2 + 1)
end

"""
    torus(; center = SVector(0, 0, 0), r = 0.5, R = 1, labels = String[])

Create a torus geometric entity.

## Arguments
- `center`: The center of the torus. Default is `SVector(0, 0, 0)`.
- `r`: The radius of the tube of the torus. Default is `0.5`.
- `R`: The radius of the torus itself. Default is `1`.
- `labels`: An array of labels for the torus. Default is an empty array.

## Returns
The key of the created ellipsoid.
"""
function torus(; center = SVector(0, 0, 0), r = 0.5, R = 1, labels = String[])
    lc = π * SVector(-1.0, -1.0)
    hc = π * SVector(1.0, 1.0)
    bnd = EntityKey[]
    patch = parametric_surface((u, v) -> _torus_parametrization(u, v, r, R, center), lc, hc)
    push!(bnd, patch)
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    torus = GeometricEntity(dim, tag, bnd, labels, pushforward)
    return key(torus)
end

function _torus_parametrization(u, v, r, R, center)
    x = SVector((R + r * cos(v)) * cos(u), (R + r * cos(v)) * sin(u), r * sin(v))
    return x .+ center
end

"""
    ellipsoid(; center = SVector(0, 0, 0), paxis = SVector(1, 1, 1), labels = String[])

Create an ellipsoid geometric entity.

## Arguments
- `center`: The center of the ellipsoid. Default is `SVector(0, 0, 0)`.
- `paxis`: The principal axis lengths of the ellipsoid. Default is `SVector(1, 1, 1)`.
- `labels`: Optional labels for the ellipsoid.

## Returns
The key of the created ellipsoid.

"""
function ellipsoid(; center = SVector(0, 0, 0), paxis = SVector(1, 1, 1), labels = String[])
    lc = SVector(-1.0, -1.0)
    hc = SVector(1.0, 1.0)
    bnd = EntityKey[]
    for i in 1:6
        patch = parametric_surface(
            (u, v) -> _ellipsoid_parametrization(u, v, i, paxis, center),
            lc,
            hc,
        )
        push!(bnd, patch)
    end
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    ellipsoid = GeometricEntity(dim, tag, bnd, labels, pushforward)
    return key(ellipsoid)
end

function _ellipsoid_parametrization(u, v, id, paxis, center)
    x = _sphere_parametrization(u, v, id)
    return x .* paxis .+ center
end

"""
    bean(; center = SVector(0, 0, 0), paxis = SVector(1, 1, 1), labels = String[])

Constructs a bean-shaped geometric entity.

## Arguments
- `center`: The center of the bean shape. Default is `SVector(0, 0, 0)`.
- `paxis`: The principal axis of the bean shape. Default is `SVector(1, 1, 1)`.
- `labels`: An array of labels for the bean shape. Default is an empty array.

## Returns
The key of the constructed bean-shaped geometric entity.

"""
function bean(; center = SVector(0, 0, 0), paxis = SVector(1, 1, 1), labels = String[])
    lc = SVector(-1.0, -1.0)
    hc = SVector(1.0, 1.0)
    bnd = EntityKey[]
    for i in 1:6
        patch = parametric_surface(
            (u, v) -> _bean_parametrization(u, v, i, paxis, center),
            lc,
            hc,
        )
        push!(bnd, patch)
    end
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    bean = GeometricEntity(dim, tag, bnd, labels, pushforward)
    return key(bean)
end

function _bean_parametrization(u, v, id, paxis, center)
    x̂ = _sphere_parametrization(u, v, id)
    a = 0.8
    b = 0.8
    alpha1 = 0.3
    alpha2 = 0.4
    alpha3 = 0.1
    x = SVector(
        a * sqrt(1.0 - alpha3 * cospi(x̂[3])) .* x̂[1],
        -alpha1 * cospi(x̂[3]) + b * sqrt(1.0 - alpha2 * cospi(x̂[3])) .* x̂[2],
        x̂[3],
    )
    return x .* paxis .+ center
end

"""
    acorn(; center = SVector(0, 0, 0), radius = 1, labels = String[])

Create a geometric entity representing an acorn shape.

## Arguments
- `center`: The center of the acorn shape. Default is `SVector(0, 0, 0)`.
- `radius`: The radius of the acorn shape. Default is `1`.
- `labels`: An array of labels for the acorn shape. Default is an empty array.

## Returns
The key of the created geometric entity.

"""
function acorn(; center = SVector(0, 0, 0), radius = 1, labels = String[])
    lc = SVector(-1.0, -1.0)
    hc = SVector(1.0, 1.0)
    bnd = EntityKey[]
    for i in 1:6
        patch = parametric_surface(
            (u, v) -> _acorn_parametrization(u, v, i, radius, center),
            lc,
            hc,
        )
        push!(bnd, patch)
    end
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    acorn = GeometricEntity(dim, tag, bnd, labels, pushforward)
    return key(acorn)
end

function _acorn_parametrization(u, v, id, radius, center, rot)
    Rx = @SMatrix [1 0 0; 0 cos(rot[1]) sin(rot[1]); 0 -sin(rot[1]) cos(rot[1])]
    Ry = @SMatrix [cos(rot[2]) 0 -sin(rot[2]); 0 1 0; sin(rot[2]) 0 cos(rot[2])]
    Rz = @SMatrix [cos(rot[3]) sin(rot[3]) 0; -sin(rot[3]) cos(rot[3]) 0; 0 0 1]
    R = Rz * Ry * Rx
    x = _sphere_parametrization(u, v, id)
    th, phi, _ = cart2sph(x...)
    r = 0.6 + sqrt(4.25 + 2 * cos(3 * (phi + pi / 2)))
    x[1] = r .* cos(th) .* cos(phi)
    x[2] = r .* sin(th) .* cos(phi)
    x[3] = r .* sin(phi)
    x = R * x
    return radius .* x .+ center
end

"""
    cushion(; center = SVector(0, 0, 0), radius = 1, labels = String[])

Constructs a cushion shape centered at the specified `center` with the given `radius`.

## Arguments
- `center`: The center coordinates of the cushion. Default is `SVector(0, 0, 0)`.
- `radius`: The radius of the cushion. Default is `1`.
- `labels`: An array of labels for the cushion. Default is an empty array.

## Returns
The key of the constructed cushion.

"""
function cushion(; center = SVector(0, 0, 0), radius = 1, labels = String[])
    lc = SVector(-1.0, -1.0)
    hc = SVector(1.0, 1.0)
    bnd = EntityKey[]
    for i in 1:6
        patch = parametric_surface(
            (u, v) -> _cushion_parametrization(u, v, i, radius, center),
            lc,
            hc,
        )
        push!(bnd, patch)
    end
    dim = 3
    tag = new_tag(dim)
    pushforward = nothing
    cushion = GeometricEntity(dim, tag, bnd, labels, pushforward)
    return key(cushion)
end

function _cushion_parametrization(u, v, id, radius, center, rot)
    Rx = @SMatrix [1 0 0; 0 cos(rot[1]) sin(rot[1]); 0 -sin(rot[1]) cos(rot[1])]
    Ry = @SMatrix [cos(rot[2]) 0 -sin(rot[2]); 0 1 0; sin(rot[2]) 0 cos(rot[2])]
    Rz = @SMatrix [cos(rot[3]) sin(rot[3]) 0; -sin(rot[3]) cos(rot[3]) 0; 0 0 1]
    R = Rz * Ry * Rx
    x = _sphere_parametrization(u, v, id)
    th, phi, _ = cart2sph(x...)
    r = sqrt(0.8 + 0.5 * (cos(2 * th) - 1) .* (cos(4 * phi) - 1))
    x[1] = r .* cos(th) .* cos(phi)
    x[2] = r .* sin(th) .* cos(phi)
    x[3] = r .* sin(phi)
    x = R * x
    return radius .* x .+ center
end
