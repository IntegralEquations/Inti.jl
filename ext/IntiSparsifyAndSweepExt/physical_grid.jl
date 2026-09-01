# Affine bridge between the unit-cube grid the core solver works on and the
# physical box the scatterer lives in.

"""
    PhysicalGrid{D}(g, origin, L)

The Cartesian unknowns `I = {1,…,n}^D` of a [`Grid`](@ref), placed in physical
space by the uniform affine map

    x = origin + L * ξ ,    ξ = i*h ∈ (0,1)^D .

Under `x = L x′` the Lippmann-Schwinger equation is invariant with `ω′ = ω L`:

* 2D — `G(L r) = (i/4)H₀(ω L r)`, and `dy = L² dy′`, so `ω² L² ∫G′ = ω′² ∫G′`;
* 3D — `G(L r) = e^{iωLr}/(4πLr) = L⁻¹G′(r′)` and `dy = L³ dy′`, so
  `ω² L³ L⁻¹ = ω′²`.

The core solver has `Ω = (0,1)^D` hard-coded: a physical problem is
solved by scaling the frequency with [`unit_frequency`](@ref) and sampling the
medium with [`sample_field`](@ref).
"""
struct PhysicalGrid{D}
    g::Grid{D}
    origin::NTuple{D,Float64}
    L::Float64
end

"""
    PhysicalGrid(Val(D), lo, hi, n; b = 8, pad = 0.15)

Cube of side `L` containing the box `[lo, hi]`, centered on it and enlarged by
`pad` (as a fraction of the longest side) on every face.

The reason for this padding is that the core solver assumes `m` is supported strictly
inside `Ω`, and since its PML occupies the outermost `b` layers, the scatterer must
not reach the edge of the cube.  [`check_support`](@ref) verifies this after the
medium has been sampled.
"""
function PhysicalGrid(::Val{D}, lo::NTuple{D,<:Real}, hi::NTuple{D,<:Real},
                      n::Integer; b::Integer = 8, pad::Real = 0.15) where {D}
    all(d -> hi[d] > lo[d], 1:D) || throw(ArgumentError("need hi > lo componentwise"))
    side = maximum(ntuple(d -> hi[d] - lo[d], D))
    L = side * (1 + 2 * pad)
    ctr = ntuple(d -> (lo[d] + hi[d]) / 2, D)
    origin = ntuple(d -> ctr[d] - L / 2, D)
    return PhysicalGrid{D}(Grid{D}(n, b), origin, float(L))
end

SparsifyAndSweep.dim(::PhysicalGrid{D}) where {D} = D

"Physical coordinates of the Cartesian node with integer index `i ∈ {1,…,n}^D`."
@inline physcoords(pg::PhysicalGrid{D}, i::NTuple{D,Int}) where {D} =
    ntuple(d -> pg.origin[d] + pg.L * (i[d] * pg.g.h), D)

"Unit-cube coordinates `ξ ∈ (0,1)^D` of a physical point."
@inline unitcoords(pg::PhysicalGrid{D}, x) where {D} =
    ntuple(d -> (x[d] - pg.origin[d]) / pg.L, D)

"""
    unit_frequency(pg, ω)

The frequency the core solver must be given so that its unit-cube problem is the
physical one: `ω′ = ω L`.
"""
unit_frequency(pg::PhysicalGrid, ω::Real) = ω * pg.L

"""
    sample_field(f, pg)

Evaluate `f` at every Cartesian node and return it shaped as `n^D`, ready to
hand to the core solver.  `f` takes a physical coordinate tuple.
"""
function sample_field(f, pg::PhysicalGrid{D}) where {D}
    n = pg.g.n
    out = Array{Float64,D}(undef, ntuple(_ -> n, D))
    for c in CartesianIndices(out)
        out[c] = f(physcoords(pg, Tuple(c)))
    end
    return out
end

"""
    check_support(m, pg; margin = pg.g.b)

Errors if the sampled medium fails to vanish within `margin` layers of the edge
of the cube, which the PML and the Lippmann–Schwinger formulation both require.
Otherwise, returns the largest `|m|` found in that shell.
"""
function check_support(m::AbstractArray{Float64,D}, pg::PhysicalGrid{D};
                       margin::Integer = pg.g.b, atol::Real = 1e-10) where {D}
    n = pg.g.n
    worst = 0.0
    for c in CartesianIndices(m)
        i = Tuple(c)
        any(d -> i[d] <= margin || i[d] > n - margin, 1:D) || continue
        worst = max(worst, abs(m[c]))
    end
    worst > atol && throw(ArgumentError(
        "the medium does not vanish within $margin layers of the box edge " *
        "(max |m| there is $worst); increase `pad` when building the PhysicalGrid"))
    return worst
end

"""
    velocity_from_contrast(m)

The core solver is parameterized by the velocity `c` with `m = 1 - 1/c²`; the
hybrid solver is parameterized by `m` directly.  This inverts the relation,
`c = 1/√(1-m)`, which needs `m < 1` (true whenever the local wave speed is
finite).
"""
function velocity_from_contrast(m::AbstractArray)
    maximum(m) < 1 || throw(ArgumentError("need m < 1 everywhere (got max m = $(maximum(m)))"))
    return @. 1 / sqrt(1 - m)
end
