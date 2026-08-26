"""
    ParametricChart(param, lc, hc; period = nothing, inverse = nothing)

A local parametrization of (part of) the curved boundary of a domain -- a plane
curve in 2D, a surface in 3D -- to be passed to [`curve_mesh`](@ref).

`param` maps the parameter box `[lc, hc] ⊂ ℝᴹ` into the ambient space `ℝᴺ` (with
`M = N - 1`), and is assumed to be a smooth injective immersion on that box.
Note that `param` takes an `SVector{M}` in both dimensions: a chart of a plane
curve is called as `param(SVector(t))`, not `param(t)`.

A vector of charts whose images cover the curved part of the boundary is an
*atlas*: `curve_mesh` curves each element using a *single* chart of the atlas
which contains all of the element's boundary nodes, so the charts must overlap
enough that every boundary element fits inside at least one of them (in 3D) or
may also be abutting in the case of piece-wise smooth curves in 2D.

A single chart suffices for boundaries admitting a global smooth parametrization
(any closed plane curve, or e.g. a torus), but not for e.g. a sphere, where any
single chart is either degenerate (spherical coordinates at the poles) or
non-injective.

Nothing is assumed about how the charts of an atlas relate to one another on
their overlaps: in 3D, `curve_mesh` builds curved faces by projection onto the
surface by default, which is chart-independent, so an arbitrary collection of
overlapping patches is fine -- see `curve_mesh`'s `face_map` keyword. In 2D the
question does not arise, since a curved edge is pinned at both endpoints by
nodes shared exactly with its neighbours.

## Keyword arguments

- `period`: vector of periods of `param`, with `Inf` marking non-periodic
  directions (the default in every direction). Parameters of nodes sharing an
  element are unwrapped modulo `period` so that they are mutually closest; this
  is what allows e.g. a torus to be curved with the single chart
  `[0,2π] × [0,2π]`.
- `inverse`: a function `x -> α` returning `α` with `param(α) ≈ x`, or `nothing`
  if `x` is not on the chart. When omitted, an inverse is built by sampling
  `param` on a grid and refining the nearest sample with a Gauss--Newton
  iteration.

See also [`curve_mesh`](@ref).
"""
struct ParametricChart{M, F, I}
    param::F
    lc::SVector{M, Float64}
    hc::SVector{M, Float64}
    period::SVector{M, Float64}
    inverse::I
end

function ParametricChart(param, lc, hc; period = nothing, inverse = nothing)
    M = length(lc)
    length(hc) == M || error("`lc` and `hc` must have the same length")
    per = if isnothing(period)
        SVector{M, Float64}(ntuple(_ -> Inf, M))
    else
        SVector{M, Float64}(period)
    end
    return ParametricChart{M, typeof(param), typeof(inverse)}(
        param,
        SVector{M, Float64}(lc),
        SVector{M, Float64}(hc),
        per,
        inverse,
    )
end

parameter_dimension(::ParametricChart{M}) where {M} = M

"""
    _unwrap(α, ref, period)

Shift each component of `α` by an integer multiple of `period` so as to land as
close as possible to `ref`. Components with an infinite period are untouched.
"""
function _unwrap(α::SVector{M, Float64}, ref::SVector{M, Float64}, period) where {M}
    return SVector{M, Float64}(
        ntuple(M) do i
            p = period[i]
            isfinite(p) && p > 0 ? α[i] + p * round((ref[i] - α[i]) / p) : α[i]
        end,
    )
end

"""
    _wrap_into_box(chart, α)

Bring the periodic components of `α` back into the chart's parameter box. Nodes
lying on the seam of a periodic chart are otherwise liable to be found just
outside the box.
"""
function _wrap_into_box(c::ParametricChart{M}, α::SVector{M, Float64}) where {M}
    return SVector{M, Float64}(
        ntuple(M) do i
            p = c.period[i]
            if isfinite(p) && p > 0
                c.lc[i] + mod(α[i] - c.lc[i], p)
            else
                α[i]
            end
        end,
    )
end

"""
    _project_on_chart(chart, x, α₀; maxiter, rtol)

Gauss--Newton iteration for the parameter `α` minimizing `‖param(α) - x‖`,
started from `α₀`. Returns `nothing` if the iteration leaves the parameter box
or fails to converge.
"""
function _project_on_chart(
        c::ParametricChart{M},
        x::SVector{N, Float64},
        α₀::SVector{M, Float64};
        maxiter = 20,
        rtol = 1.0e-13,
    ) where {M, N}
    ψ = c.param
    w = c.hc - c.lc # box size, used to bound the step and to detect escapes
    α = α₀
    scale = max(norm(x), one(Float64))
    for _ in 1:maxiter
        J = ForwardDiff.jacobian(a -> SVector{N}(ψ(a)), α)
        r = SVector{N, Float64}(ψ(α)) - x
        norm(r) ≤ rtol * scale && return α
        δ = -(J \ r)
        # `J` has more rows than columns, so this is a least-squares step and the
        # residual has a floor: `x` need not lie exactly on the chart, and a
        # mesh's boundary nodes usually only approximate it (a curve meshed from
        # a spline is off by O(hᵏ)). Convergence is therefore judged on the step,
        # and how far the converged point may be from `x` is left to
        # `_atlas_node_params`' `atol` to decide.
        norm(δ) ≤ rtol * max(norm(α), one(Float64)) && return α
        # damp steps which would jump clear across the chart, which happens when
        # `x` is not on this chart at all
        s = maximum(abs.(δ) ./ w)
        s > 1 && (δ = δ / s)
        α = α + δ
        # allow the iterate to wander a little outside the box before giving up
        all(c.lc - w .≤ α .≤ c.hc + w) || return nothing
    end
    # the iteration did not settle: report no parameter rather than a doubtful one
    return nothing
end

"""
    _closest_point_on_chart(chart, y, α₀, niter, maxexcursion, second_order)

Point of the chart's image closest to `y`, found by `niter` Gauss--Newton steps
from the parameter `α₀`.

Unlike the parametrization itself, the closest point is a *geometric* object: it
does not depend on which chart of an atlas is used to compute it. This is what
lets [`curve_mesh`](@ref) build curved faces that agree across elements curved
through different charts, for atlases whose charts are not otherwise compatible.

The iteration count is fixed rather than adaptive so that the result is a smooth
function of `y`, which it must be for the curved element map to be
differentiable.

With `second_order = false` the step is Gauss--Newton, and convergence is
*linear*, not quadratic: the residual `ψ(α) - y` does not vanish at the solution,
since `y` sits off the surface by the sagitta of the flat face, and the
contraction factor is `≈ ‖r‖/ρ` with `ρ` the smallest principal radius of
curvature at the foot point. `α₀` is accurate to `O(h²)`, so a handful of steps
still suffice, but the tail is geometric rather than self-doubling -- which is
why the residual gap between neighbouring curved faces keeps shrinking up to
`niter ≈ 8`--`12` instead of bottoming out at `niter ≈ 4`.

`second_order = true` restores the term that Gauss--Newton drops. For
`f(α) = ½‖ψ(α) - y‖²` the exact Hessian is `JᵀJ + Σᵢ rᵢ ∇²ψᵢ`, and the second
term is what carries the surface's curvature; dropping it is precisely what costs
the quadratic rate. It is had for one scalar Hessian rather than `N` component
ones, since `Σᵢ rᵢ ∇²ψᵢ = ∇²⟨r, ψ⟩` with `r` held fixed. Measured on the unit
sphere from an `O(h²)` start at `h = 0.4`, the parameter error goes
`1.6e-2, 2.7e-4, 8.4e-8, 8.4e-15` against Gauss--Newton's
`1.6e-2, 1.6e-4, 2.4e-6, 3.7e-8, ...`: machine precision in three steps instead
of eight. This is the curvature correction underlying the second-order geometric
iteration of Hu and Wallner (*A second order algorithm for orthogonal projection
onto curves and surfaces*, CAGD 22 (2005) 251--260), taken as an exact Newton
step rather than through their osculating-circle construction -- the same order,
reusing the derivatives already at hand.

The full Hessian has a second use. Writing `r = ∓d n̂` at the foot point makes
the curvature term `∓d·II`, so the Hessian is `I₁ ∓ d·II`, the first fundamental
form of the surface offset by `d`. That is positive definite exactly while `d`
stays below every principal radius of curvature -- which is to say exactly while
`y` lies within the reach, where the closest point is unique. So a Hessian that
fails to be positive definite is a sharp, local certificate that `y` has reached
the medial axis, and the step falls back to the Gauss--Newton one, which is
always definite. `JᵀJ` alone cannot see this: on the unit sphere its determinant
is `1` for every `d`, while the full Hessian's is `(1-d)²`.

That same contraction factor measures proximity to the medial axis: `‖r‖/ρ < 1`
is exactly the condition that `y` lie within the *reach* of the surface, where
the closest point is unique. As `‖r‖/ρ → 1` uniqueness is lost and the iteration
can walk onto a different sheet of the surface -- a *branch mismatch*, which
leaves a gap the size of the local thickness between this face and its
neighbour, and which no amount of extra iterations or quadrature order can fix.
`maxexcursion` bounds `‖α - α₀‖` to hold the iterate on the branch that `α₀`
selects. The correct root lies within `O(h²)` of `α₀`, so a radius of a couple
of element widths never binds on a mesh that resolves the surface. When it does
bind the result is continuous but not differentiable in `y`; that is a
deliberate trade, the alternative being a foot point on the wrong sheet, which
is discontinuous.
"""
function _closest_point_on_chart(
        c::ParametricChart{M},
        y::SVector{N},
        α₀,
        niter,
        maxexcursion,
        second_order::Bool = false,
    ) where {M, N}
    ψ = c.param
    w = c.hc - c.lc # box size, used to bound a single step
    α = α₀
    for _ in 1:niter
        J = ForwardDiff.jacobian(a -> SVector{N}(ψ(a)), α)
        r = SVector{N}(ψ(α)) - y
        Jt = transpose(J)
        A = Jt * J
        if second_order
            # the curvature term Gauss--Newton drops, as one Hessian of the
            # scalar ⟨r, ψ⟩ with `r` frozen rather than N component Hessians
            B = A + ForwardDiff.hessian(a -> dot(r, SVector{N}(ψ(a))), α)
            # B is the offset surface's metric and so is definite exactly while
            # `y` is within the reach; where it is not, `y` has reached the
            # medial axis and B no longer models a minimum, so fall back
            isposdef(Symmetric(B)) && (A = B)
        end
        δ = -(A \ (Jt * r))
        # damp a step which would jump clear across the chart
        s = maximum(abs.(δ) ./ w)
        s > 1 && (δ = δ / s)
        α = α + δ
        # hold the iterate on the branch selected by α₀
        d = norm(α - α₀)
        d > maxexcursion && (α = α₀ + (maxexcursion / d) * (α - α₀))
    end
    return SVector{N}(ψ(α))
end

"""
    _simplex_check_points(shape, n)

Interior lattice points of the reference simplex, all barycentric coordinates at
least `1/n`. The vertices are deliberately excluded: the curved map divides by
`x₁+x₂`, which vanishes at some of them -- at `(0,0)` for a triangle, and at two
vertices of a tetrahedron with only an edge on the surface.
"""
function _simplex_check_points(::ReferenceTriangle, n)
    return [SVector(i, j) ./ n for i in 1:n for j in 1:n if i + j <= n - 1]
end

function _simplex_check_points(::ReferenceTetrahedron, n)
    return [
        SVector(i, j, k) ./ n for i in 1:n for j in 1:n for k in 1:n if
            i + j + k <= n - 1
    ]
end

"""
    _check_element_jacobians(els, mode; npts)

Look for curved elements whose Jacobian determinant is non-positive somewhere,
i.e. which the curving has folded over themselves. The affine part of every
curved element is built with a positive determinant, so a valid element has
`det > 0` throughout.

`mode` is `:warn` (default), `:error`, or `:none`. Sampling is on a lattice, so
this detects folds but does not assure their absence.
"""
function _check_element_jacobians(els, mode; npts = 6)
    mode === :none && return nothing
    mode ∈ (:warn, :error) ||
        error("`check_jacobian` must be one of :warn, :error, :none")
    isempty(els) && return nothing
    # the reference shape gives both the lattice to sample and the ambient
    # dimension; `return_type` would give a `Float64` eltype, which the
    # ForwardDiff duals below cannot be converted to
    d = domain(first(els))
    pts = _simplex_check_points(d, npts)
    N = length(first(pts))
    nbad, worst, kworst = 0, Inf, 0
    for (k, el) in enumerate(els)
        m = minimum(pts) do p
            det(ForwardDiff.jacobian(u -> SVector{N}(el(u)), p))
        end
        m <= 0 && (nbad += 1)
        if m < worst
            worst, kworst = m, k
        end
    end
    nbad == 0 && return nothing
    msg = """
    $nbad of $(length(els)) curved elements are tangled: their Jacobian determinant \
    changes sign, so the curved mesh overlaps itself and integrals over it will be \
    wrong by an amount that refining the quadrature cannot fix. The worst is element \
    $kworst, with min det = $worst. Refine the mesh near the boundary, or lower the \
    smoothness order."""
    mode === :error ? error(msg) : @warn msg
    return nothing
end

# A curved face may not stray from the flat face spanned by its own three
# surface vertices by more than this multiple of that face's diameter. The
# legitimate excursion is the sagitta, which is `O(d²/ρ)` for a radius of
# curvature `ρ` and so shrinks linearly in the ratio `d/ρ`. Measured on the unit
# sphere, the largest ratio over all faces was 0.147 at `h = 0.4`, 0.078 at
# `h = 0.25` and 0.057 at `h = 0.15` (`:chart`; `:projection` is smaller
# throughout), so this leaves a factor of three in hand on a mesh already coarse
# for the geometry. A real branch mismatch overshoots by of order a diameter or
# more, so the two are well separated.
const _FACE_EXCURSION_TOL = 0.5

"""
    _report_far_faces(far, mode)

Report curved faces flagged by the excursion check, `far` being a vector of
`(element index, excursion / face diameter)` pairs.

An excursion much larger than the face itself means the curved face is not
sitting over the flat one at all. Under `face_map = :projection` the usual cause
is a branch mismatch: the flat face reached the medial axis, so the closest point
is not unique and the Gauss--Newton iteration found a foot point on another sheet
of the surface, leaving a gap against the neighbouring face. Under
`face_map = :chart` it instead points to a chart bookkeeping fault -- the face
being curved through a chart which does not really cover it.

The check is one-sided: it catches excursions large *relative to the element*, so
a branch mismatch on a body thinner than one element slips through. That regime
is caught separately, by the refusal to curve a tetrahedron with all four
vertices on the surface.
"""
function _report_far_faces(far, mode)
    (mode === :none || isempty(far)) && return nothing
    kworst, rworst = far[argmax(last.(far))]
    msg = """
    $(length(far)) curved face(s) stray more than $(_FACE_EXCURSION_TOL) of their own \
    diameter from the flat face spanned by their surface vertices; the worst is \
    element $kworst, at $(round(rworst, sigdigits = 3)) diameters. The curved face is \
    not lying over the flat one, so it will not meet its neighbours: expect a gap of \
    that size, which neither more `projection_iterations` nor a higher `qorder` can \
    close. With `face_map = :projection` this is the medial axis being reached by a \
    flat face -- refine the mesh where the surface is thin or sharply curved. With \
    `face_map = :chart` it indicates a face curved through a chart that does not \
    cover it."""
    mode === :error ? error(msg) : @warn msg
    return nothing
end

"""
    _chart_locator(chart, nsample)

Build a callable `x -> α` returning the parameter of `x` on `chart`, or
`nothing` if `x` is not on the chart. Uses `chart.inverse` when available, and
otherwise a `nsample`-per-direction sampling of the parameter box refined by
[`_project_on_chart`](@ref).
"""
function _chart_locator(c::ParametricChart{M}, ::Type{SVector{N, Float64}}, nsample) where {M, N}
    isnothing(c.inverse) || return c.inverse
    rngs = ntuple(i -> LinRange(c.lc[i], c.hc[i], nsample), M)
    αs = [SVector{M, Float64}(t) for t in Iterators.product(rngs...)]
    αs = reshape(αs, :)
    pts = [SVector{N, Float64}(c.param(α)) for α in αs]
    tree = KDTree(pts; reorder = false)
    return function (x)
        idx, _ = nn(tree, SVector{N, Float64}(x))
        return _project_on_chart(c, SVector{N, Float64}(x), αs[idx])
    end
end

"""
    _atlas_node_params(atlas, locators, nodes, idxs; atol)

For every node index in `idxs`, compute the parameters of `nodes[i]` on each
chart of `atlas` that contains it. Returns a `Dict` mapping the node index to a
`Dict` of `chart index => parameter`; nodes lying on no chart are absent.

A node is considered to lie on a chart when the chart's inverse succeeds, the
parameter is inside the chart's box, and the resulting point is within `atol` of
the node.
"""
function _atlas_node_params(
        atlas,
        locators,
        nodes::AbstractVector{SVector{N, Float64}},
        idxs,
        ::Val{M};
        atol,
    ) where {M, N}
    out = Dict{Int, Dict{Int, SVector{M, Float64}}}()
    for i in idxs
        x = nodes[i]
        found = Dict{Int, SVector{M, Float64}}()
        for (cid, c) in enumerate(atlas)
            α = locators[cid](x)
            isnothing(α) && continue
            α = _wrap_into_box(c, SVector{M, Float64}(α))
            tol = 1.0e-8 * (c.hc - c.lc)
            all(c.lc - tol .≤ α .≤ c.hc + tol) || continue
            norm(SVector{N, Float64}(c.param(α)) - x) ≤ atol || continue
            found[cid] = α
        end
        isempty(found) || (out[i] = found)
    end
    return out
end

"""
    _chart_margin(chart, α)

Relative distance of the parameter `α` to the boundary of the chart's box, in
`[0, 1/2]`. Used to select, among the charts containing an element, the one on
which the element sits most comfortably.
"""
function _chart_margin(c::ParametricChart{M}, α) where {M}
    return minimum(
        ntuple(M) do i
            w = c.hc[i] - c.lc[i]
            w > 0 ? min(α[i] - c.lc[i], c.hc[i] - α[i]) / w : 0.0
        end,
    )
end

"""
    _select_chart(atlas, node_params, idxs)

Index of the chart of `atlas` containing every node of `idxs` with the largest
margin, or `nothing` when no single chart contains them all.
"""
function _select_chart(atlas, node_params, idxs)
    best, best_margin = nothing, -Inf
    for cid in eachindex(atlas)
        m = Inf
        ok = true
        for i in idxs
            p = get(node_params, i, nothing)
            if isnothing(p) || !haskey(p, cid)
                ok = false
                break
            end
            m = min(m, _chart_margin(atlas[cid], p[cid]))
        end
        ok || continue
        if m > best_margin
            best, best_margin = cid, m
        end
    end
    return best
end

"""
    Atlas{M}

An *atlas* of the curved boundary: a vector of [`ParametricChart`](@ref)s of
parameter dimension `M` whose images cover it. `M = 1` in 2D, `M = 2` in 3D.
"""
const Atlas{M} = AbstractVector{<:ParametricChart{M}}

"""
    curve_mesh(msh, atlas_by_ent::Dict{EntityKey,<:Atlas}, order; kwargs...)
    curve_mesh(msh, atlas::AbstractVector{<:ParametricChart}, order; kwargs...)
    curve_mesh(msh, ψ, order; kwargs...)
    curve_mesh(msh, by_ent::AbstractDict{EntityKey}, order; kwargs...)

Return a new mesh in which the elements of `msh` touching the parametrized
boundary are replaced by curved (`ParametricElement`) ones of smoothness order
`order`, following the construction of Bernardi
[bernardi1989optimal](@cite). Elements away from the boundary are copied over
unchanged.

The curved boundary is described by an *atlas* of overlapping
[`ParametricChart`](@ref)s covering it, given per volume entity. The remaining
three forms are conveniences which normalize to that one:

- a bare atlas is applied to the mesh's sole volume entity;
- a bare parametrization `ψ` becomes a single-chart atlas -- in 2D the chart
  `[0,1]` with period `1`, in 3D the chart `[0,2π] × [0,2π]` with period `2π` in
  both directions. A closed plane curve or a torus can therefore be curved with
  a bare `ψ`, but a sphere needs a genuine atlas; see [`sphere_atlas`](@ref);
- a `Dict` mapping each volume entity to either a parametrization or an atlas.

As a convenience, a bare `ψ` for a 2D mesh may take either a scalar or an
`SVector{1}`. A chart's `param` always takes an `SVector{M}`.

## Keyword arguments

- `face_element_on_curved_surface`: predicate on the nodes of a boundary face
  deciding whether it lies on the curved boundary. Defaults to all faces.
- `patch_sample_num`: number of samples per parameter direction used to seed the
  inversion of charts which do not provide an `inverse`. Defaults to a value
  derived from the number of boundary elements.
- `chart_atol`: distance below which a chart is deemed to reproduce a boundary
  node, used to decide which charts a node belongs to. Defaults to a quarter of
  the largest boundary edge.
- `check_jacobian`: whether to look for curved elements the curving has damaged,
  which corrupt integrals over the mesh by an amount no quadrature order can
  reduce. One of `:warn` (the default), `:error`, or `:none`. Two faults are
  looked for: elements folded over themselves, found by sampling the Jacobian
  determinant for a sign change, and -- in 3D only -- curved faces that stray
  far from the flat face spanned by their own surface vertices, which signals a
  face laid somewhere other than over its neighbours, with `:projection`
  typically because a flat face reached the medial axis, where the closest point
  stops being unique. Both checks sample on a lattice, so they detect these
  faults but do not certify their absence. The cost scales with the number of
  curved elements: in 3D it was measured between two and three and a half times
  the cost of `curve_mesh` itself -- which still leaves it around a fifth of the
  cost of curving a mesh and then building a `Quadrature` over it -- and in 2D,
  where the elements are cheaper and fewer points are sampled, about a quarter
  of it.
- `face_map` (3D only): how a curved face is laid over the surface. `:chart`
  composes the chart's parametrization with an affine map in its parameter
  plane, which is fast but only consistent between elements curved through
  different charts when the atlas' transition maps are projective.
  `:projection` uses the closest point of the surface to the flat face, which is
  chart-independent and therefore watertight for *any* atlas, at the cost of a
  small solve per evaluation. The default `:auto` picks `:chart` for a
  single-chart atlas and `:projection` otherwise. The question does not arise in
  2D, where a curved edge is pinned at both ends by nodes shared exactly with
  its neighbours, so `:chart` is watertight for any 2D atlas.
- `projection_iterations` (3D only): number of Gauss--Newton steps used by
  `face_map = :projection` (default `4`). This controls only how nearly the
  projection has converged, visible as the residual gap between neighbouring
  curved faces; the accuracy of integrals over the resulting mesh is limited by
  `qorder` instead, so raise that rather than this if `:projection` is not
  accurate enough. Under the default `projection_method` convergence is linear
  rather than quadratic -- the closest point problem has a non-vanishing residual,
  the sagitta of the flat face -- so the gap keeps shrinking up to `n ≈ 8`--`12`
  at usual mesh sizes, reaching machine precision there, after which larger
  values change nothing.
- `projection_method`: which step `face_map = :projection` takes.
  `:gauss_newton` (the default) drops the surface's curvature from the Hessian,
  and so converges linearly. `:newton` restores it, recovering the quadratic rate
  and reaching machine precision in about three steps rather than eight; it is the
  curvature correction behind the second-order geometric iteration of Hu and
  Wallner (CAGD 22 (2005) 251--260), taken as an exact Newton step. Each step
  costs one extra Hessian, so lower `projection_iterations` alongside it -- the
  two together are what make it pay. It also detects the medial axis exactly,
  where the default can only be slowed down by it: the Newton Hessian is the
  offset surface's metric and loses positive definiteness precisely when the
  closest point stops being unique, and the step falls back to Gauss--Newton
  there.
"""
function curve_mesh end

"""
    _default_chart(ψ, ::Val{N})

Wrap a bare parametrization as the single chart `curve_mesh` interprets it as:
`[0,1]` with period `1` in 2D, `[0,2π] × [0,2π]` with period `2π` in 3D. The
2D wrapper also accepts a `ψ` taking a scalar rather than an `SVector{1}`.
"""
function _default_chart(ψ, ::Val{2})
    return ParametricChart(
        α -> ψ(α[1]),
        SVector(0.0),
        SVector(1.0);
        period = SVector(1.0),
    )
end

function _default_chart(ψ, ::Val{3})
    return ParametricChart(ψ, SVector(0.0, 0.0), SVector(2π, 2π); period = SVector(2π, 2π))
end

"""
    _as_atlas(x, ::Val{N})

Normalize one `curve_mesh` boundary description -- a bare parametrization or an
atlas -- into an atlas of charts of parameter dimension `N-1`.
"""
_as_atlas(ψ, v::Val{N}) where {N} = [_default_chart(ψ, v)]

# a vector is always meant as an atlas, so say so rather than treat it as a
# parametrization and fail later on a chart that cannot be evaluated
function _as_atlas(x::AbstractVector, v::Val{N}) where {N}
    all(c -> c isa ParametricChart, x) || error(
        "a vector is interpreted as an atlas, but this one holds $(eltype(x))s rather than `ParametricChart`s",
    )
    return _as_atlas([c::ParametricChart for c in x], v)
end

function _as_atlas(atlas::AbstractVector{<:ParametricChart}, ::Val{N}) where {N}
    isempty(atlas) && error("the atlas must contain at least one chart")
    for (cid, c) in enumerate(atlas)
        parameter_dimension(c) == N - 1 || error(
            "chart $cid of the atlas has parameter dimension $(parameter_dimension(c)), but curving a mesh in $(N)D needs charts of parameter dimension $(N - 1)",
        )
    end
    return atlas
end

"""
    _sole_volume_entity(msh, ::Val{N})

The mesh's single volume entity, erroring if there is more than one -- the case
in which the caller must say which parametrization goes with which entity.
"""
function _sole_volume_entity(msh::Mesh{N, Float64}, ::Val{N}) where {N}
    ent = nothing
    for e in entities(msh)
        e.dim == N || continue
        isnothing(ent) || error(
            "Trying to curve mesh with multiple volumetric entities, but only one parametrization passed; pass in an entity => parametrization dictionary",
        )
        ent = e
    end
    isnothing(ent) && error("the mesh has no volume entity to curve")
    return ent
end

# A bare parametrization or a bare atlas applies to the mesh's sole volume
# entity; a `Dict` says which description goes with which entity. Everything
# funnels into the canonical `Dict{EntityKey,<:Atlas}` method below.
function curve_mesh(msh::Mesh{N, Float64}, ψ::Function, order::Int; kwargs...) where {N}
    ent = _sole_volume_entity(msh, Val(N))
    return curve_mesh(msh, Dict(ent => _as_atlas(ψ, Val(N))), order; kwargs...)
end

function curve_mesh(
        msh::Mesh{N, Float64},
        atlas::AbstractVector{<:ParametricChart},
        order::Int;
        kwargs...,
    ) where {N}
    ent = _sole_volume_entity(msh, Val(N))
    return curve_mesh(msh, Dict(ent => _as_atlas(atlas, Val(N))), order; kwargs...)
end

# Deliberately typed on the key alone: `Dict(e1 => ψ1, e2 => ψ2)` infers a value
# type which varies with the literal, and a `Dict{EntityKey,Any}` may mix bare
# parametrizations with atlases. Each value is normalized on its own.
function curve_mesh(
        msh::Mesh{N, Float64},
        by_ent::AbstractDict{EntityKey},
        order::Int;
        kwargs...,
    ) where {N}
    atlas_by_ent = Dict{EntityKey, AbstractVector{<:ParametricChart{N - 1}}}()
    for (ent, x) in by_ent
        atlas_by_ent[ent] = _as_atlas(x, Val(N))
    end
    return curve_mesh(msh, atlas_by_ent, order; kwargs...)
end

function curve_mesh(
        msh::Mesh{2, Float64},
        atlas_by_ent::Dict{EntityKey, <:Atlas},
        order::Int;
        patch_sample_num = nothing,
        face_element_on_curved_surface = nothing,
        chart_atol = nothing,
        face_map = :auto,
        projection_iterations = 4,
        check_jacobian = :warn,
    )
    order > 0 || error("smoothness order must be positive")
    # implemented up to order=6 below but can be easily extended
    order <= 6 || notimplemented()
    # both keywords describe how a curved *face* is laid over a surface, which
    # has no 2D counterpart; naming them beats the unsupported-keyword
    # `MethodError` that the generic wrapper would otherwise raise
    face_map === :auto || error(
        "`face_map` is meaningful only in 3D: a curved edge in 2D is pinned at both ends by nodes shared exactly with its neighbours, so it is watertight for any atlas",
    )
    projection_iterations == 4 || error(
        "`projection_iterations` is meaningful only in 3D, where it controls `face_map = :projection`",
    )
    E_straight_bdry = LagrangeElement{ReferenceHyperCube{1}, 2, SVector{2, Float64}}
    E_straight = LagrangeElement{ReferenceSimplex{2}, 3, SVector{2, Float64}} # TODO fix this to auto be a Pk element type

    if isnothing(face_element_on_curved_surface)
        face_element_on_curved_surface = (arg) -> true
    end

    if isnothing(patch_sample_num)
        patch_sample_num = -1
        for ent in entities(msh)
            ent.dim == 1 || continue
            patch_sample_num =
                max(patch_sample_num, 10 * length(msh.ent2etags[ent][E_straight_bdry]))
        end
    end

    # A volume entity may be bounded by several entities -- one per smooth piece
    # of its boundary, which is how a mesher is asked to put a vertex at each
    # junction. Which curved edge belongs to which of them is settled per edge
    # below, not per volume entity.
    bdryent_to_volent = Dict{EntityKey, EntityKey}()
    for ent in entities(msh)
        ent.dim == 2 || continue
        # TODO error out if volume entities have nontrivial pairwise boundary intersections
        haskey(atlas_by_ent, ent) ||
            error("no parametrization was given for the volume entity $ent")
        for bdryent in boundary(ent)
            haskey(bdryent_to_volent, bdryent) && error(
                "the boundary entity $bdryent bounds more than one volume entity, which curving does not (currently) support",
            )
            bdryent_to_volent[bdryent] = ent
        end
    end

    crvmsh = Mesh{2, Float64}()
    (; nodes, etype2mat, etype2els, ent2etags) = crvmsh
    foreach(k -> ent2etags[k] = OrderedDict{DataType, Vector{Int}}(), entities(msh))
    append!(nodes, msh.nodes)

    nbdry_els = size(msh.etype2mat[E_straight_bdry])[2]

    el2ent = Dict{Tuple{DataType, Int}, Inti.EntityKey}()
    for (ent, etype2tags) in msh.ent2etags
        for (E, tags) in etype2tags
            for t in tags
                el2ent[(E, t)] = ent
            end
        end
    end

    # Collect, per volume entity, the nodes of the boundary elements lying on
    # the curved part of the boundary, and the longest such edge
    bdry_node_idx_by_ent = Dict{EntityKey, Vector{Int64}}()
    # the boundary entity each curved edge came from, keyed by its (sorted) pair
    # of nodes, so that a curved edge can be filed under the right one even when
    # its volume entity has several
    bdryent_by_edge = Dict{Tuple{Int64, Int64}, EntityKey}()
    max_bdry_edge = 0.0
    for elind in 1:nbdry_els
        bdryent = el2ent[(E_straight_bdry, elind)]
        ent = get(bdryent_to_volent, bdryent, nothing)
        isnothing(ent) && continue
        node_indices = msh.etype2mat[E_straight_bdry][:, elind]
        straight_nodes = crvmsh.nodes[node_indices]
        face_element_on_curved_surface(straight_nodes) || continue
        append!(get!(bdry_node_idx_by_ent, ent, Int64[]), node_indices)
        bdryent_by_edge[minmax(node_indices[1], node_indices[2])] = bdryent
        max_bdry_edge = max(max_bdry_edge, norm(straight_nodes[1] - straight_nodes[2]))
    end

    # A node is deemed to lie on a chart only if the chart reproduces it to
    # within `chart_atol`; the default is loose enough to tolerate a mesh whose
    # boundary nodes only approximately lie on the parametrized curve, but
    # tight enough to reject charts which simply do not cover the node.
    isnothing(chart_atol) && (chart_atol = max_bdry_edge / 4)

    # Locate every boundary node on every chart of its entity's atlas, then move
    # it onto the curve using the chart it sits most comfortably on. This is the
    # same machinery the 3D path uses, at parameter dimension one.
    node_to_param_by_ent = Dict{EntityKey, Dict{Int, Dict{Int, SVector{1, Float64}}}}()
    for ent in collect(keys(bdry_node_idx_by_ent))
        atlas = atlas_by_ent[ent]
        bdry_node_idx = unique(bdry_node_idx_by_ent[ent])
        locators =
            [_chart_locator(c, SVector{2, Float64}, patch_sample_num) for c in atlas]
        node_to_param = _atlas_node_params(
            atlas,
            locators,
            crvmsh.nodes,
            bdry_node_idx,
            Val(1);
            atol = chart_atol,
        )
        orphans = setdiff(bdry_node_idx, keys(node_to_param))
        isempty(orphans) || error(
            "$(length(orphans)) boundary node(s) of the curved boundary of entity $ent are not covered by any chart of the atlas (e.g. node $(first(orphans)) at $(crvmsh.nodes[first(orphans)])); enlarge the charts' parameter boxes or add charts",
        )
        for i in keys(node_to_param)
            cid = _select_chart(atlas, node_to_param, (i,))
            crvmsh.nodes[i] = SVector{2, Float64}(atlas[cid].param(node_to_param[i][cid]))
        end
        node_to_param_by_ent[ent] = node_to_param
        bdry_node_idx_by_ent[ent] = collect(keys(node_to_param))
    end

    connect_straight = Int[]
    connect_curve = Int[]
    connect_curve_bdry = Int[]
    # TODO Could use an ElementIterator for straight elements
    els_straight = []
    els_curve = []
    els_curve_bdry = []

    for E in element_types(msh)
        # The purpose of this check is to see if other element types are present in
        # the mesh, such as e.g. quads; This code errors when encountering a quad,
        # but the method can be extended to transfer straight quads to the new mesh,
        # similar to how straight simplices are transferred below.
        E <: Union{
            LagrangeElement{ReferenceSimplex{2}},
            LagrangeElement{ReferenceHyperCube{1}},
            SVector,
        } || error()
        E <: SVector && continue
        E <: LagrangeElement{ReferenceHyperCube{1}} && continue
        E_straight_bdry = LagrangeElement{ReferenceHyperCube{1}, 2, SVector{2, Float64}}
        els = elements(msh, E)
        for elind in eachindex(els)
            node_indices = @view msh.etype2mat[E][:, elind]
            straight_nodes = @view crvmsh.nodes[node_indices]

            # First determine entity to which volume element belongs
            ent = el2ent[(E_straight, elind)]
            atlas = atlas_by_ent[ent]
            node_to_param = node_to_param_by_ent[ent]
            bdry_node_idx = bdry_node_idx_by_ent[ent]

            # Next determine if straight or curved
            verts_on_bdry = findall(x -> x ∈ bdry_node_idx, node_indices)
            j = length(verts_on_bdry) # j in C. Bernardi SINUM Sec. 6
            if j > 1
                append!(connect_curve, node_indices)
                node_indices_on_bdry = node_indices[verts_on_bdry]

                # The affine map below sends the reference vertices (1,0), (0,1)
                # and (0,0) to `nb[2]`, `nb[1]` and the interior vertex `bₖ`
                # respectively, so its Jacobian determinant is
                # det(nb[2] - bₖ, nb[1] - bₖ); a negative one folds the element
                # over itself. Gmsh happens to emit the two boundary vertices at
                # local positions (1,3), which makes it positive, but nothing in
                # a mesh guarantees that -- so orient the pair explicitly rather
                # than inherit the convention.
                b_idx = setdiff(node_indices, node_indices_on_bdry)[1]
                if det(
                        hcat(
                            crvmsh.nodes[node_indices_on_bdry[2]] - crvmsh.nodes[b_idx],
                            crvmsh.nodes[node_indices_on_bdry[1]] - crvmsh.nodes[b_idx],
                        ),
                    ) < 0
                    reverse!(node_indices_on_bdry)
                end
                append!(connect_curve_bdry, node_indices_on_bdry)

                # Curve the edge through the chart on which it sits most
                # comfortably; every element is curved through a single chart,
                # so the atlas must cover each boundary edge entirely.
                cid = _select_chart(atlas, node_to_param, node_indices_on_bdry)
                isnothing(cid) && error(
                    "no single chart of the atlas contains the boundary nodes $(node_indices_on_bdry) of element $elind; every boundary edge must lie inside one chart, so either overlap the charts by at least one element or place their seams on mesh nodes",
                )
                chart = atlas[cid]
                # the interpolant and Φₖ blocks below call `ψ` with a scalar,
                # under ForwardDiff duals among others, so leave this untyped
                ψ = α -> chart.param(SVector(α))

                # The affine map below sends the reference vertex (1,0) to the
                # node `node_indices_on_bdry[2]` and (0,1) to
                # `node_indices_on_bdry[1]`, while `f̂ₖ_comp` sends them to α₁
                # and α₂ respectively.
                α₁ = node_to_param[node_indices_on_bdry[2]][cid]
                # unwrap the other parameter modulo the chart's period so that
                # the two are mutually closest: this is what allows a single
                # periodic chart to be used across the seam of its parameter box
                α₂ = _unwrap(node_to_param[node_indices_on_bdry[1]][cid], α₁, chart.period)
                let w = chart.hc[1] - chart.lc[1]
                    abs(α₂[1] - α₁[1]) < w / 2 || error(
                        "the curved edge of element $elind spans more than half of chart $cid; use a finer mesh or a larger/finer atlas",
                    )
                end
                α₁, α₂ = α₁[1], α₂[1]

                ## Interpolant πₖʲ construction from Inti
                α₁hat = 0.0
                α₂hat = 1.0
                f̂ₖ = (t) -> α₁ .+ (α₂ - α₁) * t
                f̂ₖ_comp = (x) -> f̂ₖ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))

                # l = 1 projection onto linear FE space
                πₖ¹_nodes =
                    reference_nodes(LagrangeElement{ReferenceLine, 2, SVector{2, Float64}})
                πₖ¹ψ_reference_nodes = Vector{SVector{2, Float64}}(undef, length(πₖ¹_nodes))
                for i in eachindex(πₖ¹_nodes)
                    πₖ¹ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ¹_nodes[i][1]))
                end
                πₖ¹ψ_reference_nodes = SVector{2}(πₖ¹ψ_reference_nodes)
                πₖ¹ψ = (x) -> LagrangeElement{ReferenceLine}(πₖ¹ψ_reference_nodes)(x)

                # l = 2 projection onto quadratic FE space
                if order > 1
                    πₖ²_nodes =
                        reference_nodes(LagrangeElement{ReferenceLine, 3, SVector{3, Float64}})
                    πₖ²ψ_reference_nodes =
                        Vector{SVector{2, Float64}}(undef, length(πₖ²_nodes))
                    for i in eachindex(πₖ²_nodes)
                        πₖ²ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ²_nodes[i][1]))
                    end
                    πₖ²ψ_reference_nodes = SVector{3}(πₖ²ψ_reference_nodes)
                    πₖ²ψ = LagrangeElement{ReferenceLine}(πₖ²ψ_reference_nodes)
                end

                # l = 3 projection onto cubic FE space
                if order > 2
                    πₖ³_nodes =
                        reference_nodes(LagrangeElement{ReferenceLine, 4, SVector{4, Float64}})
                    πₖ³ψ_reference_nodes =
                        Vector{SVector{2, Float64}}(undef, length(πₖ³_nodes))
                    for i in eachindex(πₖ³_nodes)
                        πₖ³ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ³_nodes[i][1]))
                    end
                    πₖ³ψ_reference_nodes = SVector{4}(πₖ³ψ_reference_nodes)
                    πₖ³ψ = LagrangeElement{ReferenceLine}(πₖ³ψ_reference_nodes)
                end

                # l = 4 projection onto quartic FE space
                if order > 3
                    πₖ⁴_nodes =
                        reference_nodes(LagrangeElement{ReferenceLine, 5, SVector{5, Float64}})
                    πₖ⁴ψ_reference_nodes =
                        Vector{SVector{2, Float64}}(undef, length(πₖ⁴_nodes))
                    for i in eachindex(πₖ⁴_nodes)
                        πₖ⁴ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁴_nodes[i][1]))
                    end
                    πₖ⁴ψ_reference_nodes = SVector{5}(πₖ⁴ψ_reference_nodes)
                    πₖ⁴ψ = LagrangeElement{ReferenceLine}(πₖ⁴ψ_reference_nodes)
                end

                # l = 5 projection onto quintic FE space
                if order > 4
                    πₖ⁵_nodes =
                        reference_nodes(LagrangeElement{ReferenceLine, 6, SVector{6, Float64}})
                    πₖ⁵ψ_reference_nodes =
                        Vector{SVector{2, Float64}}(undef, length(πₖ⁵_nodes))
                    for i in eachindex(πₖ⁵_nodes)
                        πₖ⁵ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁵_nodes[i][1]))
                    end
                    πₖ⁵ψ_reference_nodes = SVector{6}(πₖ⁵ψ_reference_nodes)
                    πₖ⁵ψ = LagrangeElement{ReferenceLine}(πₖ⁵ψ_reference_nodes)
                end

                # l = 6 projection onto sextic FE space
                if order > 5
                    πₖ⁶_nodes =
                        reference_nodes(LagrangeElement{ReferenceLine, 7, SVector{7, Float64}})
                    πₖ⁶ψ_reference_nodes =
                        Vector{SVector{2, Float64}}(undef, length(πₖ⁶_nodes))
                    for i in eachindex(πₖ⁶_nodes)
                        πₖ⁶ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁶_nodes[i][1]))
                    end
                    πₖ⁶ψ_reference_nodes = SVector{7}(πₖ⁶ψ_reference_nodes)
                    πₖ⁶ψ = LagrangeElement{ReferenceLine}(πₖ⁶ψ_reference_nodes)
                end

                # Nonlinear map

                # θ = 1
                if order == 1
                    Φₖ =
                        (x::AbstractVector) ->
                    (x[1] + x[2])^3 * (
                        ψ(f̂ₖ_comp(x)) -
                            πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    )
                end

                # θ = 2
                if order == 2
                    Φₖ =
                        (x::AbstractVector) ->
                    (x[1] + x[2])^4 * (
                        ψ(f̂ₖ_comp(x)) -
                            πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^2 * (
                        πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    )
                end

                # θ = 3
                if order == 3
                    Φₖ =
                        (x::AbstractVector) ->
                    (x[1] + x[2])^5 * (
                        ψ(f̂ₖ_comp(x)) -
                            πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^2 * (
                        πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^3 * (
                        πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    )
                end

                # θ = 4
                if order == 4
                    Φₖ =
                        (x::AbstractVector) ->
                    (x[1] + x[2])^6 * (
                        ψ(f̂ₖ_comp(x)) -
                            πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^2 * (
                        πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^3 * (
                        πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^4 * (
                        πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    )
                end

                # θ = 5
                if order == 5
                    Φₖ =
                        (x::AbstractVector) ->
                    (x[1] + x[2])^7 * (
                        ψ(f̂ₖ_comp(x)) -
                            πₖ⁵ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^2 * (
                        πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^3 * (
                        πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^4 * (
                        πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^5 * (
                        πₖ⁵ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    )
                end

                # θ = 6
                if order == 6
                    Φₖ =
                        (x::AbstractVector) ->
                    (x[1] + x[2])^8 * (
                        ψ(f̂ₖ_comp(x)) -
                            πₖ⁶ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^2 * (
                        πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^3 * (
                        πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^4 * (
                        πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^5 * (
                        πₖ⁵ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    ) +
                        (x[1] + x[2])^6 * (
                        πₖ⁶ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                            πₖ⁵ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    )
                end

                # Zlamal nonlinear map
                #Φₖ_Z = (x) -> x[2]/(1 - x[1]) * (ψ(x[1] * α₁ + (1 - x[1]) * α₂) - x[1] * a₁ - (1 - x[1])*a₂)

                # Affine map
                aₖ = crvmsh.nodes[node_indices_on_bdry[1]]
                bₖ = crvmsh.nodes[b_idx]
                cₖ = crvmsh.nodes[node_indices_on_bdry[2]]
                F̃ₖ =
                    (x::AbstractVector) -> [
                    (cₖ[1] - bₖ[1]) * x[1] + (aₖ[1] - bₖ[1]) * x[2] + bₖ[1],
                    (cₖ[2] - bₖ[2]) * x[1] + (aₖ[2] - bₖ[2]) * x[2] + bₖ[2],
                ]

                # Full transformation
                Fₖ = (x) -> F̃ₖ(x) + Φₖ(x)
                D = ReferenceTriangle
                T = SVector{2, Float64}
                el = ParametricElement{D, T}(x -> Fₖ(x))
                push!(els_curve, el)
                ψₖ = (s) -> Fₖ([1.0 - s[1], s[1]])
                L = ReferenceHyperCube{1}
                bdry_el = ParametricElement{L, T}(s -> ψₖ(s))
                push!(els_curve_bdry, bdry_el)

                # loop over entities
                Ecurve = typeof(first(els_curve))
                Ecurvebdry = typeof(first(els_curve_bdry))
                haskey(ent2etags[ent], Ecurve) || (ent2etags[ent][Ecurve] = Vector{Int64}())
                append!(ent2etags[ent][Ecurve], length(els_curve))
                # need boundary ent here
                bdryent = get(
                    bdryent_by_edge,
                    minmax(node_indices_on_bdry[1], node_indices_on_bdry[2]),
                    nothing,
                )
                isnothing(bdryent) && error(
                    "element $elind has two vertices on the curved boundary but the edge between them is not a boundary edge; refine the mesh near the boundary",
                )
                haskey(ent2etags[bdryent], Ecurvebdry) ||
                    (ent2etags[bdryent][Ecurvebdry] = Vector{Int64}())
                append!(ent2etags[bdryent][Ecurvebdry], length(els_curve_bdry))
            else
                append!(connect_straight, node_indices)
                el = LagrangeElement{ReferenceSimplex{2}, 3, SVector{2, Float64}}(
                    straight_nodes,
                )
                push!(els_straight, el)

                haskey(ent2etags[ent], E) || (ent2etags[ent][E] = Vector{Int64}())
                append!(ent2etags[ent][E], length(els_straight))
            end
        end
    end

    nv = 3 # Number of vertices for connectivity information in the volume
    nv_bdry = 2 # Number of vertices for connectivity information on the boundary
    Ecurve = typeof(first(els_curve))
    Ecurvebdry = typeof(first(els_curve_bdry))

    crvmsh.etype2mat[Ecurve] = reshape(connect_curve, nv, :)
    crvmsh.etype2els[Ecurve] = convert(Vector{Ecurve}, els_curve)
    crvmsh.etype2orientation[Ecurve] = ones(length(els_curve))

    crvmsh.etype2mat[E_straight] = reshape(connect_straight, nv, :)
    crvmsh.etype2els[E_straight] = convert(Vector{E_straight}, els_straight)
    crvmsh.etype2orientation[E_straight] = ones(length(els_straight))

    crvmsh.etype2mat[Ecurvebdry] = reshape(connect_curve_bdry, nv_bdry, :)
    crvmsh.etype2els[Ecurvebdry] = convert(Vector{Ecurvebdry}, els_curve_bdry)
    crvmsh.etype2orientation[Ecurvebdry] = ones(length(els_curve_bdry))

    _check_element_jacobians(crvmsh.etype2els[Ecurve], check_jacobian)

    return crvmsh
end

function curve_mesh(
        msh::Mesh{3, Float64},
        atlas_by_ent::Dict{EntityKey, <:Atlas},
        order::Int;
        patch_sample_num = nothing,
        face_element_on_curved_surface = nothing,
        chart_atol = nothing,
        face_map = :auto,
        projection_iterations = 4,
        projection_method = :gauss_newton,
        check_jacobian = :warn,
    )
    order > 0 || error("smoothness order must be positive")
    order <= 6 || notimplemented()
    length(atlas_by_ent) == 1 ||
        error("Only simply connected 3D curved domains supported presently")
    atlas = first(values(atlas_by_ent))
    isempty(atlas) && error("the atlas must contain at least one chart")
    projection_method ∈ (:gauss_newton, :newton) ||
        error("`projection_method` must be one of :gauss_newton, :newton")
    second_order = projection_method === :newton
    if face_map === :auto
        # a single chart has no transition maps and so is always consistent;
        # with several charts, assume nothing about them and stay on the safe
        # (chart-independent) side
        face_map = length(atlas) == 1 ? :chart : :projection
    end
    face_map ∈ (:chart, :projection) ||
        error("`face_map` must be one of :auto, :chart, :projection")

    E_straight = LagrangeElement{ReferenceSimplex{3}, 4, SVector{3, Float64}} # TODO fix this to auto be a Pk element type
    E_straight_bdry = LagrangeElement{ReferenceSimplex{2}, 3, SVector{3, Float64}}
    if isnothing(face_element_on_curved_surface)
        face_element_on_curved_surface = (arg) -> true
    end

    if isnothing(patch_sample_num)
        patch_sample_num = -1
        for ent in entities(msh)
            ent.dim == 2 || continue
            patch_sample_num = max(
                patch_sample_num,
                8 * round(Int, sqrt(length(msh.ent2etags[ent][E_straight_bdry]))),
            )
        end
    end

    n2e = node2etags(msh)

    nbdry_els =
        size(msh.etype2mat[LagrangeElement{ReferenceSimplex{2}, 3, SVector{3, Float64}}])[2]

    uniqueidx(v) = unique(i -> v[i], eachindex(v))

    crvmsh = Mesh{3, Float64}()
    (; nodes, etype2mat, etype2els, ent2etags) = crvmsh
    foreach(k -> ent2etags[k] = OrderedDict{DataType, Vector{Int}}(), entities(msh))
    append!(nodes, msh.nodes)

    # Collect the nodes of the face elements lying on the curved surface
    bdry_node_idx = Vector{Int64}()
    max_bdry_edge = 0.0
    for elind in 1:nbdry_els
        node_indices =
            msh.etype2mat[LagrangeElement{ReferenceSimplex{2}, 3, SVector{3, Float64}}][
            :,
            elind,
        ]
        straight_nodes = crvmsh.nodes[node_indices]
        face_element_on_curved_surface(straight_nodes) || continue
        append!(bdry_node_idx, node_indices)
        for i in 1:3, j in (i + 1):3
            max_bdry_edge = max(max_bdry_edge, norm(straight_nodes[i] - straight_nodes[j]))
        end
    end
    bdry_node_idx = unique(bdry_node_idx)

    # A node is deemed to lie on a chart only if the chart reproduces it to
    # within `chart_atol`; the default is loose enough to tolerate a mesh whose
    # boundary nodes only approximately lie on the parametrized surface, but
    # tight enough to reject charts which simply do not cover the node.
    isnothing(chart_atol) && (chart_atol = max_bdry_edge / 4)

    # Locate every boundary node on every chart of the atlas
    locators = [_chart_locator(c, SVector{3, Float64}, patch_sample_num) for c in atlas]
    node_to_param = _atlas_node_params(
        atlas,
        locators,
        crvmsh.nodes,
        bdry_node_idx,
        Val(2);
        atol = chart_atol,
    )
    orphans = setdiff(bdry_node_idx, keys(node_to_param))
    isempty(orphans) || error(
        "$(length(orphans)) boundary node(s) of the curved surface are not covered by any chart of the atlas (e.g. node $(first(orphans)) at $(crvmsh.nodes[first(orphans)])); enlarge the charts' parameter boxes or add charts",
    )
    bdry_node_idx = collect(keys(node_to_param))

    # Move each boundary node onto the surface, using its best chart
    for i in bdry_node_idx
        cid = _select_chart(atlas, node_to_param, (i,))
        crvmsh.nodes[i] = SVector{3, Float64}(atlas[cid].param(node_to_param[i][cid]))
    end

    connect_straight = Int[]
    connect_curve = Int[]
    connect_curve_bdry = Int[]
    # TODO Could use an ElementIterator for straight elements
    els_straight = []
    els_curve = []
    els_curve_bdry = []
    # (element index, excursion / face diameter) for faces flagged by the
    # excursion check below; reported once, after the loop
    far_faces = Tuple{Int, Float64}[]
    face_check_points = _simplex_check_points(ReferenceTriangle(), 4)

    for E in element_types(msh)
        # The purpose of this check is to see if other element types are present in
        # the mesh, such as e.g. cubes; This code errors when encountering a cube,
        # but the method can be extended to transfer straight cubes to the new mesh,
        # similar to how straight simplices are transferred below.
        E <: Union{
            LagrangeElement{ReferenceSimplex{3}},
            LagrangeElement{ReferenceSimplex{2}},
            LagrangeElement{ReferenceHyperCube{1}},
            SVector,
        } || error()
        E <: SVector && continue
        E <: LagrangeElement{ReferenceHyperCube{1}} && continue
        E <: LagrangeElement{ReferenceHyperCube{2}} && continue
        E <: LagrangeElement{ReferenceSimplex{2}} && continue
        E <: LagrangeElement{ReferenceSimplex{3}, 4, SVector{3, Float64}} ||
            (println(E); error())
        els = elements(msh, E)
        for elind in eachindex(els)
            node_indices = msh.etype2mat[E][:, elind]
            straight_nodes = crvmsh.nodes[node_indices]

            verts_on_bdry = findall(x -> x ∈ bdry_node_idx, node_indices)
            # j in C. Bernardi SINUM Sec. 6
            j = length(verts_on_bdry)
            if j > 1
                append!(connect_curve, node_indices)
                node_indices_on_bdry = node_indices[verts_on_bdry]
                # The curved face of this element is spanned by three surface
                # nodes: the `j` nodes of the element which lie on the surface,
                # completed (when j == 2) by a node of an adjacent surface face.
                j <= 3 || error(
                    "element $elind has all four vertices on the curved surface; refine the mesh",
                )
                if j == 3
                    face_nodes = node_indices_on_bdry
                else
                    # Find missing node α₃ that (non-uniquely) defines the curved face simplex containing α₁, α₂
                    candidate_els = elements_containing_nodes(n2e, node_indices_on_bdry)
                    # Filter out volume elements; should be at most two face simplices remaining
                    candidate_els = candidate_els[length.(candidate_els) .== 3]
                    n₃ = 0
                    for cand in candidate_els
                        k = findfirst(t -> t ∉ node_indices_on_bdry, cand)
                        isnothing(k) && continue
                        n₃ = cand[k]
                        # any of the (at most two) adjacent faces defines the
                        # same surface, so keep the first one which is covered
                        # by a chart shared with the two other nodes
                        isnothing(
                            _select_chart(
                                atlas,
                                node_to_param,
                                (node_indices_on_bdry..., n₃),
                            ),
                        ) || break
                    end
                    n₃ == 0 && error("could not complete the curved face of element $elind")
                    face_nodes = vcat(node_indices_on_bdry, n₃)
                end

                # Pick the chart of the atlas on which the whole curved face
                # sits most comfortably; every element is curved through a
                # single chart, so the atlas must cover each face entirely.
                cid = _select_chart(atlas, node_to_param, face_nodes)
                isnothing(cid) && error(
                    "no single chart of the atlas contains the surface nodes $(face_nodes) of element $elind; the charts must overlap by at least one element",
                )
                chart = atlas[cid]
                ψ = chart.param
                α₁ = node_to_param[face_nodes[1]][cid]
                # unwrap the two remaining parameters modulo the chart's period
                # so that the three of them are mutually closest: this is what
                # allows a single periodic chart (e.g. a torus) to be used
                # across the seam of its parameter box
                α₂ = _unwrap(node_to_param[face_nodes[2]][cid], α₁, chart.period)
                α₃ = _unwrap(node_to_param[face_nodes[3]][cid], α₁, chart.period)
                let w = chart.hc - chart.lc,
                        d = max.(abs.(α₂ - α₁), abs.(α₃ - α₁), abs.(α₃ - α₂))
                    all(d .< w ./ 2) || error(
                        "the curved face of element $elind spans more than half of chart $cid; use a finer mesh or a larger/finer atlas",
                    )
                end
                a₁ = SVector{3, Float64}(ψ(α₁))
                a₂ = SVector{3, Float64}(ψ(α₂))
                a₃ = SVector{3, Float64}(ψ(α₃))

                # Construction of the affine map with vertices (aₖ, bₖ, cₖ, dₖ).
                # Vertices aₖ and bₖ always lay on surface. Vertex dₖ always lays in volume.
                aₖ = a₁
                bₖ = a₂
                cₖ = a₃
                # The element's nodes which are not vertices of the curved face:
                # one of them (dₖ) always lies in the volume, and when j == 2 the
                # remaining one takes the place of a₃ as the third vertex of the
                # affine tetrahedron.
                off_face = setdiff(node_indices, node_indices_on_bdry)
                length(off_face) == 4 - j || error(
                    "element $elind has repeated nodes on the curved surface",
                )
                dₖ = crvmsh.nodes[off_face[1]]
                j == 2 && (cₖ = crvmsh.nodes[off_face[2]])

                # The following ensures an ordering of the face nodes so that
                # the resulting normal vector is properly oriented.
                if det([aₖ - dₖ bₖ - dₖ cₖ - dₖ]) < 0
                    tmp = deepcopy(α₁)
                    α₁ = deepcopy(α₂)
                    α₂ = tmp
                    a₁ = SVector{3, Float64}(ψ(α₁))
                    a₂ = SVector{3, Float64}(ψ(α₂))
                    a₃ = SVector{3, Float64}(ψ(α₃))
                    aₖ = a₁
                    bₖ = a₂
                end

                α₁hat = SVector{2, Float64}(1.0, 0.0)
                α₂hat = SVector{2, Float64}(0.0, 1.0)
                α₃hat = SVector{2, Float64}(0.0, 0.0)

                πₖ¹_nodes =
                    reference_nodes(LagrangeElement{ReferenceTriangle, 3, SVector{2, Float64}})
                α_reference_nodes = Vector{SVector{2, Float64}}(undef, length(πₖ¹_nodes))
                α_reference_nodes[1] = SVector{2}(α₃)
                α_reference_nodes[2] = SVector{2}(α₁)
                α_reference_nodes[3] = SVector{2}(α₂)
                α_reference_nodes = SVector{3}(α_reference_nodes)
                f̂ₖ = LagrangeElement{ReferenceSimplex{2}}(α_reference_nodes)
                # Flat face spanned by the same three surface vertices, in the
                # same reference-node order as f̂ₖ.
                Âₖ = LagrangeElement{ReferenceSimplex{2}}(
                    SVector{3}([SVector{3, Float64}(a₃), SVector{3, Float64}(a₁), SVector{3, Float64}(a₂)]),
                )
                # Map from the reference triangle onto the curved surface. With
                # `:chart` this is the parametrization composed with an affine map
                # in the chart's parameter plane, which is only consistent between
                # neighbouring elements when the atlas' transition maps are
                # projective. With `:projection` it is the closest point of the
                # surface to the flat face, which is chart-independent and hence
                # always consistent, at the cost of a small solve per evaluation.
                # How far the projection may wander in the parameter plane before
                # it is deemed to have left the branch its starting guess picked
                # out; see `_closest_point_on_chart`. Two element widths is
                # generous -- the correct root is `O(h²)` away -- while still
                # ruling out a foot point on another sheet of the surface.
                param_diam = max(norm(α₂ - α₁), norm(α₃ - α₁), norm(α₃ - α₂))
                χ = if face_map === :projection
                    (û) -> _closest_point_on_chart(
                        chart,
                        SVector{3}(Âₖ(û)),
                        SVector{2}(f̂ₖ(û)),
                        projection_iterations,
                        2 * param_diam,
                        second_order,
                    )
                else
                    (û) -> SVector{3}(ψ(f̂ₖ(û)))
                end

                # The curved face should sit over the flat face spanned by the
                # same three surface vertices, departing from it only by the
                # sagitta. A much larger excursion means it is somewhere else
                # entirely and will not meet its neighbours; see
                # `_report_far_faces` for the two ways that happens.
                if check_jacobian !== :none
                    face_diam = max(norm(a₂ - a₁), norm(a₃ - a₁), norm(a₃ - a₂))
                    for û in face_check_points
                        e = norm(SVector{3}(χ(û)) - SVector{3}(Âₖ(û)))
                        if e > _FACE_EXCURSION_TOL * face_diam
                            push!(far_faces, (elind, e / face_diam))
                            break
                        end
                    end
                end

                # Tests that should be satisfied; commented for performance
                #@assert (f̂ₖ(α₁hat) ≈ α₁) && (f̂ₖ(α₂hat) ≈ α₂) && (f̂ₖ(α₃hat) ≈ α₃)
                #@assert a₁ ≈ ψ(f̂ₖ(α₁hat))
                #@assert a₂ ≈ ψ(f̂ₖ(α₂hat))
                #@assert a₃ ≈ ψ(f̂ₖ(α₃hat))
                #@assert a₁ ≈ straight_nodes[1] ||
                #        a₁ ≈ straight_nodes[2] ||
                #        a₁ ≈ straight_nodes[3] ||
                #        a₁ ≈ straight_nodes[4]
                #@assert a₂ ≈ straight_nodes[1] ||
                #        a₂ ≈ straight_nodes[2] ||
                #        a₂ ≈ straight_nodes[3] ||
                #        a₂ ≈ straight_nodes[4]
                #if j == 3
                #    @assert a₃ ≈ straight_nodes[1] ||
                #            a₃ ≈ straight_nodes[2] ||
                #            a₃ ≈ straight_nodes[3] ||
                #            a₃ ≈ straight_nodes[4]
                #end
                #@assert aₖ ≈ a₁
                #@assert bₖ ≈ a₂
                F̃ₖ =
                    (x::AbstractVector) -> [
                    (aₖ[1] - dₖ[1]) * x[1] +
                        (bₖ[1] - dₖ[1]) * x[2] +
                        (cₖ[1] - dₖ[1]) * x[3] +
                        dₖ[1],
                    (aₖ[2] - dₖ[2]) * x[1] +
                        (bₖ[2] - dₖ[2]) * x[2] +
                        (cₖ[2] - dₖ[2]) * x[3] +
                        dₖ[2],
                    (aₖ[3] - dₖ[3]) * x[1] +
                        (bₖ[3] - dₖ[3]) * x[2] +
                        (cₖ[3] - dₖ[3]) * x[3] +
                        dₖ[3],
                ]

                # l = 1
                πₖ¹_nodes = reference_nodes(
                    LagrangeElement{
                        ReferenceTriangle,
                        binomial(2 + 1, 2),
                        SVector{2, Float64},
                    },
                )
                πₖ¹ψ_reference_nodes = Vector{SVector{3, Float64}}(undef, length(πₖ¹_nodes))
                for i in eachindex(πₖ¹_nodes)
                    πₖ¹ψ_reference_nodes[i] = χ(πₖ¹_nodes[i])
                end
                πₖ¹ψ_reference_nodes = SVector{binomial(2 + 1, 2)}(πₖ¹ψ_reference_nodes)
                πₖ¹ψ = LagrangeElement{ReferenceSimplex{2}}(πₖ¹ψ_reference_nodes)
                #l = 2
                if order > 1
                    πₖ²_nodes = reference_nodes(
                        LagrangeElement{
                            ReferenceTriangle,
                            binomial(2 + 2, 2),
                            SVector{2, Float64},
                        },
                    )
                    πₖ²ψ_reference_nodes =
                        Vector{SVector{3, Float64}}(undef, length(πₖ²_nodes))
                    for i in eachindex(πₖ²_nodes)
                        πₖ²ψ_reference_nodes[i] = χ(πₖ²_nodes[i])
                    end
                    πₖ²ψ_reference_nodes = SVector{binomial(2 + 2, 2)}(πₖ²ψ_reference_nodes)
                    πₖ²ψ = LagrangeElement{ReferenceSimplex{2}}(πₖ²ψ_reference_nodes)
                end
                #l = 3
                if order > 2
                    πₖ³_nodes = reference_nodes(
                        LagrangeElement{
                            ReferenceTriangle,
                            binomial(2 + 3, 2),
                            SVector{2, Float64},
                        },
                    )
                    πₖ³ψ_reference_nodes =
                        Vector{SVector{3, Float64}}(undef, length(πₖ³_nodes))
                    for i in eachindex(πₖ³_nodes)
                        πₖ³ψ_reference_nodes[i] = χ(πₖ³_nodes[i])
                    end
                    πₖ³ψ_reference_nodes = SVector{binomial(2 + 3, 2)}(πₖ³ψ_reference_nodes)
                    πₖ³ψ = LagrangeElement{ReferenceSimplex{2}}(πₖ³ψ_reference_nodes)
                end
                #l = 4
                if order > 3
                    πₖ⁴_nodes = reference_nodes(
                        LagrangeElement{
                            ReferenceTriangle,
                            binomial(2 + 4, 2),
                            SVector{2, Float64},
                        },
                    )
                    πₖ⁴ψ_reference_nodes =
                        Vector{SVector{3, Float64}}(undef, length(πₖ⁴_nodes))
                    for i in eachindex(πₖ⁴_nodes)
                        πₖ⁴ψ_reference_nodes[i] = χ(πₖ⁴_nodes[i])
                    end
                    πₖ⁴ψ_reference_nodes = SVector{binomial(2 + 4, 2)}(πₖ⁴ψ_reference_nodes)
                    πₖ⁴ψ = LagrangeElement{ReferenceSimplex{2}}(πₖ⁴ψ_reference_nodes)
                end
                #l = 5
                if order > 4
                    πₖ⁵_nodes = reference_nodes(
                        LagrangeElement{
                            ReferenceTriangle,
                            binomial(2 + 5, 2),
                            SVector{2, Float64},
                        },
                    )
                    πₖ⁵ψ_reference_nodes =
                        Vector{SVector{3, Float64}}(undef, length(πₖ⁵_nodes))
                    for i in eachindex(πₖ⁵_nodes)
                        πₖ⁵ψ_reference_nodes[i] = χ(πₖ⁵_nodes[i])
                    end
                    πₖ⁵ψ_reference_nodes = SVector{binomial(2 + 5, 2)}(πₖ⁵ψ_reference_nodes)
                    πₖ⁵ψ = LagrangeElement{ReferenceSimplex{2}}(πₖ⁵ψ_reference_nodes)
                end
                #l = 6
                if order > 5
                    πₖ⁶_nodes = reference_nodes(
                        LagrangeElement{
                            ReferenceTriangle,
                            binomial(2 + 6, 2),
                            SVector{2, Float64},
                        },
                    )
                    πₖ⁶ψ_reference_nodes =
                        Vector{SVector{3, Float64}}(undef, length(πₖ⁶_nodes))
                    for i in eachindex(πₖ⁶_nodes)
                        πₖ⁶ψ_reference_nodes[i] = χ(πₖ⁶_nodes[i])
                    end
                    πₖ⁶ψ_reference_nodes = SVector{binomial(2 + 6, 2)}(πₖ⁶ψ_reference_nodes)
                    πₖ⁶ψ = LagrangeElement{ReferenceSimplex{2}}(πₖ⁶ψ_reference_nodes)
                end

                # Nonlinear map
                if j == 3
                    f̂ₖ_comp =
                        (x::AbstractVector) -> f̂ₖ(
                        (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                            (x[1] + x[2] + x[3]),
                    )
                    χ_comp =
                        (x::AbstractVector) -> χ(
                        (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                            (x[1] + x[2] + x[3]),
                    )
                    if order == 1
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2] + x[3])^3 * (
                                χ_comp(x) - πₖ¹ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            )
                        )
                    end
                    if order == 2
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2] + x[3])^4 * (
                                χ_comp(x) - πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^2 * (
                                πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ¹ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            )
                        )
                    end
                    if order == 3
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2] + x[3])^5 * (
                                χ_comp(x) - πₖ³ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^2 * (
                                πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ¹ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^3 * (
                                πₖ³ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            )
                        )
                    end
                    if order == 4
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2] + x[3])^6 * (
                                χ_comp(x) - πₖ⁴ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^2 * (
                                πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ¹ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^3 * (
                                πₖ³ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^4 * (
                                πₖ⁴ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ³ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            )
                        )
                    end
                    if order == 5
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2] + x[3])^7 * (
                                χ_comp(x) - πₖ⁵ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^2 * (
                                πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ¹ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^3 * (
                                πₖ³ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^4 * (
                                πₖ⁴ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ³ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^5 * (
                                πₖ⁵ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ⁴ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            )
                        )
                    end
                    if order == 6
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2] + x[3])^8 * (
                                χ_comp(x) - πₖ⁶ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^2 * (
                                πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ¹ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^3 * (
                                πₖ³ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ²ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^4 * (
                                πₖ⁴ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ³ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^5 * (
                                πₖ⁵ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ⁴ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            ) +
                                (x[1] + x[2] + x[3])^6 * (
                                πₖ⁶ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                ) - πₖ⁵ψ(
                                    (x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                )
                            )
                        )
                    end
                else
                    f̂ₖ_comp = (x) -> f̂ₖ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    χ_comp = (x) -> χ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                    if order == 1
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2])^3 * (
                                χ_comp(x) -
                                    πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            )
                        )
                    end
                    if order == 2
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2])^4 * (
                                χ_comp(x) -
                                    πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^2 * (
                                πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            )
                        )
                    end
                    if order == 3
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2])^5 * (
                                χ_comp(x) -
                                    πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^2 * (
                                πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^3 * (
                                πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            )
                        )
                    end
                    if order == 4
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2])^6 * (
                                χ_comp(x) -
                                    πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^2 * (
                                πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^3 * (
                                πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^4 * (
                                πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            )
                        )
                    end
                    if order == 5
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2])^7 * (
                                χ_comp(x) -
                                    πₖ⁵ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^2 * (
                                πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^3 * (
                                πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^4 * (
                                πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^5 * (
                                πₖ⁵ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            )
                        )
                    end
                    if order == 6
                        Φₖ =
                            (x::AbstractVector) -> (
                            (x[1] + x[2])^8 * (
                                χ_comp(x) -
                                    πₖ⁶ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^2 * (
                                πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ¹ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^3 * (
                                πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ²ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^4 * (
                                πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ³ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^5 * (
                                πₖ⁵ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ⁴ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            ) +
                                (x[1] + x[2])^6 * (
                                πₖ⁶ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2])) -
                                    πₖ⁵ψ((x[1] * α₁hat + x[2] * α₂hat) / (x[1] + x[2]))
                            )
                        )
                    end
                end

                # Full transformation
                Fₖ = (x::AbstractVector) -> F̃ₖ(x) + Φₖ(x)
                # Tests that Fₖ should satisfy; commented for performance
                #@assert norm(Fₖ([1.0, 0.0, 0.0]) - a₁) < atol
                #@assert norm(Fₖ([0.0, 1.0, 0.0]) - a₂) < atol
                #@assert norm(Fₖ([1.0, 0.0, 0.0]) - aₖ) < atol
                #@assert norm(Fₖ([0.0, 1.0, 0.0]) - bₖ) < atol
                #@assert norm(Fₖ([0.0, 0.0000000000000001, 1.0]) - cₖ) < atol
                #@assert norm(Fₖ([0.0, 0.0000000000000001, 0.0]) - dₖ) < atol
                #if j == 3
                #    @assert norm(a₃ - cₖ) < atol
                #    @assert norm(Fₖ([0.0, 0.0, 1.0]) - cₖ) < atol
                #    @assert norm(Fₖ([0.0, 0.0, 1.0]) - a₃) < atol
                #    @assert norm(Φₖ([0.0, 0.0, 0.3])) < atol
                #    @assert norm(Φₖ([0.0, 0.3, 0.0])) < atol
                #    @assert norm(Φₖ([0.3, 0.0, 0.0])) < atol
                #    @assert norm(
                #        Φₖ([0.3, 0.45, 0.25]) -
                #        (ψ(f̂ₖ_comp([0.3, 0.45, 0.25])) - 0.3*a₁ - 0.45*a₂ - 0.25*a₃),
                #    ) < atol
                #    @assert norm(
                #        Φₖ([0.55, 0.45, 0.0]) -
                #        (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂),
                #    ) < atol
                #end
                #@assert norm(Φₖ([0.0, 0.0000000000000001, 0.3])) < atol
                #@assert norm(Φₖ([0.0, 0.3, 0.0])) < atol
                #@assert norm(Φₖ([0.3, 0.0, 0.0])) < atol
                #if j == 2
                #    @assert norm(Φₖ([0.6, 0.0, 0.4])) < atol
                #    @assert norm(Φₖ([0.0, 0.6, 0.4])) < atol
                #    @assert norm(
                #        Φₖ([0.55, 0.45, 0.0]) -
                #        (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂),
                #    ) < atol
                #end

                D = ReferenceTetrahedron
                T = SVector{3, Float64}
                el = ParametricElement{D, T}(x -> Fₖ(x))
                push!(els_curve, el)
                if j == 3
                    ψₖ = (s) -> Fₖ([s[1], s[2], 1.0 - s[1] - s[2]])
                    F = ReferenceTriangle
                    bdry_el = ParametricElement{F, T}(s -> ψₖ(s))
                    push!(els_curve_bdry, bdry_el)
                    Ecurvebdry = typeof(first(els_curve_bdry))
                    append!(connect_curve_bdry, node_indices_on_bdry)
                end

                Ecurve = typeof(first(els_curve))
                for k in entities(msh)
                    # determine if the straight (LagrangeElement) mesh element
                    # belongs to the entity and, if so, add the curved
                    # (ParametricElement) element.
                    if haskey(msh.ent2etags[k], E)
                        if length(elements_containing_nodes(n2e, node_indices)) > 0
                            haskey(ent2etags[k], Ecurve) ||
                                (ent2etags[k][Ecurve] = Vector{Int64}())
                            append!(ent2etags[k][Ecurve], length(els_curve))
                        end
                    end
                    # find entity that contains straight (LagrangeElement) face
                    # element which is now being replaced by a curved
                    # (ParametricElement) face element
                    if (j == 3) && (haskey(msh.ent2etags[k], E_straight_bdry))
                        k.dim == 2 || continue
                        n_straight_bdry_els = size(msh.etype2mat[E_straight_bdry])[2]
                        candidate_els = elements_containing_nodes(n2e, node_indices_on_bdry)
                        candidate_els = candidate_els[length.(candidate_els) .== 3]
                        if length(candidate_els) > 0
                            haskey(ent2etags[k], Ecurvebdry) ||
                                (ent2etags[k][Ecurvebdry] = Vector{Int64}())
                            append!(ent2etags[k][Ecurvebdry], length(els_curve_bdry))
                        end
                    end
                end
            else
                append!(connect_straight, node_indices)
                el = LagrangeElement{ReferenceSimplex{3}, 4, SVector{3, Float64}}(
                    straight_nodes,
                )
                push!(els_straight, el)

                for k in entities(msh)
                    # determine if the straight mesh element belongs to the entity and, if so, add.
                    if haskey(msh.ent2etags[k], E)
                        #n_straight_vol_els = size(msh.etype2mat[E])[2]
                        if length(elements_containing_nodes(n2e, node_indices)) > 0
                            haskey(ent2etags[k], E) || (ent2etags[k][E] = Vector{Int64}())
                            append!(ent2etags[k][E], length(els_straight))
                        end
                    end
                    # Note: This code does not consider the possibility of boundary
                    # entities that are the boundary of straight simplices.  This is
                    # because of the assumption above that if j > 1 the triangle is
                    # curved.
                end
            end
        end
    end

    nv = 4 # Number of vertices for connectivity information in the volume
    nv_bdry = 3 # Number of vertices for connectivity information on the boundary
    Ecurve = typeof(first(els_curve))
    Ecurvebdry = typeof(first(els_curve_bdry))

    crvmsh.etype2mat[Ecurve] = reshape(connect_curve, nv, :)
    crvmsh.etype2els[Ecurve] = convert(Vector{Ecurve}, els_curve)
    crvmsh.etype2orientation[Ecurve] = ones(length(els_curve))

    crvmsh.etype2mat[E_straight] = reshape(connect_straight, nv, :)
    crvmsh.etype2els[E_straight] = convert(Vector{E_straight}, els_straight)
    crvmsh.etype2orientation[E_straight] = ones(length(els_straight))

    crvmsh.etype2mat[Ecurvebdry] = reshape(connect_curve_bdry, nv_bdry, :)
    crvmsh.etype2els[Ecurvebdry] = convert(Vector{Ecurvebdry}, els_curve_bdry)
    crvmsh.etype2orientation[Ecurvebdry] = ones(length(els_curve_bdry))

    _report_far_faces(far_faces, check_jacobian)
    _check_element_jacobians(crvmsh.etype2els[Ecurve], check_jacobian)

    return crvmsh
end

"""
    sphere_atlas(; radius = 1, center = (0,0,0), overlap = 2.5)

Atlas of six overlapping charts covering a sphere, for use with
[`curve_mesh`](@ref).

Chart `i` is the gnomonic ("cube-sphere") parametrization of the open hemisphere
facing the `i`-th face of the cube, restricted to the parameter box
`[-overlap, overlap]²`; `overlap == 1` makes the six charts tile the sphere
without overlapping, which is *not* enough for `curve_mesh` since no chart would
then contain a boundary element straddling two sextants. The default leaves each
chart reaching about 68° away from its axis, i.e. roughly 23° past its own
sextant, and thus accommodates boundary elements of up to that angular size.

Since the sphere admits no global smooth parametrization, this is the smallest
interesting example of an atlas with more than one chart:

```julia
Ω, msh = ... # a volume mesh of the ball
crvmsh = Inti.curve_mesh(msh, Inti.sphere_atlas(; radius = 1), 3)
```
"""
function sphere_atlas(;
        radius = 1.0,
        center = SVector(0.0, 0.0, 0.0),
        overlap = 2.5,
    )
    c = SVector{3, Float64}(center)
    r = Float64(radius)
    return deformed_sphere_atlas(
        x̂ -> r * x̂ + c;
        inverse = x -> (SVector{3, Float64}(x) - c) / r,
        overlap,
    )
end

"""
    deformed_sphere_atlas(f; inverse = nothing, overlap = 2.5)

Atlas covering the image of the unit sphere under the map `f`, for use with
[`curve_mesh`](@ref). This is the recommended way to chart a surface which is a
smooth deformation of a sphere -- which is most surfaces of interest that admit
no global parametrization.

Chart `i` is `f` composed with the `i`-th gnomonic ("cube-sphere")
parametrization of the unit sphere; see [`sphere_atlas`](@ref) for the meaning of
`overlap`. Charting a surface this way guarantees the atlas is *compatible* in
the sense `curve_mesh` requires: the transition maps are those of the gnomonic
charts of the sphere, which are projective, so elements curved through different
charts still agree on their shared faces and the curved mesh has no cracks.
Assembling an atlas out of unrelated parametrizations carries no such guarantee.

`inverse` should map a point of the surface back to the unit sphere, i.e. invert
`f`. When omitted, the charts are inverted numerically instead, which is slower
and less accurate.
"""
function deformed_sphere_atlas(f; inverse = nothing, overlap = 2.5)
    overlap > 1 || error("`overlap` must be larger than one for the charts to overlap")
    lc = SVector(-Float64(overlap), -Float64(overlap))
    hc = SVector(Float64(overlap), Float64(overlap))
    return [
        ParametricChart(
                α -> f(_unit_sphere_parametrization(α[1], α[2], i)),
                lc,
                hc;
                inverse = isnothing(inverse) ? nothing :
                x -> _unit_sphere_inverse(SVector{3, Float64}(inverse(x)), i),
            ) for i in 1:6
    ]
end

"""
    bean_atlas(; translation, rotation, scaling, overlap = 2.5)

Atlas of six overlapping charts covering the bean surface of [`bean`](@ref), for
use with [`curve_mesh`](@ref). The bean is a deformation of the sphere which
leaves the third coordinate fixed and rescales the other two by functions of it,
so the charts are inverted in closed form.
"""
function bean_atlas(;
        translation = SVector(0.0, 0.0, 0.0),
        rotation = SVector(0.0, 0.0, 0.0),
        scaling = SVector(1.0, 1.0, 1.0),
        overlap = 2.5,
    )
    t = SVector{3, Float64}(translation)
    s = SVector{3, Float64}(scaling)
    R = rotation_matrix(rotation)
    return deformed_sphere_atlas(
        x̂ -> R * (s .* _bean_deformation(x̂)) + t;
        inverse = x -> _bean_deformation_inverse(
            (transpose(R) * (SVector{3, Float64}(x) - t)) ./ s,
        ),
        overlap,
    )
end

"""
    _unit_sphere_inverse(p, id)

Inverse of [`_unit_sphere_parametrization`](@ref) on the hemisphere covered by
chart `id`, or `nothing` when `p` is not on that hemisphere.
"""
function _unit_sphere_inverse(p::SVector{3, Float64}, id)
    if id == 1
        p[1] > 0 || return nothing
        return SVector(p[2] / p[1], p[3] / p[1])
    elseif id == 2
        p[2] > 0 || return nothing
        return SVector(-p[1] / p[2], p[3] / p[2])
    elseif id == 3
        p[3] > 0 || return nothing
        return SVector(p[1] / p[3], p[2] / p[3])
    elseif id == 4
        p[1] < 0 || return nothing
        return SVector(p[2] / p[1], -p[3] / p[1])
    elseif id == 5
        p[2] < 0 || return nothing
        return SVector(-p[1] / p[2], -p[3] / p[2])
    elseif id == 6
        p[3] < 0 || return nothing
        return SVector(p[1] / p[3], -p[2] / p[3])
    end
    return error("unknown chart id $id")
end
