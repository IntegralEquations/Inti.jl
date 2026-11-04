# Benchmarks for the two ways `Inti.curve_mesh` can lay a curved face over a
# parametrized surface:
#
#   face_map = :chart       parametrization ∘ affine map in the parameter plane
#   face_map = :projection  closest point of the surface to the flat face
#
# `:chart` is only consistent between elements curved through *different* charts
# when the atlas' transition maps are projective; `:projection` is
# chart-independent and therefore correct for any atlas. Sections 1-5 run on
# surfaces whose atlas is compatible (so both are correct and can be compared on
# equal footing); section 6 runs on an atlas of arbitrary overlapping patches,
# where only `:projection` is correct.
#
# Run all sections:
#     julia --project=test -t auto bench/curved_face_map_bench.jl
# Run a subset:
#     julia --project=test -t auto bench/curved_face_map_bench.jl 1 4 6
#
# ---------------------------------------------------------------------------
# NOTATION -- three different "orders" appear here and are easy to confuse:
#
#   n_gn    Gauss--Newton steps in the closest-point solve; this is
#           `curve_mesh`'s `projection_iterations`. Applies to :projection only.
#   n_gl    Gauss--Legendre points *per direction* in the per-element rule used
#           by section 3 (n_gl = 4 means a 4x4 = 16 point rule). Local to this
#           file; nothing in Inti uses it.
#   qorder  Inti's quadrature order, applied over a whole mesh.
#
#   θ       Bernardi smoothness order passed to `curve_mesh` (4 throughout).
#   h       mesh size.
#
# Wherever one of n_gn / n_gl / qorder is swept, the others are pinned.
# All errors reported are relative. Reference values are computed independently
# of `curve_mesh`.
# ---------------------------------------------------------------------------

using Inti
using Gmsh
using StaticArrays
using LinearAlgebra
using Printf
using QuadGK
using ForwardDiff

const θ = 4          # Bernardi smoothness order, fixed everywhere
const NGN_CONVERGED = 12   # n_gn large enough to be fully converged
const NREPEAT = 3    # timing repetitions; the minimum is reported

# =========================================================== mesh builders ===

"Tetrahedral mesh of the unit ball, optionally dilated to a spheroid (1,1,c)."
function ball_mesh(; meshsize, c = 1.0)
    Inti.clear_entities!()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    tag = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    c != 1 && gmsh.model.occ.dilate([(3, tag)], 0, 0, 0, 1.0, 1.0, c)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(1)
    msh = Inti.import_mesh(; dim = 3)
    Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, Inti.entities(msh))
    gmsh.finalize()
    return Ω, msh
end

"""
Apply `f` to every node of `msh` in place. `ElementIterator` reads the nodes
lazily, so the straight elements follow along; if `f` maps the ball onto a solid,
this turns a ball mesh into a mesh of that solid whose boundary nodes lie exactly
on its surface.
"""
function deform!(msh, f)
    for i in eachindex(msh.nodes)
        msh.nodes[i] = f(msh.nodes[i])
    end
    return msh
end

bean_mesh(; meshsize) = let (Ω, msh) = ball_mesh(; meshsize)
    (Ω, deform!(msh, Inti._bean_deformation))
end

# The "blob" of section 6: a bumpy, non-axisymmetric, star-shaped surface r(x̂)x̂.
blob_r(x̂) = 1 + 0.25 * x̂[1] * x̂[2] + 0.15 * x̂[3]^3
blob_f(x̂) = blob_r(x̂) * SVector{3}(x̂)
blob_finv(x) = SVector{3, Float64}(x) / norm(x)
function blob_ball(x)
    n = norm(x)
    n < 1.0e-14 && return SVector{3, Float64}(x)
    return blob_r(x / n) * SVector{3, Float64}(x)
end

blob_mesh(; meshsize) = let (Ω, msh) = ball_mesh(; meshsize)
    (Ω, deform!(msh, blob_ball))
end

# ================================================================= atlases ===

spheroid_atlas(c; overlap = 6.0) = Inti.deformed_sphere_atlas(
    x̂ -> SVector(x̂[1], x̂[2], c * x̂[3]);
    inverse = y -> SVector(y[1], y[2], y[3] / c),
    overlap,
)

"""
Two rotated spherical-coordinate patches of the blob. Each is smooth on its box
and they overlap generously, but they are unrelated to one another: their
transition maps are not projective, so `:chart` is *not* valid for this atlas.
"""
function blob_atlas()
    dir(θ_, ϕ) = SVector(sin(θ_) * cos(ϕ), sin(θ_) * sin(ϕ), cos(θ_))
    roty(p) = SVector(p[3], p[2], -p[1])
    lc, hc = SVector(0.25, -π), SVector(π - 0.25, π)
    per = SVector(Inf, 2π)
    return [
        Inti.ParametricChart(α -> blob_f(dir(α[1], α[2])), lc, hc; period = per),
        Inti.ParametricChart(α -> blob_f(roty(dir(α[1], α[2]))), lc, hc; period = per),
    ]
end

# ============================================================== references ===

"Integral of `g` over the unit sphere, via the six exact cube-sphere patches."
function patch_integral(g; rtol = 1.0e-12)
    return sum(1:6) do id
        p = α -> SVector{3}(Inti._unit_sphere_parametrization(α[1], α[2], id))
        val, _ = quadgk(-1.0, 1.0; rtol) do u
            inner, _ = quadgk(-1.0, 1.0; rtol) do v
                J = ForwardDiff.jacobian(p, SVector(u, v))
                g(p(SVector(u, v))) * norm(cross(J[:, 1], J[:, 2]))
            end
            inner
        end
        val
    end
end

"Surface area of the image of the unit sphere under `f`."
function atlas_area(f; rtol = 1.0e-12)
    return sum(1:6) do id
        g = α -> SVector{3}(f(Inti._unit_sphere_parametrization(α[1], α[2], id)))
        val, _ = quadgk(-1.0, 1.0; rtol) do u
            inner, _ = quadgk(-1.0, 1.0; rtol) do v
                J = ForwardDiff.jacobian(g, SVector(u, v))
                norm(cross(J[:, 1], J[:, 2]))
            end
            inner
        end
        val
    end
end

"Volume of the bean: det(∇G) = A(z)B(z) depends only on z."
function bean_volume()
    A(z) = Inti._BEAN_A * sqrt(1 - Inti._BEAN_ALPHA3 * cospi(z))
    B(z) = Inti._BEAN_B * sqrt(1 - Inti._BEAN_ALPHA2 * cospi(z))
    v, _ = quadgk(z -> A(z) * B(z) * π * (1 - z^2), -1, 1; rtol = 1.0e-14)
    return v
end

const REFS = Dict{Symbol, Float64}()
function refs!()
    isempty(REFS) || return REFS
    REFS[:bean_area] = atlas_area(Inti._bean_deformation)
    REFS[:bean_vol] = bean_volume()
    REFS[:blob_area] = atlas_area(blob_f)
    REFS[:blob_vol] = patch_integral(x̂ -> blob_r(x̂)^3 / 3)
    REFS[:sphere_area] = atlas_area(identity)
    return REFS
end

# ==================================================== gap between elements ===

const REFVERTS = (SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(0.0, 0.0))

"Nudge a point into the interior of the reference simplex."
function clampsimplex(u)
    a, b = max(u[1], 1.0e-13), max(u[2], 1.0e-13)
    s = a + b
    if s > 1 - 1.0e-13
        f = (1 - 1.0e-13) / s
        a, b = a * f, b * f
    end
    return SVector(a, b)
end

"""
Distance from `x` to the curve `c(s)`, `s ∈ [0,1]`: a coarse scan followed by a
golden-section search, so this is a true distance and not the distance to the
nearest sample (which would floor out at the sampling spacing and mask how small
the gaps really are).
"""
function dist_to_curve(x, c; ncoarse = 48, niter = 60)
    sbest, dbest = 0.0, Inf
    for s in range(0, 1, ncoarse)
        d = norm(c(s) - x)
        d < dbest && ((sbest, dbest) = (s, d))
    end
    h = 1 / (ncoarse - 1)
    lo, hi = max(0.0, sbest - h), min(1.0, sbest + h)
    φ = (sqrt(5) - 1) / 2
    s1, s2 = hi - φ * (hi - lo), lo + φ * (hi - lo)
    f1, f2 = norm(c(s1) - x), norm(c(s2) - x)
    for _ in 1:niter
        if f1 < f2
            hi, s2, f2 = s2, s1, f1
            s1 = hi - φ * (hi - lo)
            f1 = norm(c(s1) - x)
        else
            lo, s1, f1 = s1, s2, f2
            s2 = lo + φ * (hi - lo)
            f2 = norm(c(s2) - x)
        end
    end
    return min(dbest, f1, f2)
end

curved_faces(crv, Γ) = let Γₕ = crv[Γ]
    E = only(filter(T -> T <: Inti.ParametricElement, Inti.element_types(Γₕ)))
    (Inti.elements(Γₕ, E), Inti.connectivity(Γₕ, E), Inti.nodes(Γₕ))
end

"""
Largest and mean gap between the two curved images of an edge shared by two
curved boundary faces. Zero iff the curved surface mesh is watertight.
"""
function neighbour_gap(crv, Γ)
    els, mat, nds = curved_faces(crv, Γ)
    nel = length(els)
    vertmap = map(1:nel) do k
        pts = [els[k](r) for r in REFVERTS]
        Dict(n => REFVERTS[argmin(norm(pts[j] - nds[n]) for j in 1:3)] for n in mat[:, k])
    end
    edge2els = Dict{Tuple{Int, Int}, Vector{Int}}()
    for k in 1:nel
        n = sort(mat[:, k])
        for e in ((n[1], n[2]), (n[1], n[3]), (n[2], n[3]))
            push!(get!(edge2els, e, Int[]), k)
        end
    end
    worst, tot, cnt = 0.0, 0.0, 0
    for (e, ks) in edge2els
        length(ks) == 2 || continue
        k1, k2 = ks
        edge(k) = let a = vertmap[k][e[1]], b = vertmap[k][e[2]]
            s -> els[k](clampsimplex(a + s * (b - a)))
        end
        d = maximum(s -> dist_to_curve(edge(k1)(s), edge(k2)), range(0, 1, 9))
        worst = max(worst, d)
        tot += d
        cnt += 1
    end
    return worst, tot / max(cnt, 1)
end

# ============================================ volume-mesh consistency (§7) ===

const TETVERTS = (
    SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0),
    SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 0.0),
)

"Volume elements of the curved mesh as `(element, nodetags, iscurved)`, plus the nodes."
function volume_elements(crv, Ω)
    Ωₕ = crv[Ω]
    out, nds = [], Inti.nodes(Ωₕ)
    for E in Inti.element_types(Ωₕ)
        (
            E <: Inti.ParametricElement{Inti.ReferenceSimplex{3}} ||
                E <: Inti.LagrangeElement{Inti.ReferenceSimplex{3}}
        ) || continue
        els, mat = Inti.elements(Ωₕ, E), Inti.connectivity(Ωₕ, E)
        for k in eachindex(els)
            push!(out, (els[k], mat[:, k], E <: Inti.ParametricElement))
        end
    end
    return out, nds
end

"Map each node tag of a tetrahedron to the reference vertex it sits at."
function tet_vertmap(el, tags, nds)
    # the j == 2 curved map divides by x₁+x₂, which vanishes at two reference
    # vertices, so probe just inside them
    ctr, δ = SVector(0.25, 0.25, 0.25), 1.0e-6
    pts = [SVector{3}(el((1 - δ) * r + δ * ctr)) for r in TETVERTS]
    return Dict(n => TETVERTS[argmin(norm(pts[j] - nds[n]) for j in 1:4)] for n in tags)
end

"""
Distance from `x` to the surface patch `g(a,b)` (`a,b > 0`, `a+b < 1`) by nested
grid refinement. This is a true distance, so a reparametrization of the same
patch registers as zero -- without that, two elements meeting perfectly but
parametrizing their shared face differently would look like a large gap.
"""
function dist_to_face(x, g; levels = 7, n = 8)
    lo1, lo2, span = 0.0, 0.0, 1.0
    best, ba, bb = Inf, 1 / 3, 1 / 3
    for _ in 1:levels
        found = false
        for i in 0:n, j in 0:n
            a, b = lo1 + span * i / n, lo2 + span * j / n
            (a > 1.0e-12 && b > 1.0e-12 && a + b < 1 - 1.0e-12) || continue
            d = norm(g(a, b) - x)
            if d < best
                best, ba, bb, found = d, a, b, true
            end
        end
        found || break
        span = 2 * span / n
        lo1, lo2 = max(ba - span / 2, 0.0), max(bb - span / 2, 0.0)
    end
    return best
end

"""
Largest and mean gap between the two images of an internal face shared by two
volume elements, split by whether the neighbours are curved. Nonzero means the
curved tetrahedra do not tile the domain, which shows up directly as a volume
error no quadrature order can remove.

Values near 1e-9 are the floor of the measurement: face points are nudged into
the element interior by that much to stay inside the reference simplex.
"""
function internal_face_gap(crv, Ω)
    els, nds = volume_elements(crv, Ω)
    vms = [tet_vertmap(e[1], e[2], nds) for e in els]
    face2els = Dict{NTuple{3, Int}, Vector{Int}}()
    for (k, (_, tags, _)) in enumerate(els)
        s = sort(tags)
        for f in (
                (s[1], s[2], s[3]), (s[1], s[2], s[4]),
                (s[1], s[3], s[4]), (s[2], s[3], s[4]),
            )
            push!(get!(face2els, f, Int[]), k)
        end
    end
    nsub = 8
    λs = [
        SVector(i, j, nsub - i - j) ./ nsub for i in 1:(nsub - 2)
            for j in 1:(nsub - 1 - i)
    ]
    stats = Dict(
        k => Float64[] for k in (:curved_curved, :curved_straight, :straight_straight)
    )
    for (f, ks) in face2els
        length(ks) == 2 || continue
        k1, k2 = ks
        c1, c2 = els[k1][3], els[k2][3]
        key = c1 && c2 ? :curved_curved : (c1 || c2) ? :curved_straight : :straight_straight
        off(k) = only(setdiff(collect(TETVERTS), [vms[k][n] for n in f]))
        ε = 1.0e-9
        function img(k, a, b)
            λ = SVector(a, b, 1 - a - b)
            p = sum(λ[i] * vms[k][f[i]] for i in 1:3)
            return SVector{3}(els[k][1]((1 - ε) * p + ε * off(k)))
        end
        d = maximum(λs) do λ
            dist_to_face(img(k1, λ[1], λ[2]), (a, b) -> img(k2, a, b))
        end
        push!(stats[key], d)
    end
    return stats
end

"Minimum Jacobian determinant over the volume elements; negative means a tangled element."
function min_jacobian(crv, Ω)
    els, _ = volume_elements(crv, Ω)
    pts = (
        SVector(0.2, 0.2, 0.2), SVector(0.6, 0.2, 0.1), SVector(0.1, 0.6, 0.1),
        SVector(0.1, 0.1, 0.6), SVector(0.05, 0.05, 0.05), SVector(0.3, 0.3, 0.3),
    )
    m = Inf
    for (el, _, _) in els, p in pts
        m = min(m, det(ForwardDiff.jacobian(u -> SVector{3}(el(u)), p)))
    end
    return m
end

# ============================================= per-element quadrature (n_gl) ==

"""
Area of a single curved element by an `n_gl`-point-per-direction tensor
Gauss--Legendre rule, mapped onto the reference triangle by a Duffy transform.
Used to probe the smoothness of one element map, free of mesh-level
cancellation.
"""
function element_area(el, n_gl)
    x, w = QuadGK.gauss(n_gl, 0, 1)
    s = 0.0
    for i in 1:n_gl, j in 1:n_gl
        u = x[i]
        v = x[j] * (1 - u)
        J = ForwardDiff.jacobian(p -> SVector{3}(el(p)), SVector(u, v))
        s += w[i] * w[j] * (1 - u) * norm(cross(J[:, 1], J[:, 2]))
    end
    return s
end

"Curved boundary element whose centre points closest to the direction `dir`, and its size."
function element_near(crv, Γ, dir)
    els, _, _ = curved_faces(crv, Γ)
    mid = SVector(1 / 3, 1 / 3)
    _, k = findmin(el -> -dot(normalize(SVector{3}(el(mid))), normalize(dir)), els)
    el = els[k]
    corners = (SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0e-9, 1.0e-9))
    h = maximum(norm(SVector{3}(el(a)) - SVector{3}(el(b))) for a in corners, b in corners)
    return el, h
end

# ================================================================== timing ===

"Minimum wall time of `f()` over `NREPEAT` runs, after one warm-up run."
function best_time(f)
    f()
    return minimum(1:NREPEAT) do _
        t = time_ns()
        f()
        return (time_ns() - t) / 1.0e9
    end
end

# ================================================================ sections ===

ntets(msh) = size(
    msh.etype2mat[
        Inti.LagrangeElement{
            Inti.ReferenceSimplex{3}, 4, SVector{3, Float64},
        },
    ], 2,
)

cases(ngns) = [(:chart, 0); [(:projection, n) for n in ngns]]
label(fm, n) = fm === :chart ? "chart" : "projection n_gn=$n"

function section1()
    println(
        """
        ### 1. Cost and accuracy on the bean (θ = $θ, qorder = 5)
        Both face maps are correct here: the bean's atlas is compatible.
        Absolute times are noisy run to run; the ratios are the stable part."""
    )
    r = refs!()
    for meshsize in (0.4, 0.25)
        Ω, msh = bean_mesh(; meshsize)
        Γ = Inti.external_boundary(Ω)
        @printf("\nh = %.2f  (%d tetrahedra)\n", meshsize, ntets(msh))
        @printf(
            "%-20s | %7s %7s %7s %8s | %9s %9s %9s | %9s %9s\n",
            "face_map", "curve", "volQ", "surfQ", "ns/eval",
            "area err", "vol err", "div mis", "gap max", "gap avg"
        )
        println("-"^122)
        for (fm, n_gn) in cases((2, 4, 6, 8))
            kw = (; face_map = fm, projection_iterations = max(n_gn, 1))
            tcurve = best_time(() -> Inti.curve_mesh(msh, Inti.bean_atlas(), θ; kw...))
            crv = Inti.curve_mesh(msh, Inti.bean_atlas(), θ; kw...)
            tvol = best_time(() -> Inti.Quadrature(crv[Ω]; qorder = 5))
            tsurf = best_time(() -> Inti.Quadrature(crv[Γ]; qorder = 5))
            Ωq, Γq = Inti.Quadrature(crv[Ω]; qorder = 5), Inti.Quadrature(crv[Γ]; qorder = 5)
            vol = Inti.integrate(x -> 1, Ωq)
            area = Inti.integrate(x -> 1, Γq)
            divvol = Inti.integrate(q -> dot(q.coords, q.normal) / 3, Γq)
            els, _, _ = curved_faces(crv, Ω)
            pts = (SVector(0.2, 0.2, 0.2), SVector(0.5, 0.2, 0.1), SVector(0.1, 0.6, 0.1))
            nev = length(els) * length(pts)
            tev = best_time(() -> sum(el(p) for el in els, p in pts))
            gap, gapavg = neighbour_gap(crv, Γ)
            @printf(
                "%-20s | %6.3fs %6.3fs %6.3fs %8.0f | %9.2e %9.2e %9.2e | %9.2e %9.2e\n",
                label(fm, n_gn), tcurve, tvol, tsurf, 1.0e9 * tev / nev,
                abs(area - r[:bean_area]) / r[:bean_area],
                abs(vol - r[:bean_vol]) / r[:bean_vol],
                abs(vol - divvol) / r[:bean_vol], gap, gapavg
            )
        end
    end
    return println()
end

function section2()
    println(
        """
        ### 2. Sweeping n_gn past convergence (bean, θ = $θ)
        n_gn controls the gap and nothing else: past convergence the rows are
        bit-identical while the area error does not move. If :projection is not
        accurate enough, raise qorder rather than n_gn."""
    )
    r = refs!()
    for meshsize in (0.25, 0.16)
        Ω, msh = bean_mesh(; meshsize)
        Γ = Inti.external_boundary(Ω)
        @printf("\nh = %.2f\n", meshsize)
        @printf(
            "%-20s | %10s | %11s %11s %11s\n", "face_map", "gap max",
            "area q=5", "area q=8", "area q=12"
        )
        println("-"^72)
        for (fm, n_gn) in cases((4, 8, 12, 16, 24, 32, 48))
            crv = Inti.curve_mesh(
                msh, Inti.bean_atlas(), θ;
                face_map = fm, projection_iterations = max(n_gn, 1),
            )
            gap, _ = neighbour_gap(crv, Γ)
            errs = map((5, 8, 12)) do qo
                a = Inti.integrate(x -> 1, Inti.Quadrature(crv[Γ]; qorder = qo))
                abs(a - r[:bean_area]) / r[:bean_area]
            end
            @printf(
                "%-20s | %10.2e | %11.3e %11.3e %11.3e\n",
                label(fm, n_gn), gap, errs...
            )
        end
    end
    return println()
end

const NGL = 2:2:16

function section3()
    println(
        """
        ### 3. Per-element convergence in n_gl (n_gn = $NGN_CONVERGED, θ = $θ)
        One curved boundary element integrated in isolation, reference n_gl = 40 on
        the same element. Both maps are analytic and both reach machine precision;
        which converges faster depends on the local curvature."""
    )
    @printf("\n%-38s %-11s", "case", "face_map")
    for n in NGL
        @printf(" %8d", n)
    end
    println()
    println("-"^(50 + 9 * length(NGL)))
    probes = []
    Ω, msh = ball_mesh(; meshsize = 0.18)
    push!(probes, ("sphere, any element", Inti.sphere_atlas(), Ω, msh, SVector(1.0, 0.0, 0.0)))
    Ω, msh = ball_mesh(; meshsize = 0.18, c = 0.5)
    push!(probes, ("spheroid c=0.5, equator (reach 0.25)", spheroid_atlas(0.5), Ω, msh, SVector(1.0, 0.0, 0.0)))
    push!(probes, ("spheroid c=0.5, pole (reach 2.0)", spheroid_atlas(0.5), Ω, msh, SVector(0.0, 0.0, 1.0)))
    Ω, msh = bean_mesh(; meshsize = 0.25)
    for (nm, d) in (
            ("bean, +x", SVector(1.0, 0.0, 0.0)),
            ("bean, +z", SVector(0.0, 0.0, 1.0)),
            ("bean, +y", SVector(0.0, 1.0, 0.0)),
        )
        push!(probes, (nm, Inti.bean_atlas(), Ω, msh, d))
    end
    for (nm, atlas, Ω_, msh_, dir) in probes
        Γ = Inti.external_boundary(Ω_)
        for fm in (:chart, :projection)
            crv = Inti.curve_mesh(
                msh_, atlas, θ;
                face_map = fm, projection_iterations = NGN_CONVERGED,
            )
            el, _ = element_near(crv, Γ, dir)
            ref = element_area(el, 40)
            @printf("%-38s %-11s", fm === :chart ? nm : "", fm)
            for n in NGL
                @printf(" %8.1e", abs(element_area(el, n) - ref) / ref)
            end
            println()
        end
    end
    return println()
end

function section4()
    println(
        """
        ### 4. Reach sweep on oblate spheroids (1,1,c) (n_gn = $NGN_CONVERGED, θ = $θ)
        Local reach at the equator is c², so the flat face's sagitta/reach ratio grows
        as c shrinks. Both maps degrade, but :projection is far more sensitive --
        the signature expected if its analyticity is capped by the medial axis."""
    )
    @printf(
        "\n%5s %7s %7s %9s | %9s %9s | %9s %9s\n",
        "c", "reach", "h(el)", "sag/reach", "chart", "chart", "proj", "proj"
    )
    @printf(
        "%5s %7s %7s %9s | %9s %9s | %9s %9s\n",
        "", "(c²)", "", "", "n_gl=4", "n_gl=8", "n_gl=4", "n_gl=8"
    )
    println("-"^76)
    for c in (1.0, 0.8, 0.65, 0.5, 0.4, 0.32)
        Ω, msh = ball_mesh(; meshsize = 0.18, c)
        Γ = Inti.external_boundary(Ω)
        atlas = spheroid_atlas(c)
        out, hel, failed = Dict{Tuple{Symbol, Int}, Float64}(), NaN, false
        for fm in (:chart, :projection)
            try
                crv = Inti.curve_mesh(
                    msh, atlas, θ;
                    face_map = fm, projection_iterations = NGN_CONVERGED,
                )
                el, h = element_near(crv, Γ, SVector(1.0, 0.0, 0.0))
                hel = h
                ref = element_area(el, 40)
                for n in (4, 8)
                    out[(fm, n)] = abs(element_area(el, n) - ref) / ref
                end
            catch e
                failed = true
                @printf(
                    "%5.2f %7.3f %7s %9s | curve_mesh failed (%s): %s\n",
                    c, c^2, "-", "-", fm,
                    first(split(sprint(showerror, e), '\n'))
                )
            end
        end
        failed && continue
        @printf(
            "%5.2f %7.3f %7.3f %9.3f | %9.1e %9.1e | %9.1e %9.1e\n",
            c, c^2, hel, hel^2 / (8 * c^4),
            out[(:chart, 4)], out[(:chart, 8)], out[(:projection, 4)], out[(:projection, 8)]
        )
    end
    return println()
end

function section5()
    println(
        """
        ### 5. Mesh refinement (bean area error, n_gn = $NGN_CONVERGED, θ = $θ)
        Each row is an independently generated gmsh mesh, NOT a nested refinement
        sequence, so no convergence order should be read off these rows: the error is
        dominated by mesh-to-mesh variation and is not even monotone. What the qorder
        columns do show is that the error is quadrature-limited, and that :chart
        reaches the reference/roundoff floor while :projection does not."""
    )
    r = refs!()
    @printf(
        "\n%7s %9s | %11s %11s | %11s %11s\n",
        "target h", "mean h", "chart q=5", "chart q=8", "proj q=5", "proj q=8"
    )
    println("-"^68)
    for meshsize in (0.5, 0.4, 0.32, 0.25, 0.2, 0.16)
        Ω, msh = bean_mesh(; meshsize)
        Γ = Inti.external_boundary(Ω)
        E = Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 3, SVector{3, Float64}}
        mat = msh.etype2mat[E]
        tot, cnt = 0.0, 0
        for k in axes(mat, 2)
            p = [msh.nodes[mat[i, k]] for i in 1:3]
            for (i, j) in ((1, 2), (1, 3), (2, 3))
                tot += norm(p[i] - p[j])
                cnt += 1
            end
        end
        errs = Float64[]
        for fm in (:chart, :projection)
            crv = Inti.curve_mesh(
                msh, Inti.bean_atlas(), θ;
                face_map = fm, projection_iterations = NGN_CONVERGED,
            )
            for qo in (5, 8)
                a = Inti.integrate(x -> 1, Inti.Quadrature(crv[Γ]; qorder = qo))
                push!(errs, abs(a - r[:bean_area]) / r[:bean_area])
            end
        end
        @printf(
            "%7.2f %9.4f | %11.3e %11.3e | %11.3e %11.3e\n",
            meshsize, tot / cnt, errs...
        )
    end
    return println()
end

function section6()
    println(
        """
        ### 6. An atlas of arbitrary overlapping patches (θ = $θ, qorder = 5)
        A bumpy star-shaped surface charted by two unrelated rotated
        spherical-coordinate patches. Their transition maps are not projective, so
        :chart cracks the mesh -- note that every element still lies exactly on the
        surface (off-surface column); what :chart gets wrong is only that neighbours
        traverse the shared edge along different curves. This is the case
        `face_map = :auto` selects :projection for."""
    )
    r = refs!()
    for meshsize in (0.3, 0.2)
        Ω, msh = blob_mesh(; meshsize)
        Γ = Inti.external_boundary(Ω)
        @printf("\nh = %.2f  (%d tetrahedra)\n", meshsize, ntets(msh))
        @printf(
            "%-20s | %9s %9s %9s | %9s %9s\n", "face_map",
            "area err", "vol err", "div mis", "off-surf", "gap max"
        )
        println("-"^76)
        for (fm, n_gn) in cases((2, 4, 6))
            crv = Inti.curve_mesh(
                msh, blob_atlas(), θ;
                face_map = fm, projection_iterations = max(n_gn, 1),
            )
            Ωq, Γq = Inti.Quadrature(crv[Ω]; qorder = 5), Inti.Quadrature(crv[Γ]; qorder = 5)
            vol = Inti.integrate(x -> 1, Ωq)
            area = Inti.integrate(x -> 1, Γq)
            divvol = Inti.integrate(q -> dot(q.coords, q.normal) / 3, Γq)
            off = maximum(q -> abs(norm(q.coords) - blob_r(blob_finv(q.coords))), Γq)
            gap, _ = neighbour_gap(crv, Γ)
            @printf(
                "%-20s | %9.2e %9.2e %9.2e | %9.2e %9.2e\n",
                label(fm, n_gn),
                abs(area - r[:blob_area]) / r[:blob_area],
                abs(vol - r[:blob_vol]) / r[:blob_vol],
                abs(vol - divvol) / r[:blob_vol], off, gap
            )
        end
    end
    return println()
end

function section7()
    println(
        """
        ### 7. The volume analogue (bean, θ = $θ)
        The area error is quadrature-limited, so it falls with qorder. The volume
        error is not: it plateaus, because the curved tetrahedra do not tile the
        domain exactly. Two independent defects are responsible, and only one of
        them is about the face map.

        (a) volume error against qorder -- note the plateau"""
    )
    r = refs!()
    qs = (1, 2, 3, 5, 7, 9, 11, 13, 15)   # orders available for tetrahedra
    for meshsize in (0.4, 0.25)
        Ω, msh = bean_mesh(; meshsize)
        @printf("\nh = %.2f\n%-14s |", meshsize, "face_map")
        for q in qs
            @printf(" %10s", "q=$q")
        end
        println()
        println("-"^(16 + 11 * length(qs)))
        for (fm, n_gn) in ((:chart, 0), (:projection, NGN_CONVERGED))
            crv = Inti.curve_mesh(
                msh, Inti.bean_atlas(), θ;
                face_map = fm, projection_iterations = max(n_gn, 1),
            )
            @printf("%-14s |", fm)
            for q in qs
                v = Inti.integrate(x -> 1, Inti.Quadrature(crv[Ω]; qorder = q))
                @printf(" %10.2e", abs(v - r[:bean_vol]) / r[:bean_vol])
            end
            println()
        end
    end

    println(
        """

        (b) where the volume defect comes from. `gap` is the true distance between
            the two images of an internal face shared by two volume elements; ~1e-9
            is the floor of the measurement. `min detJ` < 0 means a tangled element."""
    )
    @printf(
        "\n%5s %-11s | %9s %10s | %9s %9s %9s\n", "h", "face_map",
        "vol err", "min detJ", "gap crv-crv", "crv-str", "str-str"
    )
    println("-"^76)
    for meshsize in (0.4, 0.25)
        Ω, msh = bean_mesh(; meshsize)
        for (fm, n_gn) in ((:chart, 0), (:projection, NGN_CONVERGED))
            crv = Inti.curve_mesh(
                msh, Inti.bean_atlas(), θ;
                face_map = fm, projection_iterations = max(n_gn, 1),
            )
            v = Inti.integrate(x -> 1, Inti.Quadrature(crv[Ω]; qorder = 11))
            st = internal_face_gap(crv, Ω)
            g(k) = isempty(st[k]) ? 0.0 : maximum(st[k])
            @printf(
                "%5.2f %-11s | %9.2e %+10.2e | %9.2e %9.2e %9.2e\n",
                meshsize, fm, abs(v - r[:bean_vol]) / r[:bean_vol], min_jacobian(crv, Ω),
                g(:curved_curved), g(:curved_straight), g(:straight_straight)
            )
        end
    end
    return println()
end

const SECTIONS = (section1, section2, section3, section4, section5, section6, section7)

function main(which = eachindex(SECTIONS))
    println(
        """
        Curved face maps: :chart vs :projection
        =======================================
        n_gn   Gauss--Newton steps  (= curve_mesh's `projection_iterations`)
        n_gl   Gauss--Legendre points per direction, per-element rule (this file only)
        q      Inti's `qorder`, over a whole mesh
        θ      Bernardi smoothness order (= $θ throughout)
        Errors are relative; references are computed independently of curve_mesh.
        """
    )
    r = refs!()
    @printf(
        "references: bean area %.10f vol %.10f | blob area %.10f vol %.10f\n",
        r[:bean_area], r[:bean_vol], r[:blob_area], r[:blob_vol]
    )
    @printf(
        "sanity: the same area routine on the undeformed sphere gives %.10f (4π = %.10f)\n\n",
        r[:sphere_area], 4π
    )
    for i in which
        SECTIONS[i]()
    end
    return nothing
end

main(isempty(ARGS) ? eachindex(SECTIONS) : parse.(Int, ARGS))
