using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

# # [Lippmann-Schwinger with a sparsify-and-sweep preconditioner](@id lippmann_schwinger_sparsify_and_sweep)

# !!! note "Important points covered in this example"
#       - Preconditioning a volume integral equation with a Cartesian sweeping preconditioner
#       - Coupling an unstructured VDIM discretisation to a structured grid
#       - Checking the result against the exact Mie series

# !!! warning "Extra dependency"
#       This example needs
#       [SparsifyAndSweep.jl](https://github.com/IntegralEquations/SparsifyAndSweep.jl),
#       which is unregistered and therefore not a dependency of the docs.  Add it
#       to the docs environment first, e.g.
#       ```
#       julia --project=docs -e 'using Pkg; Pkg.develop(path = "../SparsifyAndSweep.jl")'
#       ```
#       Loading it alongside `Inti` triggers `IntiSparsifyAndSweepExt`, which
#       supplies everything used below.

# ## The idea
#
# The Lippmann-Schwinger equation
# ```math
#   u + k^2 \mathcal{V}_k[m\,u] = u^{\textit{inc}} \quad \text{in } \Omega
# ```
# discretised with VDIM gives a dense, badly-conditioned system:
# unpreconditioned GMRES needs a number of iterations that grows with the
# frequency.
#
# Liu and Ying's sparsify-and-sweep preconditioner fixes that on a uniform
# Cartesian grid: the dense Nyström system is first sparsified into a
# ``3^D``-point stencil system ``H`` (a PML-truncated discretisation of the
# Helmholtz operator obtained by data fitting), and ``H`` is then inverted
# approximately by a moving-PML sweeping factorisation.  Both stages are
# ``O(N)`` and the iteration count is essentially frequency independent for
# smooth media (discontinuous media appears to be more challenging; Ying's
# sparsifying preconditioner, usable via :direct rather than :sweep, performs
# better in this regime).
#
# To use it on an unstructured discretisation we sandwich it between transfer
# operators with the hybrid preconditioner
# ```math
#   P = I + T_C^Q \left( S^{(C)} - I \right) T_Q^C
# ```
# acting on the ``N_Q`` unstructured unknowns: interpolate onto a Cartesian grid
# with ``T_Q^C``, apply the Cartesian preconditioner ``S^{(C)}``, come back with
# ``T_C^Q``.  The two resolutions are independent; in this example a grid 80
# times smaller than the mesh buys most of the benefit of the preconditioner.

using LinearAlgebra, SparseArrays, Printf
using OrderedCollections
using Inti, LinearMaps, IterativeSolvers
using FMM2D                      # loads Inti's FMM2D extension (compression = :fmm)
using Gmsh
using SpecialFunctions

using SparsifyAndSweep           # loads IntiSparsifyAndSweepExt

# Extension types aren't `using`-able:
const SAS = Base.get_extension(Inti, :IntiSparsifyAndSweepExt)
using .SAS: PhysicalGrid, unit_frequency, sample_field, physcoords,
    check_support, velocity_from_contrast,
    interpolation_matrix, renormalized_transpose, interpolation_reach,
    covered_nodes, roundtrip_error, node_coordinates, HybridPreconditioner

# ## Geometry, quadrature, and the exact solution
#
# A curved triangular mesh of a disk with a Vioreanu-Rokhlin volume quadrature,
# plus the Mie series to check against.  `occ.addDisk` represents the boundary as
# one periodic curve, and gmsh silently stops honouring `MeshSizeMax` once a
# single curve needs more than ~4100 subdivisions, so above that threshold the
# boundary is built from four quarter arcs instead.

function gmsh_disk(; name, meshsize, order = 2, center = (0.0, 0.0), radius = 1.0)
    ndiv = 2π * radius / meshsize
    try
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.add("circle-mesh")
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        cx, cy = center
        if ndiv ≤ 3000
            gmsh.model.occ.addDisk(cx, cy, 0.0, radius, radius)
            gmsh.model.occ.synchronize()
        else
            C = gmsh.model.geo.addPoint(cx, cy, 0.0, meshsize)
            pE = gmsh.model.geo.addPoint(cx + radius, cy, 0.0, meshsize)
            pN = gmsh.model.geo.addPoint(cx, cy + radius, 0.0, meshsize)
            pW = gmsh.model.geo.addPoint(cx - radius, cy, 0.0, meshsize)
            pS = gmsh.model.geo.addPoint(cx, cy - radius, 0.0, meshsize)
            arcs = [
                gmsh.model.geo.addCircleArc(pE, C, pN),
                gmsh.model.geo.addCircleArc(pN, C, pW),
                gmsh.model.geo.addCircleArc(pW, C, pS),
                gmsh.model.geo.addCircleArc(pS, C, pE),
            ]
            loop = gmsh.model.geo.addCurveLoop(arcs)
            gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()
        end
        gmsh.model.mesh.generate(2)
        ntri = sum(length.(gmsh.model.mesh.getElements(2)[2]))
        nexp = 4π * radius^2 / (sqrt(3) * meshsize^2)
        ntri ≥ 0.2 * nexp || error(
            "gmsh_disk under-resolved: $ntri triangles, expected ≈$(round(Int, nexp)) " *
            "at meshsize=$meshsize (per-curve cap?)",
        )
        gmsh.model.mesh.setOrder(order)
        gmsh.write(name)
    finally
        gmsh.finalize()
    end
end

# The mesh plus its volume and boundary quadratures.  `Ωₕ_quad` carries the
# unstructured unknowns.

function disk_geometry(
        k; h_T, interpolation_order::Int = 2, order::Int = 2, radius::Real = 1.0,
        mesh_name::AbstractString = joinpath(tempdir(), "sas_disk.msh"),
    )
    qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(interpolation_order)
    op = Inti.Helmholtz(; dim = 2, k = k)

    Inti.clear_entities!()
    mkpath(dirname(mesh_name))
    gmsh_disk(; name = mesh_name, meshsize = h_T, order = order, radius = radius)
    msh = Inti.import_mesh(mesh_name; dim = 2)
    Ω = Inti.Domain(ent -> Inti.geometric_dimension(ent) == 2, Inti.entities(msh))
    Γ = Inti.boundary(Ω)
    Ωₕ = view(msh, Ω)
    Γₕ = view(msh, Γ)
    Q = Inti.VioreanuRokhlin(; domain = :triangle, order = qorder)
    dict = OrderedDict(E => Q for E in Inti.element_types(Ωₕ))
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder)
    Ωₕ_quad = Inti.Quadrature(Ωₕ, dict)

    return (;
        op, k, h_T, interpolation_order, qorder, msh, Ω, Γ, Ωₕ, Γₕ,
        Γₕ_quad, Ωₕ_quad, Ndof = length(Ωₕ_quad),
    )
end

# `ν(x,y) = 1 - n²` with `n² = 1 + η` inside the disk: the `m` of the
# Lippmann-Schwinger equation as SparsifyAndSweep writes it, where the local
# wavenumber is `ω²(1-m)`.

constant_contrast(η; radius = 1.0) =
    (x, y) -> hypot(x, y) < radius ? -float(η) : 0.0

# Exact 2D Mie series for a penetrable circular cylinder of constant contrast
# `η = n² - 1` and incident field `exp(ik(x cosθin + y sinθin))`.  Returns the
# *total* field: `u_inc + u_scat` outside, the transmitted field inside.

function mie_penetrable_disk(pts; radius = 1.0, k, η, θin = 0.0, tol = 1.0e-12)
    R = float(radius)
    k_r = k * sqrt(complex(1 + η))
    Npts = length(pts)

    xs = Vector{Float64}(undef, Npts)
    ys = Vector{Float64}(undef, Npts)
    rs = Vector{Float64}(undef, Npts)
    e1 = Vector{ComplexF64}(undef, Npts)
    @inbounds for i in 1:Npts
        p = pts[i]
        xs[i] = float(p[1])
        ys[i] = float(p[2])
        rs[i] = hypot(xs[i], ys[i])
        e1[i] = cis(atan(ys[i], xs[i]))
    end
    ext = rs .> R

    u = zeros(ComplexF64, Npts)
    cθi, sθi = cos(θin), sin(θin)
    @inbounds for i in 1:Npts
        ext[i] && (u[i] = exp(im * k * (xs[i] * cθi + ys[i] * sθi)))
    end

    Jd(n, z) = (besselj(n - 1, z) - besselj(n + 1, z)) / 2
    Hd(n, z) = (besselh(n - 1, z) - besselh(n + 1, z)) / 2
    @inline function mie_coeffs(n)
        αn = im^n * exp(-im * n * θin)
        Jn = besselj(n, k * R)
        Hn = besselh(n, k * R)
        Jrn = besselj(n, k_r * R)
        kHdn = k * Hd(n, k * R)
        kJdn = k * Jd(n, k * R)
        krJrdn = k_r * Jd(n, k_r * R)
        Dn = kHdn * Jrn - krJrdn * Hn
        return αn * (krJrdn * Jn - kJdn * Jrn) / Dn, αn * 2im / (π * R * Dn)
    end

    a0, b0 = mie_coeffs(0)
    @inbounds for i in 1:Npts
        u[i] += ext[i] ? a0 * besselh(0, k * rs[i]) : b0 * besselj(0, k_r * rs[i])
    end

    en = copy(e1)
    converged = falses(Npts)
    n = 1.0
    while !all(converged)
        an, bn = mie_coeffs(n)
        amn, bmn = mie_coeffs(-n)
        s = iseven(n) ? 1 : -1                       # H_{-n} = sH_n, J_{-n} = sJ_n
        @inbounds for i in 1:Npts
            converged[i] && continue
            eni = en[i]
            emni = conj(eni)
            if ext[i]
                Hn_val = besselh(n, k * rs[i])
                tn = an * Hn_val * eni
                tmn = amn * (s * Hn_val) * emni
            else
                Jn_val = besselj(n, k_r * rs[i])
                tn = bn * Jn_val * eni
                tmn = bmn * (s * Jn_val) * emni
            end
            abs(tn) + abs(tmn) ≤ tol ? (converged[i] = true) : (u[i] += tn + tmn)
        end
        @inbounds for i in 1:Npts
            en[i] *= e1[i]
        end
        n += 1
    end
    return u
end

# Cartesian spacing of the reference hybrid solver, `h_C = 2^{-(ppwind-1)}/k`.

ppw_hC(k, ppwind::Int) = 2.0^(-(ppwind - 1)) / k

# ## Parameters
#
# A penetrable disk of radius 1 with `n² = 1 + η`; the knobs are environment
# variables so the same script can be swept.

#k = parse(Float64, get(ENV, "K", "5.0"))
k = 10.0
η = 1.25
radius = 1.0
ppwind = 4
gapλ = parse(Float64, get(ENV, "GAP", "2.0"))  # scatterer-to-box gap, in wavelengths
mode = Symbol(get(ENV, "MODE", "sweep"))       # :sweep or :direct
levels = parse(Int, get(ENV, "LEVELS", "1"))
reltol = 1.0e-8
fmm_tol = 1.0e-8

h_C = ppw_hC(k, ppwind)
h_T = h_C / sqrt(1 + η)
ν = constant_contrast(η; radius)
@printf(
    "k = %.2f   h_C = %.5f   h_T = %.5f   contrast ν = %.3f inside\n",
    k, h_C, h_T, ν(0.0, 0.0)
)

# ## The unstructured half

geom = disk_geometry(k; h_T, interpolation_order = 2, order = 2, radius)
# the extension lets the quadrature go straight into the transfer operators;
# `nodes` is only kept for the Mie comparison below
nodes = node_coordinates(geom.Ωₕ_quad)
Nq = geom.Ndof
m_loc = ComplexF64[ν(q.coords[1], q.coords[2]) for q in geom.Ωₕ_quad]
@printf("unstructured DOFs N_Q = %d\n", Nq)

# ## The Cartesian half
#
# The preconditioner grid need **not** match `h_C`: the transfer operators couple
# two independent resolutions.  What matters is that the `b` layers of PML are a
# wavelength or so thick, which ties `b` to the points-per-wavelength of *this*
# grid, not to the unstructured mesh.  `PPWC = 0` means "match `h_C`".
#
# The box must fit the scatterer **plus** a PML about one wavelength thick
# **plus** some clearance, so the padding is set in wavelengths rather than as a
# fraction of the scatterer: a 4x4 box around a unit disk leaves only 0.8λ at
# `k = 5`, which is not enough for the PML and costs ~50% more iterations.

ppw_cart = parse(Float64, get(ENV, "PPWC", "0"))
#bpml = parse(Int, get(ENV, "B", "0"))
bpml = 8
λ = 2π / k
h_cart = ppw_cart > 0 ? λ / ppw_cart : h_C
pad = gapλ * λ / (2 * radius)               # PhysicalGrid pads by pad*side
pg_n = max(15, round(Int, (2 * radius + 2 * gapλ * λ) / h_cart) - 1)
bpml = bpml > 0 ? bpml : max(4, round(Int, λ / h_cart))          # ≈ one wavelength
pg = PhysicalGrid(Val(2), (-radius, -radius), (radius, radius), pg_n; b = bpml, pad)
@printf(
    "Cartesian grid: n = %d (N = %d), L = %.3f, h = %.5f  (%.1f pts/wavelength)\n",
    pg_n, pg_n^2, pg.L, pg.L * pg.g.h, (2π / k) / (pg.L * pg.g.h)
)
@printf(
    "PML: b = %d layers = %.3f = %.2f wavelengths;  gap %.2f λ, clearance %.2f λ\n",
    bpml, bpml * pg.L * pg.g.h, bpml * pg.L * pg.g.h / λ, gapλ,
    gapλ - bpml * pg.L * pg.g.h / λ
)

# The medium is sampled on the Cartesian grid and must vanish inside the PML
# shell, which `check_support` verifies.  `unit_frequency` performs the exact
# rescale onto the unit cube the Cartesian solver works on.

m_grid = sample_field(x -> ν(x[1], x[2]), pg)
check_support(m_grid, pg)
ω_unit = unit_frequency(pg, k)
@printf(
    "unit-cube frequency ω' = ωL = %.3f  (%.1f wavelengths across the box)\n",
    ω_unit, ω_unit / 2π
)

# `withconv = true` builds the FFT operator for `I + ω²KM` as well as the
# preconditioner, so this same object can be *solved*, not just used to
# precondition — see the structured reference solve below.  A run that only ever
# needs `S` can pass `withconv = false` and skip it.

Pcart = LSProblem(pg_n, ω_unit, velocity_from_contrast(m_grid); b = bpml, withconv = true)
t_cart = @elapsed S = SweepPreconditioner(Pcart; mode, levels)
@printf("Cartesian preconditioner (mode=%s, levels=%d): %.2f s\n", mode, levels, t_cart)

# ## The preconditioner on its home turf
#
# Before coupling it to the mesh, solve the *structured* problem the
# preconditioner was actually built for. No transfer operators are involved, so
# this is the preconditioner's best case, and the difference from the hybrid
# count reported below is the price of the coupling — the interpolation error of
# `T_C^Q`/`T_Q^C`, plus the fact that the mesh resolves a curved boundary that
# the Cartesian grid can only step through.

uI_C = plane_wave(Val(2), pg_n, ω_unit, (1.0, 0.0))
b_C = rhs(Pcart, uI_C)
r_C = solve(Pcart, b_C; M = S, tol = reltol, restart = 50, maxiter = 50)
@printf(
    "structured (Cartesian) solve: N_C = %d, iters = %d, rel. residual = %.2e%s\n",
    pg_n^2, r_C.iters, r_C.relres, r_C.converged ? "" : "   (NOT CONVERGED)"
)

# ## The transfer operators
#
# `T_C^Q` is multilinear interpolation (`2^D` points per node, rows summing to
# one) and `T_Q^C` is its column-renormalised transpose, which preserves
# constants — the round trip on `ones` is the consistency check.

t_tr = @elapsed begin
    W = interpolation_matrix(pg, geom.Ωₕ_quad)   # T_C^Q  (Cartesian -> nodes)
    Tqc = renormalized_transpose(W)              # T_Q^C  (nodes -> Cartesian)
end
lo, hi = interpolation_reach(pg, geom.Ωₕ_quad)
@printf(
    "transfers: %.2f s   node reach = (%.4f, %.4f) in the unit cube   covered %d/%d\n",
    t_tr, lo, hi, length(covered_nodes(W)), pg_n^2
)
@printf(
    "round trip on constants ‖T_C^Q T_Q^C 1 − 1‖/‖1‖ = %.2e\n",
    roundtrip_error(Tqc, W, ones(ComplexF64, Nq))
)

Pl = HybridPreconditioner(S, Tqc, W)

# ## The forward operator
#
# `L = I + k² V_dd M`, with the volume potential compressed by the FMM and
# corrected by DIM.

t_V = @elapsed V_dd = Inti.volume_potential(;
    op = geom.op, target = geom.Ωₕ_quad, source = geom.Ωₕ_quad,
    compression = (method = :fmm, tol = fmm_tol),
    correction = (
        method = :dim, maxdist = 7 * h_T,
        interpolation_order = geom.interpolation_order,
    ),
)
@printf("volume potential (d2d): %.2f s\n", t_V)

Lω = LinearMap{ComplexF64}((y, x) -> (y .= x .+ k^2 .* (V_dd * (m_loc .* x))), Nq, Nq)
bΩ = ComplexF64[exp(im * k * q.coords[1]) for q in geom.Ωₕ_quad]

# ## Solve, with and without the preconditioner

function run(label, P)
    u = zeros(ComplexF64, Nq)
    local hist
    t = @elapsed begin
        u, hist = P === nothing ?
            IterativeSolvers.gmres!(
            u, Lω, bΩ; log = true, reltol, abstol = 0.0,
            maxiter = 80, restart = 40
        ) :
            IterativeSolvers.gmres!(
            u, Lω, bΩ; Pl = P, log = true, reltol, abstol = 0.0,
            maxiter = 80, restart = 40
        )
    end
    @printf("%-24s iters = %4d   time = %7.2f s\n", label, hist.iters, t)
    return u, hist.iters
end

println()
u_pre, it_pre = run("hybrid-preconditioned", Pl)
u_non, it_non = run("unpreconditioned", nothing)

# ## Accuracy against the exact Mie series

u_mie = mie_penetrable_disk(nodes; radius, k, η)
relerr(u) = maximum(abs, u .- u_mie) / maximum(abs, u_mie)
@printf(
    "\nrel. max-norm error vs Mie : preconditioned %.3e , unpreconditioned %.3e\n",
    relerr(u_pre), relerr(u_non)
)
@printf("the two solutions agree to : %.3e\n", norm(u_pre - u_non) / norm(u_non))
@printf("iteration reduction        : %.1f×\n", it_non / it_pre)

# At `k = 5`, `η = 1.25` and `N_Q = 157 686` this converges in **7** iterations
# against **29** unpreconditioned, both agreeing with the Mie series to about
# `4e-9`.  Coarsening the Cartesian grid from 50 to 8 points per wavelength (a
# grid 80× smaller than the mesh) costs only 13 iterations, but the PML must stay
# about one wavelength thick as it coarsens: `b` tracks the *grid's*
# points-per-wavelength, not the mesh's.
