# Tests for the SparsifyAndSweep coupling (ext/IntiSparsifyAndSweepExt).
using Inti
using SparsifyAndSweep
using StaticArrays
using LinearAlgebra, Random, Test

# everything the coupling defines lives in the package extension, which is not
# `using`-able by name
const SAS = Base.get_extension(Inti, :IntiSparsifyAndSweepExt)
@test SAS !== nothing

using .SAS: PhysicalGrid, physcoords, unitcoords, unit_frequency, sample_field,
    check_support, velocity_from_contrast,
    interpolation_matrix, renormalized_transpose, interpolation_reach,
    covered_nodes, roundtrip_error,
    HybridPreconditioner, apply, node_coordinates, timings, reset_timings!, timed_gmres!

@testset "physical grid" begin
    pg = PhysicalGrid(Val(2), (-1.0, -1.0), (1.0, 1.0), 63; b = 8, pad = 0.5)
    @test dim(pg) == 2
    @test pg.L ≈ 2 * (1 + 2 * 0.5)                     # side * (1 + 2pad)
    @test all(pg.origin .≈ -pg.L / 2)                  # centred on the box
    # the affine map and its inverse agree
    for i in ((1, 1), (17, 40), (63, 63))
        @test all(unitcoords(pg, physcoords(pg, i)) .≈ i .* pg.g.h)
    end
    # the rescale is only meaningful if it preserves points per wavelength
    ω = 3.7
    ω_u = unit_frequency(pg, ω)
    @test ω_u ≈ ω * pg.L
    λ_phys, h_phys = 2π / ω, pg.L * pg.g.h
    λ_unit, h_unit = 2π / ω_u, pg.g.h
    @test λ_phys / h_phys ≈ λ_unit / h_unit

    # sampling, support check and the contrast/velocity conversion
    f = x -> hypot(x...) < 1.0 ? -1.25 : 0.0
    m = sample_field(f, pg)
    @test size(m) == (63, 63)
    @test check_support(m, pg) == 0.0
    c = velocity_from_contrast(m)
    @test maximum(c) ≈ 1 / sqrt(1 - (-1.25) * 0 + 0) atol = 1.0e-12   # c = 1 outside
    @test minimum(c) ≈ 1 / sqrt(1 + 1.25)
    @test_throws ArgumentError velocity_from_contrast([2.0])
    # a scatterer that reaches the PML must be rejected
    pg2 = PhysicalGrid(Val(2), (-1.0, -1.0), (1.0, 1.0), 63; b = 8, pad = 0.01)
    @test_throws ArgumentError check_support(sample_field(f, pg2), pg2)
end

@testset "transfer operators" begin
    n = 41
    pg = PhysicalGrid(Val(2), (-1.0, -1.0), (1.0, 1.0), n; b = 6, pad = 0.5)

    # (a) nodes placed ON the Cartesian nodes: both transfers are the identity,
    #     exactly, which is the sharp check that the indexing is consistent
    grid_nodes = vec([physcoords(pg, Tuple(c)) for c in CartesianIndices((n, n))])
    W0 = interpolation_matrix(pg, grid_nodes)
    # not bitwise identity: the coordinate round trip carries ~1e-16 roundoff
    @test norm(W0 - I) < 1.0e-12
    @test norm(renormalized_transpose(W0) - I) < 1.0e-12

    # (b) a scattered set: interpolation reproduces constants and linears
    Random.seed!(7)
    Nq = 3000
    nodes = [(1.4 * (2rand() - 1), 1.4 * (2rand() - 1)) for _ in 1:Nq]
    W = interpolation_matrix(pg, nodes)
    @test size(W) == (Nq, n^2)
    @test all(≈(1), vec(sum(W; dims = 2)))                       # rows sum to one
    @test count(!iszero, W[1, :]) <= 4                            # 2^D stencil
    gx = [physcoords(pg, Tuple(c))[1] for c in CartesianIndices((n, n))] |> vec
    @test maximum(abs, W * gx .- [x[1] for x in nodes]) < 1.0e-12   # bilinear is exact on x

    # (c) the renormalised transpose preserves constants where it reaches
    T = renormalized_transpose(W)
    cov = covered_nodes(W)
    @test !isempty(cov)
    @test maximum(abs, (T * ones(ComplexF64, Nq))[cov] .- 1) < 1.0e-12
    @test all(iszero, (T * ones(ComplexF64, Nq))[setdiff(1:(n^2), cov)])
    @test roundtrip_error(T, W, ones(ComplexF64, Nq)) < 1.0e-12

    lo, hi = interpolation_reach(pg, nodes)
    @test 0 < lo < hi < 1
end

@testset "hybrid preconditioner" begin
    n, ω = 41, 2π * 2
    pg = PhysicalGrid(Val(2), (-1.0, -1.0), (1.0, 1.0), n; b = 6, pad = 0.5)
    m = sample_field(x -> hypot(x...) < 0.7 ? -0.6 : 0.0, pg)
    check_support(m, pg)
    P = LSProblem(
        n, unit_frequency(pg, ω), velocity_from_contrast(m); b = 6,
        withconv = false
    )
    S = SweepPreconditioner(P; mode = :direct)

    # withconv = false really does skip the FFT operator
    @test P.conv === nothing
    @test_throws ErrorException apply_A(P, zeros(ComplexF64, n^2))

    # with the nodes on the Cartesian grid both transfers are the identity, so
    # P_Ω must reduce to the Cartesian preconditioner *exactly*
    grid_nodes = vec([physcoords(pg, Tuple(c)) for c in CartesianIndices((n, n))])
    W = interpolation_matrix(pg, grid_nodes)
    T = renormalized_transpose(W)
    b = randn(ComplexF64, n^2)
    H = HybridPreconditioner(S, T, W)
    @test size(H) == (n^2, n^2)
    @test norm(apply(H, b) - apply_M(S, b)) / norm(b) < 1.0e-12

    # on a genuine node set it is still linear, and agrees through all three APIs
    Random.seed!(11)
    nodes = [(1.2 * (2rand() - 1), 1.2 * (2rand() - 1)) for _ in 1:800]
    W2 = interpolation_matrix(pg, nodes)
    H2 = HybridPreconditioner(S, renormalized_transpose(W2), W2)
    b1 = randn(ComplexF64, 800)
    b2 = randn(ComplexF64, 800)
    α = 0.4 - 1.1im
    @test norm(apply(H2, α * b1 + b2) - (α * apply(H2, b1) + apply(H2, b2))) /
        norm(apply(H2, b1)) < 1.0e-10
    @test H2 \ b1 == apply(H2, b1)
    y = similar(b1)
    ldiv!(y, H2, b1)
    @test y == apply(H2, b1)
    z = copy(b1)
    ldiv!(H2, z)
    @test z == apply(H2, b1)

    # every application is timed, stage by stage, until reset
    reset_timings!(H2)
    @test timings(H2).napply == 0 && timings(H2).total == 0
    apply(H2, b1); H2 \ b2
    τ = timings(H2)
    @test τ.napply == 2
    @test τ.total ≈ τ.to_grid + τ.sparsifying + τ.from_grid
    @test all(>(0), (τ.to_grid, τ.sparsifying, τ.from_grid))
    @test timings(reset_timings!(H2)).napply == 0

    # the timed GMRES matches the plain one and accounts for all of its time
    A = I + 0.1 * randn(ComplexF64, 800, 800) / sqrt(800)
    u, hist, t = timed_gmres!(zeros(ComplexF64, 800), A, b1; Pl = H2, reltol = 1.0e-10)
    @test hist.isconverged && norm(A * u - b1) / norm(b1) < 1.0e-9
    @test t.nforward == t.nprecond >= hist.iters
    @test timings(H2).napply == t.nprecond
    @test t.forward + t.precond + t.other ≈ t.total
    @test t.sparsifying + t.transfer ≈ t.precond
    _, _, t0 = timed_gmres!(zeros(ComplexF64, 800), A, b1; reltol = 1.0e-10)
    @test t0.nprecond == 0 && t0.precond == 0 && isnan(t0.sparsifying)

    # flexible GMRES: the same solution with a linear preconditioner, ...
    uf, hf, tf = timed_gmres!(
        zeros(ComplexF64, 800), A, b1; Pl = H2, reltol = 1.0e-10,
        flexible = true, restart = 100
    )
    @test hf.isconverged && norm(A * uf - b1) / norm(b1) <= 1.0e-10   # true residual
    @test norm(uf - u) / norm(u) < 1.0e-8
    # one cycle: a forward map per iteration, plus the initial and the final true residual
    @test tf.nprecond == hf.iters && tf.nforward == hf.iters + 2
    @test tf.forward + tf.precond + tf.other ≈ tf.total
    @test !SAS.is_nonlinear(H2)
    # ... and it is chosen by itself when the sweep runs inner iterations
    Si = SweepPreconditioner(P; mode = :sweep, inner = 4, inner_tol = 1.0e-3)
    Hi = HybridPreconditioner(Si, renormalized_transpose(W2), W2)
    @test SAS.is_nonlinear(Hi)
    ui, hi, ti = timed_gmres!(zeros(ComplexF64, 800), A, b1; Pl = Hi, reltol = 1.0e-10)
    @test hi isa NamedTuple && hi.isconverged
    @test norm(A * ui - b1) / norm(b1) <= 1.0e-10
    @test ti.nsweeps > ti.nprecond                  # several sweeps per application
    @test_throws ArgumentError timed_gmres!(
        zeros(ComplexF64, 800), A, b1; Pl = Hi,
        verbose = true
    )

    # the operator really is I + T_C^Q(S - I)T_Q^C, spelled out
    Tq = renormalized_transpose(W2)
    expect = b1 + W2 * (apply_M(S, Tq * b1) - Tq * b1)
    @test norm(apply(H2, b1) - expect) / norm(b1) < 1.0e-12
end

@testset "node_coordinates" begin
    # bare coordinates pass through untouched ...
    pts = [(0.1, 0.2), (0.3, 0.4)]
    @test node_coordinates(pts) === pts
    pg = PhysicalGrid(Val(2), (-1.0, -1.0), (1.0, 1.0), 21; b = 4, pad = 0.5)
    @test interpolation_matrix(pg, pts) == interpolation_matrix(pg, node_coordinates(pts))

    # ... and this is what the extension is for: an Inti quadrature goes into the
    # transfer operators directly, in the order its nodes carry
    Γ = Inti.parametric_curve(t -> SVector(cos(t), sin(t)), 0, 2π) |> Inti.Domain
    quad = Inti.Quadrature(Inti.meshgen(Γ; meshsize = 0.1); qorder = 3)
    coords = node_coordinates(quad)
    @test length(coords) == length(quad)
    @test coords == [q.coords for q in quad]
    @test node_coordinates(collect(quad)) == coords
    # a grid big enough to hold the unit circle with room for the PML
    pgd = PhysicalGrid(Val(2), (-1.0, -1.0), (1.0, 1.0), 41; b = 4, pad = 0.5)
    Wq = interpolation_matrix(pgd, quad)
    @test size(Wq) == (length(quad), 41^2)
    @test all(≈(1), vec(sum(Wq; dims = 2)))
    @test Wq == interpolation_matrix(pgd, coords)
end

@testset "3D transfer operators" begin
    # the coupling is dimension-generic even though the solves are tested in 2D
    n = 15
    pg = PhysicalGrid(Val(3), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), n; b = 3, pad = 0.5)
    @test dim(pg) == 3
    grid_nodes = vec([physcoords(pg, Tuple(c)) for c in CartesianIndices((n, n, n))])
    @test norm(interpolation_matrix(pg, grid_nodes) - I) < 1.0e-12

    Random.seed!(3)
    nodes = [(1.2 * (2rand() - 1), 1.2 * (2rand() - 1), 1.2 * (2rand() - 1)) for _ in 1:500]
    W = interpolation_matrix(pg, nodes)
    @test size(W) == (500, n^3)
    @test all(≈(1), vec(sum(W; dims = 2)))
    @test count(!iszero, W[1, :]) <= 8                 # 2^3 stencil
    T = renormalized_transpose(W)
    @test maximum(abs, (T * ones(ComplexF64, 500))[covered_nodes(W)] .- 1) < 1.0e-12
end
