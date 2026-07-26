#=
    Basic tests for the Green's identities for the single/double layer and adjoint
    double-layer/hypersingular operators using various correction methods.
=#

using Test
using LinearAlgebra
using Inti
using Random
using StaticArrays
using QPGreen
using ForwardDiff

include("test_utils.jl")

Random.seed!(1)

## parameters for testing
rtol1 = 1.0e-2 # single and double layer
rtol2 = 5.0e-2 # hypersingular (higher tolerance to avoid use of fine mesh + long unit tests)
dims = (2, 3)
meshsize    = 0.2  # 2D mesh
meshsize_3d = 0.2  # same as 2D; kept separate so it can be tuned independently
types = (:interior, :exterior)

# corrections defined inside the loop to support dimension-dependent tolerances

# Build mesh once per dimension and loop corrections inside, so the geometry is not
# recreated for every correction method.
for N in dims
    Inti.clear_entities!()
    tol_adaptive = N == 2 ? 1.0e-2 : 1.0e-3
    corrections = [
        (method = :dim,),
        (method = :adaptive, maxdist = 2 * (N == 2 ? meshsize : meshsize_3d), rtol = tol_adaptive, atol = tol_adaptive)
    ]
    msize = N == 2 ? meshsize : meshsize_3d
    local Γ
    if N == 2
        Γ = Inti.parametric_curve(x -> SVector(cos(x), sin(x)), 0.0, 2π) |> Inti.Domain
        quad = Inti.Quadrature(Γ; meshsize = msize, qorder = 3)
    else
        Ω = Inti.GeometricEntity("ellipsoid") |> Inti.Domain
        Γ = Inti.external_boundary(Ω)
        quad = Inti.Quadrature(Γ; meshsize = msize, qorder = 3)
    end

    for correction in corrections
        @testset "Method = $(correction.method), $(N)d" begin
            ops = (
                Inti.Laplace(; dim = N),
                Inti.Helmholtz(; k = 1.2, dim = N),
                Inti.Stokes(; μ = 1.2, dim = N),
                Inti.Elastostatic(; λ = 1, μ = 1, dim = N),
            )
            # periodic operators only defined in 2D
            if N == 2
                ops = (
                    Inti.LaplacePeriodic1D(; dim = N, period = 2π),
                    Inti.HelmholtzPeriodic1D(; alpha = 0.3, k = 1.2, dim = N),
                    ops...,
                )
            end

            for op in ops
                @testset "Greens identity $(N)d $op" begin
                    # HelmholtzPeriodic1D with adaptive correction needs higher quadrature order.
                    quad_op = if op isa
                                 Base.get_extension(Inti, :IntiQPGreenExt).HelmholtzPeriodic1D &&
                                 correction.method == :adaptive
                        Inti.Quadrature(Γ; meshsize = msize, qorder = 5)
                    else
                        quad
                    end

                    # ── Single / double layer ─────────────────────────────────────────────
                    # Uncorrected and corrected operators are independent of the test point
                    # (interior vs exterior), so build them once and reuse across both types.
                    Smat = Inti.assemble_matrix(Inti.IntegralOperator(Inti.SingleLayerKernel(op), quad_op))
                    Dmat = Inti.assemble_matrix(Inti.IntegralOperator(Inti.DoubleLayerKernel(op), quad_op))
                    S, D = Inti.single_double_layer(;
                        op,
                        target = quad_op,
                        source = quad_op,
                        compression = (method = :none,),
                        correction,
                    )

                    @testset "Single/double layer $(string(op))" begin
                        for t in types
                            σ  = t == :interior ? 1 / 2 : -1 / 2
                            xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
                            T  = Inti.default_density_eltype(op)
                            c  = rand(T)
                            u    = (qnode) -> Inti.SingleLayerKernel(op)(qnode, xs) * c
                            dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(op)(qnode, xs) * c
                            γ₀u = map(u, quad_op)
                            γ₁u = map(dudn, quad_op)
                            γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
                            e0 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                            e1 = norm(S * γ₁u - D * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                            @test norm(e0, Inf) > norm(e1, Inf)
                            @test norm(e1, Inf) < rtol1
                        end
                    end

                    # ── Adjoint double-layer / hypersingular ──────────────────────────────
                    Kmat = Inti.assemble_matrix(Inti.IntegralOperator(Inti.AdjointDoubleLayerKernel(op), quad_op))
                    Hmat = Inti.assemble_matrix(Inti.IntegralOperator(Inti.HyperSingularKernel(op), quad_op))
                    K, H = Inti.adj_double_layer_hypersingular(;
                        op,
                        target = quad_op,
                        source = quad_op,
                        compression = (method = :none,),
                        correction,
                    )

                    @testset "Adjoint double-layer/hypersingular $(string(op))" begin
                        for t in types
                            σ  = t == :interior ? 1 / 2 : -1 / 2
                            xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
                            T  = Inti.default_density_eltype(op)
                            c  = rand(T)
                            u    = (qnode) -> Inti.SingleLayerKernel(op)(qnode, xs) * c
                            dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(op)(qnode, xs) * c
                            γ₀u = map(u, quad_op)
                            γ₁u = map(dudn, quad_op)
                            γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
                            e0 = norm(Kmat * γ₁u - Hmat * γ₀u - σ * γ₁u, Inf) / γ₁u_norm
                            e1 = norm(K * γ₁u - H * γ₀u - σ * γ₁u, Inf) / γ₁u_norm
                            @test norm(e0, Inf) > norm(e1, Inf)
                            @test norm(e1, Inf) < rtol2
                        end
                    end
                end
            end
        end
    end
end

## Gradient Green's identity: W*γ₁u - GDL*γ₀u = ∇u (interior representation formula)
@testset "Gradient Green's identity (kernel_variant = :gradient)" begin
    for N in (2, 3)
        for op in (Inti.Laplace(; dim = N), Inti.Helmholtz(; k = 1.2, dim = N),
                   Inti.Elastostatic(; μ = 0.8, λ = 1.3, dim = N),
                   Inti.Stokes(; μ = 1.2, dim = N))
            Inti.clear_entities!()
            msize = N == 2 ? meshsize : meshsize_3d
            if N == 2
                Γ = Inti.parametric_curve(x -> SVector(cos(x), sin(x)), 0.0, 2π) |> Inti.Domain
                quad = Inti.Quadrature(Γ; meshsize = msize, qorder = 5)
                target = vec([SVector(x, y) for x in -0.9:0.2:0.9, y in -0.9:0.2:0.9
                              if x^2 + y^2 < 0.85])
            else
                Ω = Inti.GeometricEntity("ellipsoid") |> Inti.Domain
                Γ = Inti.external_boundary(Ω)
                quad = Inti.Quadrature(Γ; meshsize = msize, qorder = 3)
                target = vec([SVector(x, y, z) for x in -0.7:0.3:0.7, y in -0.7:0.3:0.7,
                              z in -0.7:0.3:0.7 if x^2 + y^2 + z^2 < 0.5])
            end
            xs = ntuple(i -> 3, N)
            T = Inti.default_density_eltype(op)
            c = rand(T)
            u      = qnode -> Inti.SingleLayerKernel(op)(qnode, xs) * c
            dudn   = qnode -> Inti.AdjointDoubleLayerKernel(op)(qnode, xs) * c
            ∇u_ref = x     -> Inti.GradientSingleLayerKernel(op)(x, xs) * c
            γ₀u  = map(u, quad)
            γ₁u  = map(dudn, quad)
            ∇u   = map(∇u_ref, target)
            ∇u_norm = norm(norm.(∇u), Inf)
            # uncorrected
            Wmat   = Inti.assemble_matrix(Inti.IntegralOperator(Inti.GradientSingleLayerKernel(op), target, quad))
            GDLmat = Inti.assemble_matrix(Inti.IntegralOperator(Inti.GradientDoubleLayerKernel(op), target, quad))
            e0 = norm(Wmat * γ₁u - GDLmat * γ₀u - ∇u, Inf) / ∇u_norm
            # corrected
            W, GDL = Inti.single_double_layer(;
                op,
                target,
                source = quad,
                compression = (method = :none,),
                correction = (method = :dim, maxdist = Inf, target_location = :inside),
                kernel_variant = :gradient,
            )
            e1 = norm(W * γ₁u - GDL * γ₀u - ∇u, Inf) / ∇u_norm
            @testset "Gradient identity $(N)d $(typeof(op).name.name)" begin
                @test e0 > e1
                @test e1 < rtol1
            end
        end
    end
end
