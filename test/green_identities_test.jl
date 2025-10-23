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

## Extend QuadGK to support ForwardDiff.Dual types (see https://github.com/JuliaMath/QuadGK.jl/issues/122)
## (only needed if computing derivatives of QPGreen using ForwardDiff)
# using QuadGK
# function QuadGK.cachedrule(
#     ::Type{<:ForwardDiff.Dual{<:Any,T}},
#     n::Integer,
# ) where {T<:Number}
#     return QuadGK._cachedrule(typeof(float(real(one(T)))), Int(n))
# end

include("test_utils.jl")

Random.seed!(1)

## parameters for testing
rtol1 = 1.0e-2 # single and double layer
rtol2 = 5.0e-2 # hypersingular (higher tolerance to avoid use of fine mesh + long unit tests)
dims = (2, 3)
meshsize = 0.5
types = (:interior, :exterior)

corrections = [(method = :dim,), (method = :adaptive, maxdist = 2 * meshsize, rtol = 1.0e-2)]

for correction in corrections
    @testset "Method = $(correction.method)" begin
        for N in dims
            # create geometry
            Inti.clear_entities!()
            if N == 2
                Γ =
                    Inti.parametric_curve(x -> SVector(cos(x), sin(x)), 0.0, 2π) |>
                    Inti.Domain
                quad = Inti.Quadrature(Γ; meshsize, qorder = 5)
            else
                # Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.2)
                Ω = Inti.GeometricEntity("ellipsoid") |> Inti.Domain
                Γ = Inti.external_boundary(Ω)
                quad = Inti.Quadrature(Γ; meshsize, qorder = 5)
            end
            for t in types
                σ = t == :interior ? 1 / 2 : -1 / 2
                ops = (
                    Inti.Laplace(; dim = N),
                    Inti.Helmholtz(; k = 1.2, dim = N),
                    Inti.Stokes(; μ = 1.2, dim = N),
                    Inti.Elastostatic(; λ = 1, μ = 1, dim = N),
                )
                # periodic Laplace only defined for 2d, so we add it conditionally
                if N == 2
                    ops = (
                        Inti.LaplacePeriodic1D(; dim = N, period = 2π),
                        Inti.HelmholtzPeriodic1D(; alpha = 0.3, k = 1.2, dim = N),
                        ops...,
                    )
                end
                for op in ops
                    @testset "Greens identity ($t) $(N)d $op" begin
                        if op isa
                                Base.get_extension(Inti, :IntiQPGreenExt).HelmholtzPeriodic1D &&
                                correction ==
                                (method = :adaptive, maxdist = 2 * meshsize, rtol = 1.0e-2)
                            quad = Inti.Quadrature(Γ; meshsize = 0.2, qorder = 5)
                        end
                        xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
                        T = Inti.default_density_eltype(op)
                        c = rand(T)
                        u = (qnode) -> Inti.SingleLayerKernel(op)(qnode, xs) * c
                        dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(op)(qnode, xs) * c
                        γ₀u = map(u, quad)
                        γ₁u = map(dudn, quad)
                        γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
                        γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
                        # single and double layer
                        G = Inti.SingleLayerKernel(op)
                        Sop = Inti.IntegralOperator(G, quad)
                        Smat = Inti.assemble_matrix(Sop)
                        dG = Inti.DoubleLayerKernel(op)
                        Dop = Inti.IntegralOperator(dG, quad)
                        Dmat = Inti.assemble_matrix(Dop)
                        e0 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                        S, D = Inti.single_double_layer(;
                            op,
                            target = quad,
                            source = quad,
                            compression = (method = :none,),
                            correction,
                        )
                        e1 = norm(S * γ₁u - D * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                        @testset "Single/double layer $(string(op))" begin
                            @test norm(e0, Inf) > norm(e1, Inf)
                            @test norm(e1, Inf) < rtol1
                        end
                        # adjoint double-layer and hypersingular.
                        if op isa Inti.Stokes
                            # skip cases where hypersingular has not been implemented
                            continue
                        end
                        Kop = Inti.IntegralOperator(Inti.AdjointDoubleLayerKernel(op), quad)
                        Kmat = Inti.assemble_matrix(Kop)
                        Hop = Inti.IntegralOperator(Inti.HyperSingularKernel(op), quad)
                        Hmat = Inti.assemble_matrix(Hop)
                        e0 = norm(Kmat * γ₁u - Hmat * γ₀u - σ * γ₁u, Inf) / γ₁u_norm
                        K, H = Inti.adj_double_layer_hypersingular(;
                            op = op,
                            target = quad,
                            source = quad,
                            compression = (method = :none,),
                            correction,
                        )
                        e1 = norm(K * γ₁u - H * γ₀u - σ * γ₁u, Inf) / γ₁u_norm
                        @testset "Adjoint double-layer/hypersingular $(string(op))" begin
                            @test norm(e0, Inf) > norm(e1, Inf)
                            @test norm(e1, Inf) < rtol2
                        end
                    end
                end
            end
        end
    end
end
