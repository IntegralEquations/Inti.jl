using Test
using LinearAlgebra
using Inti
using Random
using StaticArrays

include("test_utils.jl")

Random.seed!(1)

## parameters for testing
rtol1 = 1e-2 # single and double layer
rtol2 = 5e-2 # hypersingular (higher tolerance to avoid use of fine mesh + long unit tests)
dims = (2, 3)
types = (:interior, :exterior)

for N in dims
    # create geometry
    Inti.clear_entities!()
    if N == 2
        Γ = Inti.parametric_curve(x -> SVector(cos(x), sin(x)), 0.0, 2π) |> Inti.Domain
        quad = Inti.Quadrature(Γ; meshsize = 0.5, qorder = 3)
    else
        # Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.2)
        Ω = Inti.GeometricEntity("ellipsoid") |> Inti.Domain
        Γ = Inti.external_boundary(Ω)
        quad = Inti.Quadrature(Γ; meshsize = 0.5, qorder = 3)
    end
    for t in types
        σ = t == :interior ? 1 / 2 : -1 / 2
        ops = (
            Inti.Laplace(; dim = N),
            Inti.Helmholtz(; k = 1.2, dim = N),
            Inti.Stokes(; μ = 1.2, dim = N),
            Inti.Elastostatic(; λ = 1, μ = 1, dim = N),
        )
        for pde in ops
            @testset "Greens identity ($t) $(N)d $pde" begin
                xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
                T = Inti.default_density_eltype(pde)
                c = rand(T)
                u = (qnode) -> Inti.SingleLayerKernel(pde)(qnode, xs) * c
                dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(pde)(qnode, xs) * c
                γ₀u = map(u, quad)
                γ₁u = map(dudn, quad)
                γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
                γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
                # single and double layer
                G = Inti.SingleLayerKernel(pde)
                S = Inti.IntegralOperator(G, quad)
                Smat = Inti.assemble_matrix(S)
                dG = Inti.DoubleLayerKernel(pde)
                D = Inti.IntegralOperator(dG, quad)
                Dmat = Inti.assemble_matrix(D)
                e0 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                Sdim, Ddim = Inti.single_double_layer(;
                    pde,
                    target      = quad,
                    source      = quad,
                    compression = (method = :none,),
                    correction  = (method = :dim,),
                )
                e1 = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                @testset "Single/double layer $(string(pde))" begin
                    @test norm(e0, Inf) > norm(e1, Inf)
                    @test norm(e1, Inf) < rtol1
                end
                # adjoint double-layer and hypersingular.
                pde isa Inti.Stokes && continue # TODO: implement hypersingular for Stokes?

                K = Inti.IntegralOperator(Inti.AdjointDoubleLayerKernel(pde), quad)
                Kmat = Inti.assemble_matrix(K)
                H = Inti.IntegralOperator(Inti.HyperSingularKernel(pde), quad)
                Hmat = Inti.assemble_matrix(H)
                e0 = norm(Kmat * γ₁u - Hmat * γ₀u - σ * γ₁u, Inf) / γ₁u_norm
                Kdim, Hdim = Inti.adj_double_layer_hypersingular(;
                    pde = pde,
                    target = quad,
                    source = quad,
                    compression = (method = :none,),
                    correction = (method = :dim,),
                )
                e1 = norm(Kdim * γ₁u - Hdim * γ₀u - σ * γ₁u, Inf) / γ₁u_norm
                @testset "Adjoint double-layer/hypersingular $(string(pde))" begin
                    @test norm(e0, Inf) > norm(e1, Inf)
                    @test norm(e1, Inf) < rtol2
                end
            end
        end
    end
end
