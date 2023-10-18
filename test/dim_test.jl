using Test
using LinearAlgebra
using Inti
using Random

include("test_utils.jl")

Random.seed!(1)

atol = 5e-2

# for t in (:interior,:exterior)
for N in (2, 3)
    # create geometry
    Inti.clear_entities!()
    if N == 2
        Ω, msh = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.1)
    else
        Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.4)
    end
    Γ    = Inti.external_boundary(Ω)
    quad = Inti.Quadrature(view(msh, Γ); qorder = 3)
    for t in (:interior, :exterior)
        σ = t == :interior ? 1 / 2 : -1 / 2
        ops = (
            Inti.Laplace(; dim = N),
            Inti.Helmholtz(; k = 1.2, dim = N),
            Inti.Stokes(; μ = 1.2, dim = N),
        )
        for pde in ops
            @testset "Greens identity ($t) $(N)d $pde" begin
                xs = t == :interior ? ntuple(i -> 3, N) : ntuple(i -> 0.1, N)
                T = Inti.default_density_eltype(pde)
                c = rand(T)
                u = (qnode) -> Inti.SingleLayerKernel(pde)(qnode, xs) * c
                dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(pde)(qnode, xs) * c
                γ₀u = Inti.NystromDensity(u, quad)
                γ₁u = Inti.NystromDensity(dudn, quad)
                γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
                γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
                # single and double layer
                S = Inti.single_layer_operator(pde, quad)
                Smat = Matrix(S)
                D = Inti.double_layer_operator(pde, quad)
                Dmat = Matrix(D)
                e0 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                δS, δD = Inti.bdim_correction(pde, quad, quad, Smat, Dmat)
                Sdim, Ddim = Smat + δS, Dmat + δD
                e1 = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
                @testset "Single/double layer $(string(pde))" begin
                    @test norm(e0, Inf) > 10 * norm(e1, Inf)
                    @test norm(e1, Inf) < atol
                end
                # adjoint double-layer and hypersingular.
                pde isa Inti.Stokes && continue # TODO: implement hypersingular for Stokes?
                K = Inti.adjoint_double_layer_operator(pde, quad)
                Kmat = Matrix(K)
                H = Inti.hypersingular_operator(pde, quad)
                Hmat = Matrix(H)
                e0 = norm(Kmat * γ₁u - Hmat * γ₀u - σ * γ₁u, Inf)
                δK, δH =
                    Inti.bdim_correction(pde, quad, quad, Kmat, Hmat; derivative = true)
                Kdim, Hdim = Kmat + δK, Hmat + δH
                e1 = norm(Kdim * γ₁u - Hdim * γ₀u - σ * γ₁u, Inf)
                @testset "Adjoint double-layer/hypersingular $(string(pde))" begin
                    @test norm(e0, Inf) > 10 * norm(e1, Inf)
                    @test norm(e1, Inf) < atol
                end
            end
        end
    end
end
