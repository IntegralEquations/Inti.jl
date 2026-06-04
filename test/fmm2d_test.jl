using Test
using Inti
using FMM2D
using Gmsh
using LinearAlgebra
using Random

include("test_utils.jl")

# create a boundary and area meshes and quadrature only once
Ω₁, msh₁ = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.1)
Ω₂, msh₂ = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.05)

# Test on two meshes to test both sources == targets, and not.
Γ₁ = Inti.external_boundary(Ω₁)
Γ₁_quad = Inti.Quadrature(view(msh₁, Γ₁); qorder = 3)
Γ₂ = Inti.external_boundary(Ω₂)
Γ₂_quad = Inti.Quadrature(view(msh₂, Γ₂); qorder = 3)

for op in (
        Inti.Laplace(; dim = 2),
        Inti.Helmholtz(; dim = 2, k = 1.2),
        Inti.Stokes(; dim = 2, μ = 0.5),
    )
    @testset "PDE: $op" begin
        for K in (
                Inti.DoubleLayerKernel(op),
                Inti.SingleLayerKernel(op),
                Inti.AdjointDoubleLayerKernel(op),
                Inti.HyperSingularKernel(op),
            )
            # TODO Stokes has only single and double layer implemented for now
            (K isa Inti.AdjointDoubleLayerKernel && op isa Inti.Stokes) && continue
            (K isa Inti.HyperSingularKernel && op isa Inti.Stokes) && continue
            for Γ_quad in (Γ₁_quad, Γ₂_quad)
                iop = Inti.IntegralOperator(K, Γ₁_quad, Γ_quad)
                iop_fmm = Inti.assemble_fmm(iop; rtol = 1.0e-8)
                x = rand(Inti.default_density_eltype(op), size(iop, 2))
                yapprox = iop_fmm * x
                idx_test = rand(1:size(iop, 1), 10)
                exact = iop[idx_test, :] * x
                @test yapprox[idx_test] ≈ exact rtol = 1.0e-8
            end
        end
    end
end
