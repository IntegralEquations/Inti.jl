using Test
using Inti
using IFGF
using Gmsh
using LinearAlgebra
using Random

include("test_utils.jl")

# TODO Test with k = 3.14f0 and mesh with Float32 elements

# N = 2
# create a boundary and area meshes and quadrature only once
Ω₁, msh₁ = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.1)
Ω₂, msh₂ = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.05)

# Test on two meshes to test both sources == targets, and not.
Γ₁ = Inti.external_boundary(Ω₁)
Γ₁_quad = Inti.Quadrature(view(msh₁, Γ₁); qorder = 3)
Γ₂ = Inti.external_boundary(Ω₂)
Γ₂_quad = Inti.Quadrature(view(msh₂, Γ₂); qorder = 3)

for pde in (Inti.Laplace(; dim = 2), Inti.Helmholtz(; dim = 2, k = 3.14e0))
    @testset "PDE: $pde" begin
        for K in (
            Inti.DoubleLayerKernel(pde),
            Inti.SingleLayerKernel(pde),
            Inti.AdjointDoubleLayerKernel(pde),
            Inti.HyperSingularKernel(pde),
        )
            for Γ_quad in (Γ₁_quad, Γ₂_quad)
                iop = Inti.IntegralOperator(K, Γ₁_quad, Γ_quad)
                iop_ifgf = Inti.assemble_ifgf(iop; atol = 1e-8)
                x = rand(eltype(iop), size(iop, 2))
                yapprox = iop_ifgf * x
                # test on a given index set
                idx_test = rand(1:size(iop, 1), 10)
                exact = iop[idx_test, :] * x
                @test yapprox[idx_test] ≈ exact atol = 1e-7
            end
        end
    end
end

# N = 3
# create a boundary and area meshes and quadrature only once
Ω₁, msh₁ = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.1)
Ω₂, msh₂ = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.075)
# Test on two meshes to test both sources == targets, and not.
Γ₁ = Inti.external_boundary(Ω₁)
Γ₁_quad = Inti.Quadrature(view(msh₁, Γ₁); qorder = 3)
Γ₂ = Inti.external_boundary(Ω₂)
Γ₂_quad = Inti.Quadrature(view(msh₂, Γ₂); qorder = 3)

for pde in (Inti.Laplace(; dim = 3), Inti.Helmholtz(; dim = 3, k = 3.14e0))
    @testset "PDE: $pde" begin
        for K in (
            Inti.DoubleLayerKernel(pde),
            Inti.SingleLayerKernel(pde),
            Inti.AdjointDoubleLayerKernel(pde),
            Inti.HyperSingularKernel(pde),
        )
            for Γ_quad in (Γ₁_quad, Γ₂_quad)
                iop = Inti.IntegralOperator(K, Γ₁_quad, Γ_quad)
                iop_ifgf = Inti.assemble_ifgf(iop; atol = 1e-8)
                x = rand(eltype(iop), size(iop, 2))
                yapprox = iop_ifgf * x
                # test on a given index set
                idx_test = rand(1:size(iop, 1), 10)
                exact = iop[idx_test, :] * x
                @test yapprox[idx_test] ≈ exact atol = 1e-7
            end
        end
    end
end
