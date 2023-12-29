using Test
using Inti
using FMM3D
using Gmsh
using LinearAlgebra
using Random

include("test_utils.jl")

# create a boundary and area meshes and quadrature only once
Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.1)
Γ = Inti.external_boundary(Ω)
Γ_quad = Inti.Quadrature(view(msh, Γ); qorder = 3)

for pde in (Inti.Laplace(; dim = 3), Inti.Helmholtz(; dim = 3, k = 1.2))
    @testset "PDE: $pde" begin
        pde = Inti.Laplace(; dim = 3)
        for K in (Inti.DoubleLayerKernel(pde), Inti.SingleLayerKernel(pde))
            iop = Inti.IntegralOperator(K, Γ_quad, Γ_quad)
            iop_fmm = Inti.assemble_fmm3d(iop; atol = 1e-8)
            x = rand(eltype(iop), size(iop, 2))
            yapprox = iop_fmm * x
            # test on a given index set
            idx_test = rand(1:size(iop, 1), 10)
            exact = iop[idx_test, :] * x
            @test yapprox[idx_test] ≈ exact atol = 1e-7
        end
    end
end
