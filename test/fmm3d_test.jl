using Test
using Inti
using FMM3D
using Gmsh
using LinearAlgebra
using Random
using StaticArrays

include("test_utils.jl")

# create a boundary and area meshes and quadrature only once
Ω₁, msh₁ = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.1)
Ω₂, msh₂ = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.075)
# Test on two meshes to test both sources == targets, and not.
Γ₁ = Inti.external_boundary(Ω₁)
Γ₁_quad = Inti.Quadrature(view(msh₁, Γ₁); qorder = 4)
Γ₂ = Inti.external_boundary(Ω₂)
Γ₂_quad = Inti.Quadrature(view(msh₂, Γ₂); qorder = 4)

for op in (
        Inti.Laplace(; dim = 3),
        Inti.Helmholtz(; dim = 3, k = 1.2),
        Inti.Stokes(; dim = 3, μ = 0.5),
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
                # test on a given index set
                idx_test = rand(1:size(iop, 1), 10)
                exact = iop[idx_test, :] * x
                @test yapprox[idx_test] ≈ exact rtol = 1.0e-7
            end
        end
    end
end

# Test VDIM correction with FMM for Stokes (vector-valued PDE)
@testset "VDIM + FMM for Stokes 3D" begin
    op = Inti.Stokes(; μ = 1.0, dim = 3)

    # Create a coarse mesh for this test
    Ω_coarse, msh_coarse = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.4)
    Γ_coarse = Inti.external_boundary(Ω_coarse)
    Γₕ_quad = Inti.Quadrature(view(msh_coarse, Γ_coarse); qorder = 4)

    # Create volume quadrature
    Ωₕ = view(msh_coarse, Ω_coarse)
    VR_qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(2)
    Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = VR_qorder)
    Ωₕ_quad = Inti.Quadrature(Ωₕ, Dict(E => Q for E in Inti.element_types(Ωₕ)))

    # Build operators with FMM
    S, D = Inti.single_double_layer(;
        op, target = Ωₕ_quad, source = Γₕ_quad,
        compression = (method = :fmm, tol = 1.0e-10),
        correction = (method = :dim, maxdist = 0.5, target_location = :inside)
    )
    V = Inti.volume_potential(;
        op, target = Ωₕ_quad, source = Ωₕ_quad,
        compression = (method = :fmm, tol = 1.0e-10), correction = (method = :none,)
    )
    δV = Inti.vdim_correction(
        op, Ωₕ_quad, Ωₕ_quad, Γₕ_quad, S, D, V;
        green_multiplier = -ones(length(Ωₕ_quad)), interpolation_order = 2
    )

    # Test Green's identity with polynomial solution
    basis = Inti.polynomial_solutions_vdim(op, 2)
    c = SVector(1.0, 2.0, 3.0)
    u_d = [basis[1].solution(q) * c for q in Ωₕ_quad]
    u_b = [basis[1].solution(q) * c for q in Γₕ_quad]
    du_b = [basis[1].neumann_trace(q) * c for q in Γₕ_quad]
    f_d = [basis[1].source(q) * c for q in Ωₕ_quad]

    vref = -u_d - D * u_b + S * du_b
    vapprox = V * f_d + δV * f_d
    @test norm(vref - vapprox, Inf) < 1.0e-10
end
