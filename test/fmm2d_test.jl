using Test
using Inti
using FMM2D
using Gmsh
using LinearAlgebra
using OrderedCollections
using Random
using StaticArrays

include("test_utils.jl")

# create a boundary and area meshes and quadrature only once
Ω₁, msh₁ = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.1)
Ω₂, msh₂ = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.05)

# Test on two meshes to test both sources == targets, and not.
Γ₁ = Inti.external_boundary(Ω₁)
Γ₁_quad = Inti.Quadrature(view(msh₁, Γ₁); qorder = 3)
Γ₂ = Inti.external_boundary(Ω₂)
Γ₂_quad = Inti.Quadrature(view(msh₂, Γ₂); qorder = 3)

for op in (Inti.Laplace(; dim = 2), Inti.Helmholtz(; dim = 2, k = 1.2))
    @testset "PDE: $op" begin
    op = Inti.Laplace(; dim = 2)
        for K in (
                Inti.DoubleLayerKernel(op),
                Inti.SingleLayerKernel(op),
                Inti.AdjointDoubleLayerKernel(op),
                Inti.HyperSingularKernel(op),
                Inti.GradientSingleLayerKernel(op),
                Inti.GradientDoubleLayerKernel(op),
            )
            K = Inti.GradientDoubleLayerKernel(op)
            for Γ_quad in (Γ₁_quad, Γ₂_quad)
                iop = Inti.IntegralOperator(K, Γ₁_quad, Γ_quad)
                iop_fmm = Inti.assemble_fmm(iop; rtol = 1.0e-8)
                x = rand(Inti.default_density_eltype(op), size(iop, 2))
                yapprox = iop_fmm * x
                # test on a given index set
                idx_test = rand(1:size(iop, 1), 10)
                exact = iop[idx_test, :] * x
                # The discrepancy in tolerance for assemble_fmm and the test is because
                # the library is tuned for error in potential but not in gradient
                @test yapprox[idx_test] ≈ exact rtol = 5.0e-6
            end
        end
    end
end

@testset "VDIM grad V Operator for Laplace 2D" begin
    op = Inti.Laplace(; dim = 2)

    # Create a coarse mesh for this test
    Ω_coarse, msh_coarse = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.4)
    Γ_coarse = Inti.external_boundary(Ω_coarse)
    Γₕ_quad = Inti.Quadrature(view(msh_coarse, Γ_coarse); qorder = 4)

    # Create volume quadrature
    Ωₕ = view(msh_coarse, Ω_coarse)
    VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(2)
    Q = Inti.VioreanuRokhlin(; domain = :triangle, order = VR_qorder)
    Ωₕ_quad = Inti.Quadrature(Ωₕ, OrderedDict(E => Q for E in Inti.element_types(Ωₕ)))

    # Build gradient operators with FMM
    S, D = Inti.single_double_layer(;
        op, target = Ωₕ_quad, source = Γₕ_quad,
        compression = (method = :fmm, tol = 1.0e-10),
        correction = (method = :dim, maxdist = 0.5, target_location = :inside),
        kernel_variant = :gradient
    )
    V = Inti.volume_potential(;
        op, target = Ωₕ_quad, source = Ωₕ_quad,
        compression = (method = :fmm, tol = 1.0e-10), correction = (method = :none,),
        kernel_variant = :gradient
    )
    δV = Inti.vdim_correction(
        op, Ωₕ_quad, Ωₕ_quad, Γₕ_quad, S, D, V;
        green_multiplier = -ones(length(Ωₕ_quad)), interpolation_order = 2,
        kernel_variant = :gradient
    )

    # Test Green's identity with polynomial solution
    basis = Inti.polynomial_solutions_vdim(op, 2)

    # Use polynomial 2: x
    u_d = [basis[2].gradient_solution(q) for q in Ωₕ_quad]
    u_b = [basis[2].solution(q) for q in Γₕ_quad]
    du_b = [basis[2].neumann_trace(q) for q in Γₕ_quad]
    f_d = [basis[2].source(q) for q in Ωₕ_quad]

    vref = u_d + D * u_b - S * du_b
    vapprox = V * f_d + δV * f_d
    @test norm(vref - vapprox, Inf) < 1.0e-10
end

@testset "W operator (FMM vs dense) 2D" begin
    Ω_c, msh_c = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.2)
    Γₕ_quad = Inti.Quadrature(view(msh_c, Inti.external_boundary(Ω_c)); qorder = 4)
    Ωₕ = view(msh_c, Ω_c)
    VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(2)
    Q = Inti.VioreanuRokhlin(; domain = :triangle, order = VR_qorder)
    Ωₕ_quad = Inti.Quadrature(Ωₕ, OrderedDict(E => Q for E in Inti.element_types(Ωₕ)))
    for (op, Tout) in (
            (Inti.Laplace(; dim = 2), Float64),
            (Inti.Helmholtz(; dim = 2, k = 1.2), ComplexF64),
        )
        @testset "PDE: $op" begin
            cor = (method = :dim, maxdist = 0.5, boundary = Γₕ_quad, interpolation_order = 2)
            Wd = Inti.volume_potential(; op, target = Ωₕ_quad, source = Ωₕ_quad,
                compression = (method = :none,), correction = cor, kernel_variant = :gradient_source)
            Wf = Inti.volume_potential(; op, target = Ωₕ_quad, source = Ωₕ_quad,
                compression = (method = :fmm, tol = 1.0e-12), correction = cor, kernel_variant = :gradient_source)
            g = [rand(SVector{2, Tout}) for _ in 1:length(Ωₕ_quad)]
            yd = Wd * g
            yf = Wf * g
            @test eltype(yd) == Tout
            @test eltype(yf) == Tout
            @test norm(yf - yd, Inf) / norm(yd, Inf) < 1.0e-6
        end
    end
end

@testset "X charge→Hessian volume op (FMM vs dense) Laplace 2D" begin
    # Validates the charge→Hessian FMM realization of the `HessianKernel` (the
    # scalar-density → `SMatrix` Hessian single-layer volume operator used by the X = ∇W
    # VDIM correction) against the dense Hessian operator.
    op = Inti.Laplace(; dim = 2)
    Ω_c, msh_c = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.2)
    Ωₕ = view(msh_c, Ω_c)
    VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(2)
    Q = Inti.VioreanuRokhlin(; domain = :triangle, order = VR_qorder)
    Ωₕ_quad = Inti.Quadrature(Ωₕ, OrderedDict(E => Q for E in Inti.element_types(Ωₕ)))
    Vh = Inti.IntegralOperator(Inti.HessianKernel(op, :charge), Ωₕ_quad, Ωₕ_quad)
    m, n = size(Vh)
    ρ = rand(n)
    # dense reference: Mᵢ = Σₖ Vh[i,k] ρₖ  (an SMatrix per target)
    Mref = [sum(Vh[i, k] * ρ[k] for k in 1:n) for i in 1:m]
    Vfmm = Inti.assemble_fmm(Vh; rtol = 1.0e-12)
    @test eltype(Vfmm) == SMatrix{2, 2, Float64, 4}
    Mfmm = Vector{SMatrix{2, 2, Float64, 4}}(undef, m)
    mul!(Mfmm, Vfmm, ρ)
    @test norm(Mref - Mfmm, Inf) / norm(Mref, Inf) < 1.0e-8
end

@testset "X (∇W) operator (FMM vs dense) 2D" begin
    Ω_c, msh_c = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = 0.2)
    Γₕ_quad = Inti.Quadrature(view(msh_c, Inti.external_boundary(Ω_c)); qorder = 4)
    Ωₕ = view(msh_c, Ω_c)
    VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(2)
    Q = Inti.VioreanuRokhlin(; domain = :triangle, order = VR_qorder)
    Ωₕ_quad = Inti.Quadrature(Ωₕ, OrderedDict(E => Q for E in Inti.element_types(Ωₕ)))
    cor = (method = :dim, maxdist = 0.5, boundary = Γₕ_quad, interpolation_order = 2)
    for (op, Tout) in (
            (Inti.Laplace(; dim = 2), Float64),
            (Inti.Helmholtz(; dim = 2, k = 1.2), ComplexF64),
        )
        @testset "PDE: $op" begin
            Xd = Inti.volume_potential(; op, target = Ωₕ_quad, source = Ωₕ_quad,
                compression = (method = :none,), correction = cor, kernel_variant = :hessian)
            Xf = Inti.volume_potential(; op, target = Ωₕ_quad, source = Ωₕ_quad,
                compression = (method = :fmm, tol = 1.0e-12), correction = cor, kernel_variant = :hessian)
            g = [rand(SVector{2, Tout}) for _ in 1:length(Ωₕ_quad)]
            yd = Xd * g
            yf = Xf * g
            @test eltype(yd) == SVector{2, Tout}
            @test eltype(yf) == SVector{2, Tout}
            @test norm(yf - yd, Inf) / norm(yd, Inf) < 1.0e-6
        end
    end
end