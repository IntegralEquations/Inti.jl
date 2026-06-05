using Test
using Inti
using FMM3D
using Gmsh
using LinearAlgebra
using Random
using OrderedCollections
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
                Inti.GradientSingleLayerKernel(op),
                Inti.GradientDoubleLayerKernel(op),
            )
            # TODO Stokes has only single and double layer implemented for now
            (K isa Inti.AdjointDoubleLayerKernel && op isa Inti.Stokes) && continue
            (K isa Inti.HyperSingularKernel && op isa Inti.Stokes) && continue
            (K isa Inti.GradientSingleLayerKernel && op isa Inti.Stokes) && continue
            (K isa Inti.GradientDoubleLayerKernel && op isa Inti.Stokes) && continue
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
    Ωₕ_quad = Inti.Quadrature(Ωₕ, OrderedDict(E => Q for E in Inti.element_types(Ωₕ)))

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

    vref = u_d + D * u_b - S * du_b
    vapprox = V * f_d + δV * f_d
    @test norm(vref - vapprox, Inf) < 1.0e-10
end


@testset "VDIM grad V Operator for Laplace 3D" begin
    op = Inti.Laplace(; dim = 3)

    # Create a coarse mesh for this test
    Ω_coarse, msh_coarse = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.4)
    Γ_coarse = Inti.external_boundary(Ω_coarse)
    Γₕ_quad = Inti.Quadrature(view(msh_coarse, Γ_coarse); qorder = 4)

    # Create volume quadrature
    Ωₕ = view(msh_coarse, Ω_coarse)
    VR_qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(2)
    Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = VR_qorder)
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

@testset "W operator (FMM vs dense) 3D" begin
    Ω_c, msh_c = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.4)
    Γₕ_quad = Inti.Quadrature(view(msh_c, Inti.external_boundary(Ω_c)); qorder = 4)
    Ωₕ = view(msh_c, Ω_c)
    VR_qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(1)
    Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = VR_qorder)
    Ωₕ_quad = Inti.Quadrature(Ωₕ, OrderedDict(E => Q for E in Inti.element_types(Ωₕ)))
    for (op, Tout) in (
            (Inti.Laplace(; dim = 3), Float64),
            (Inti.Helmholtz(; dim = 3, k = 1.2), ComplexF64),
        )
        @testset "PDE: $op" begin
            cor = (method = :dim, maxdist = 0.5, boundary = Γₕ_quad, interpolation_order = 1)
            Wd = Inti.volume_potential(; op, target = Ωₕ_quad, source = Ωₕ_quad,
                compression = (method = :none,), correction = cor, kernel_variant = :gradient_source)
            Wf = Inti.volume_potential(; op, target = Ωₕ_quad, source = Ωₕ_quad,
                compression = (method = :fmm, tol = 1.0e-12), correction = cor, kernel_variant = :gradient_source)
            g = [rand(SVector{3, Tout}) for _ in 1:length(Ωₕ_quad)]
            yd = Wd * g
            yf = Wf * g
            @test eltype(yd) == Tout
            @test eltype(yf) == Tout
            @test norm(yf - yd, Inf) / norm(yd, Inf) < 1.0e-6
        end
    end
end

@testset "X (∇W) operator (FMM vs dense) Laplace 3D" begin
    op = Inti.Laplace(; dim = 3)
    Ω_c, msh_c = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.4)
    Γₕ_quad = Inti.Quadrature(view(msh_c, Inti.external_boundary(Ω_c)); qorder = 4)
    Ωₕ = view(msh_c, Ω_c)
    VR_qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(1)
    Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = VR_qorder)
    Ωₕ_quad = Inti.Quadrature(Ωₕ, OrderedDict(E => Q for E in Inti.element_types(Ωₕ)))
    cor = (method = :dim, maxdist = 0.5, boundary = Γₕ_quad, interpolation_order = 1)
    Xd = Inti.volume_potential(; op, target = Ωₕ_quad, source = Ωₕ_quad,
        compression = (method = :none,), correction = cor, kernel_variant = :hessian_source)
    Xf = Inti.volume_potential(; op, target = Ωₕ_quad, source = Ωₕ_quad,
        compression = (method = :fmm, tol = 1.0e-12), correction = cor, kernel_variant = :hessian_source)
    g = [rand(SVector{3, Float64}) for _ in 1:length(Ωₕ_quad)]
    yd = Xd * g
    yf = Xf * g
    @test eltype(yd) == SVector{3, Float64}
    @test eltype(yf) == SVector{3, Float64}
    @test norm(yf - yd, Inf) / norm(yd, Inf) < 1.0e-6
end
