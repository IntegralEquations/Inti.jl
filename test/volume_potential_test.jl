#=

Basic tests for the volume potentials based on

S*γ₁u - D*γ₀u + V*f - σ*γ₀u ≈ 0     when    Lu = f

where S, D are the single and double layer boundary integral operators, V is the volume
potential operator, γ₀u and γ₁u are the Dirichlet and Neumann traces of u on the boundary, σ
is 1/2 or -1/2 depending on whether the target point is inside or outside the domain, and L
is the differential operator (Laplace, Helmholtz, Stokes, Elastostatic).

For testing, we use known polynomial solutions of the PDEs to generate u and f = Lu.

=#

using Test
using LinearAlgebra
using Inti
using Random
using StaticArrays
using ForwardDiff
using Gmsh

Random.seed!(1)

## Test parameters
rtol = 1.0e-10  # relative tolerance for volume potential tests
meshsize    = 0.4  # 2D mesh size
meshsize_3d = 0.8  # 3D mesh size — tests check polynomial exactness (not convergence),
                   # so a coarser mesh is valid and keeps matrices small
meshorder = 1
bdry_qorder = 5
interpolation_order = 2

"""
    test_volume_potential(op, Ωₕ_quad, Γₕ_quad, meshsize)

Test the volume potential operator for a given PDE operator `op`.
Verifies the identity: S*γ₁u - D*γ₀u + V*f - σ*γ₀u ≈ 0 for polynomial solutions.
Accepts pre-built quadratures so they can be shared across operators.
"""
function test_volume_potential(op, Ωₕ_quad, Γₕ_quad, meshsize; interpolation_order = 2)
    # Build boundary operators
    S_b2d, D_b2d = Inti.single_double_layer(;
        op,
        target = Ωₕ_quad,
        source = Γₕ_quad,
        compression = (method = :none,),
        correction = (method = :dim, maxdist = 5 * meshsize, target_location = :inside),
    )

    # Build volume potential
    V_d2d = Inti.volume_potential(;
        op,
        target = Ωₕ_quad,
        source = Ωₕ_quad,
        compression = (method = :none,),
        correction = (method = :none,),
    )

    # Build VDIM correction
    δV_d2d = Inti.vdim_correction(
        op, Ωₕ_quad, Ωₕ_quad, Γₕ_quad, S_b2d, D_b2d, V_d2d;
        green_multiplier = -ones(length(Ωₕ_quad)),
        interpolation_order,
        maxdist = Inf
    )

    basis = Inti.polynomial_solutions_vdim(op, interpolation_order)

    errors_uncorrected = Float64[]
    errors_corrected = Float64[]

    for idx in 1:length(basis)
        if Inti.default_density_eltype(op) <: SVector
            N = Inti.ambient_dimension(Ωₕ_quad)
            c = SVector(ntuple(i -> rand(), N)...)
            f = (q) -> basis[idx].source(q) * c
            u = (q) -> basis[idx].solution(q) * c
            t = (q) -> basis[idx].neumann_trace(q) * c
        else
            f = (q) -> basis[idx].source(q)
            u = (q) -> basis[idx].solution(q)
            t = (q) -> basis[idx].neumann_trace(q)
        end

        u_d  = map(q -> u(q), Ωₕ_quad)
        u_b  = map(q -> u(q), Γₕ_quad)
        du_b = map(q -> t(q), Γₕ_quad)
        f_d  = map(q -> f(q), Ωₕ_quad)

        vref        = u_d + D_b2d * u_b - S_b2d * du_b
        vapprox     = V_d2d * f_d
        vapprox_corr = vapprox + δV_d2d * f_d

        push!(errors_uncorrected, norm(vref - vapprox, Inf))
        push!(errors_corrected, norm(vref - vapprox_corr, Inf))
    end

    return errors_uncorrected, errors_corrected
end

"""
    test_gradient_volume_potential(op, Ωₕ_quad, Γₕ_quad, meshsize)

Test the gradient volume potential identity.
Accepts pre-built quadratures so they can be shared across operators.
"""
function test_gradient_volume_potential(op, Ωₕ_quad, Γₕ_quad, meshsize; interpolation_order = 2)
    W, GDL = Inti.single_double_layer(;
        op,
        target = Ωₕ_quad,
        source = Γₕ_quad,
        compression = (method = :none,),
        correction = (method = :dim, maxdist = 5 * meshsize, target_location = :inside),
        kernel_variant = :gradient,
    )
    V_grad = Inti.volume_potential(;
        op,
        target = Ωₕ_quad,
        source = Ωₕ_quad,
        compression = (method = :none,),
        correction = (method = :none,),
        kernel_variant = :gradient,
    )
    δV_grad = Inti.vdim_correction(
        op, Ωₕ_quad, Ωₕ_quad, Γₕ_quad, W, GDL, V_grad;
        green_multiplier = -ones(length(Ωₕ_quad)),
        interpolation_order,
        kernel_variant = :gradient,
    )

    basis = Inti.polynomial_solutions_vdim(op, interpolation_order)
    errors_corrected = Float64[]

    for idx in 1:length(basis)
        ∇u_d = [basis[idx].gradient_solution(q) for q in Ωₕ_quad]
        u_b  = [basis[idx].solution(q) for q in Γₕ_quad]
        du_b = [basis[idx].neumann_trace(q) for q in Γₕ_quad]
        f_d  = [basis[idx].source(q) for q in Ωₕ_quad]
        vref    = ∇u_d + GDL * u_b - W * du_b
        vapprox = V_grad * f_d + δV_grad * f_d
        push!(errors_corrected, norm(vref - vapprox, Inf))
    end
    return errors_corrected
end

## Helper to build the shared volume + boundary quadratures for a given mesh/domain.
function build_quadratures(Ωₕ, Γₕ, dim; interpolation_order, bdry_qorder)
    if dim == 2
        VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(interpolation_order)
        Q = Inti.VioreanuRokhlin(; domain = :triangle, order = VR_qorder)
    else
        VR_qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(interpolation_order)
        Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = VR_qorder)
    end
    Ωₕ_quad = Inti.Quadrature(Ωₕ, Q)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = bdry_qorder)
    return Ωₕ_quad, Γₕ_quad
end

# ── 2D tests ─────────────────────────────────────────────────────────────────────────────────
# Build the 2D disk mesh ONCE and reuse it across all 2D volume and gradient tests.
Inti.clear_entities!()
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(meshorder)
msh_2d = Inti.import_mesh(; dim = 2)
Ω_2d = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh_2d))
gmsh.finalize()
Γ_2d = Inti.external_boundary(Ω_2d)
Ωₕ_2d = view(msh_2d, Ω_2d)
Γₕ_2d = view(msh_2d, Γ_2d)
Ωₕ_quad_2d, Γₕ_quad_2d = build_quadratures(Ωₕ_2d, Γₕ_2d, 2; interpolation_order, bdry_qorder)

@testset "Volume potential operators 2D" begin
    for (name, op) in [
        ("2D Laplace",      Inti.Laplace(; dim = 2)),
        ("2D Helmholtz",    Inti.Helmholtz(; k = 0.7, dim = 2)),
        ("2D Elastostatic", Inti.Elastostatic(; μ = 0.8, λ = 1.3, dim = 2)),
        ("2D Stokes",       Inti.Stokes(; μ = 1.0, dim = 2)),
    ]
        @testset "$name" begin
            err_uncorr, err_corr = test_volume_potential(op, Ωₕ_quad_2d, Γₕ_quad_2d, meshsize; interpolation_order)
            @test maximum(err_corr) < rtol
            @test maximum(err_corr) < maximum(err_uncorr)
        end
    end
end

@testset "Gradient volume potential 2D" begin
    for (name, op) in [
        ("Laplace",      Inti.Laplace(; dim = 2)),
        ("Helmholtz",    Inti.Helmholtz(; k = 0.7, dim = 2)),
        ("Elastostatic", Inti.Elastostatic(; μ = 0.8, λ = 1.3, dim = 2)),
        ("Stokes",       Inti.Stokes(; μ = 1.2, dim = 2)),
    ]
        @testset "Gradient volume potential 2D $name" begin
            err_corr = test_gradient_volume_potential(op, Ωₕ_quad_2d, Γₕ_quad_2d, meshsize; interpolation_order)
            @test maximum(err_corr) < rtol
        end
    end
end

# ── 3D tests ─────────────────────────────────────────────────────────────────────────────────
# Build the 3D sphere mesh ONCE and reuse it across all 3D volume and gradient tests.
Inti.clear_entities!()
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize_3d)
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize_3d)
gmsh.model.occ.addSphere(0, 0, 0, 1)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.model.mesh.setOrder(meshorder)
msh_3d = Inti.import_mesh(; dim = 3)
Ω_3d = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, Inti.entities(msh_3d))
gmsh.finalize()
Γ_3d = Inti.external_boundary(Ω_3d)
Ωₕ_3d = view(msh_3d, Ω_3d)
Γₕ_3d = view(msh_3d, Γ_3d)
Ωₕ_quad_3d, Γₕ_quad_3d = build_quadratures(Ωₕ_3d, Γₕ_3d, 3; interpolation_order, bdry_qorder)

@testset "Volume potential operators 3D" begin
    for (name, op) in [
        ("3D Laplace",      Inti.Laplace(; dim = 3)),
        ("3D Helmholtz",    Inti.Helmholtz(; k = 1.2, dim = 3)),
        ("3D Elastostatic", Inti.Elastostatic(; μ = 1.1, λ = 0.9, dim = 3)),
        ("3D Stokes",       Inti.Stokes(; μ = 1.0, dim = 3)),
    ]
        @testset "$name" begin
            err_uncorr, err_corr = test_volume_potential(op, Ωₕ_quad_3d, Γₕ_quad_3d, meshsize_3d; interpolation_order)
            @test maximum(err_corr) < rtol
            @test maximum(err_corr) < maximum(err_uncorr)
        end
    end
end

@testset "Gradient volume potential 3D" begin
    for (name, op) in [
        ("Laplace",      Inti.Laplace(; dim = 3)),
        ("Helmholtz",    Inti.Helmholtz(; k = 1.2, dim = 3)),
        ("Elastostatic", Inti.Elastostatic(; μ = 1.1, λ = 0.9, dim = 3)),
        ("Stokes",       Inti.Stokes(; μ = 1.2, dim = 3)),
    ]
        @testset "Gradient volume potential 3D $name" begin
            err_corr = test_gradient_volume_potential(op, Ωₕ_quad_3d, Γₕ_quad_3d, meshsize_3d; interpolation_order)
            @test maximum(err_corr) < rtol
        end
    end
end
