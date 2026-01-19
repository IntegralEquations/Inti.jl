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
rtol = 1.0e-2  # relative tolerance for volume potential tests
meshsize = 0.4
meshorder = 1
bdry_qorder = 5
interpolation_order = 2

"""
    test_volume_potential(op, Ω, Γ, meshsize; interpolation_order=2, bdry_qorder=8)

Test the volume potential operator for a given PDE operator `op` on domain `Ω` with boundary `Γ`.
Verifies the identity: S*γ₁u - D*γ₀u + V*f - σ*γ₀u ≈ 0 for polynomial solutions.
"""
function test_volume_potential(op, Ω, Γ, msh; interpolation_order = 2, bdry_qorder = 8)
    Ωₕ = view(msh, Ω)
    Γₕ = view(msh, Γ)

    # Setup quadrature based on dimension
    dim = Inti.geometric_dimension(Ω)
    if dim == 2
        VR_qorder = Inti.Triangle_VR_interpolation_order_to_quadrature_order(interpolation_order)
        Q = Inti.VioreanuRokhlin(; domain = :triangle, order = VR_qorder)
    elseif dim == 3
        VR_qorder = Inti.Tetrahedron_VR_interpolation_order_to_quadrature_order(interpolation_order)
        Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = VR_qorder)
    else
        error("Unsupported dimension: $dim")
    end
    dict = Dict(E => Q for E in Inti.element_types(Ωₕ))
    Ωₕ_quad = Inti.Quadrature(Ωₕ, dict)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = bdry_qorder)

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

    # Get polynomial solutions for this operator
    basis = Inti.polynomial_solutions_vdim(op, interpolation_order)

    # Test with each polynomial basis function
    errors_uncorrected = Float64[]
    errors_corrected = Float64[]

    for idx in 1:length(basis)
        # For vector-valued operators, multiply by a constant vector
        if Inti.default_density_eltype(op) <: SVector
            N = Inti.ambient_dimension(Ωₕ)
            c = SVector(ntuple(i -> rand(), N)...)
            f = (q) -> basis[idx].source(q) * c
            u = (q) -> basis[idx].solution(q) * c
            t = (q) -> basis[idx].neumann_trace(q) * c
        else
            # Scalar case
            f = (q) -> basis[idx].source(q)
            u = (q) -> basis[idx].solution(q)
            t = (q) -> basis[idx].neumann_trace(q)
        end

        # Evaluate on quadrature nodes
        u_d = map(q -> u(q), Ωₕ_quad)
        u_b = map(q -> u(q), Γₕ_quad)
        du_b = map(q -> t(q), Γₕ_quad)
        f_d = map(q -> f(q), Ωₕ_quad)

        # Compute reference solution: -u - D*u_b + S*du_b
        # This comes from Green's representation: u = S*t - D*u + V*f
        # So V*f = u - S*t + D*u, and we test -V*f = S*t - D*u - u
        vref = -u_d - D_b2d * u_b + S_b2d * du_b

        # Compute uncorrected approximation
        vapprox = V_d2d * f_d
        er = vref - vapprox
        push!(errors_uncorrected, norm(er, Inf))

        # Compute corrected approximation
        vapprox_corr = vapprox + δV_d2d * f_d
        er_corr = vref - vapprox_corr
        push!(errors_corrected, norm(er_corr, Inf))
    end

    return errors_uncorrected, errors_corrected
end

@testset "Volume potential operators 2D" begin
    # Create 2D geometry once
    Inti.clear_entities!()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(meshorder)
    msh = Inti.import_mesh(; dim = 2)
    Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
    gmsh.finalize()
    Γ = Inti.external_boundary(Ω)

    # Define operators to test
    operators = [
        ("2D Laplace", Inti.Laplace(; dim = 2)),
        ("2D Helmholtz", Inti.Helmholtz(; k = 0.7, dim = 2)),
        ("2D Elastostatic", Inti.Elastostatic(; μ = 0.8, λ = 1.3, dim = 2)),
        ("2D Stokes", Inti.Stokes(; μ = 1.0, dim = 2)),
    ]

    for (name, op) in operators
        @testset "$name" begin
            err_uncorr, err_corr = test_volume_potential(op, Ω, Γ, msh; interpolation_order, bdry_qorder)
            @test maximum(err_corr) < rtol
            @test maximum(err_corr) < maximum(err_uncorr)  # Correction should improve accuracy
        end
    end
end

@testset "Volume potential operators 3D" begin
    # Create 3D geometry once
    Inti.clear_entities!()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(meshorder)
    msh = Inti.import_mesh(; dim = 3)
    Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, Inti.entities(msh))
    gmsh.finalize()
    Γ = Inti.external_boundary(Ω)

    # Define operators to test
    operators = [
        ("3D Laplace", Inti.Laplace(; dim = 3)),
        ("3D Helmholtz", Inti.Helmholtz(; k = 1.2, dim = 3)),
        ("3D Elastostatic", Inti.Elastostatic(; μ = 1.1, λ = 0.9, dim = 3)),
        ("3D Stokes", Inti.Stokes(; μ = 1.0, dim = 3)),
    ]

    for (name, op) in operators
        @testset "$name" begin
            err_uncorr, err_corr = test_volume_potential(op, Ω, Γ, msh; interpolation_order, bdry_qorder)
            @test maximum(err_corr) < rtol
            @test maximum(err_corr) < maximum(err_uncorr)  # Correction should improve accuracy
        end
    end
end
