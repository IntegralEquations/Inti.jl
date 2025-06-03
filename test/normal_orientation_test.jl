using Inti
using Gmsh
using Test
using StaticArrays
using LinearAlgebra

@testset "Two triangles" begin
    # geometry composed of two triangles (entities of dimension 2) that share a common edge.
    #
    #       p3
    #       /\
    #      /  \
    #     /    \
    #    /      \
    #   /        \
    #  p1--------p2
    #   \        /
    #    \      /
    #     \    /
    #      \  /
    #       \/
    #       p4
    #
    gmsh.initialize()
    Inti.clear_entities!()
    gmsh.option.setNumber("General.Verbosity", 2)
    p1 = gmsh.model.occ.addPoint(-1, 0, 0, 1)
    p2 = gmsh.model.occ.addPoint(1, 0, 0, 1)
    p3 = gmsh.model.occ.addPoint(0, 1, 0, 1)
    p4 = gmsh.model.occ.addPoint(0, -1, 0, 1)
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p1)
    l4 = gmsh.model.occ.addLine(p1, p4)
    l5 = gmsh.model.occ.addLine(p4, p2)
    cl1 = gmsh.model.occ.addCurveLoop([l1, l2, l3])
    cl2 = gmsh.model.occ.addCurveLoop([-l1, l4, l5])
    t1 = gmsh.model.occ.addPlaneSurface([cl1])
    t2 = gmsh.model.occ.addPlaneSurface([cl2])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [t1], -1, "t1")
    gmsh.model.addPhysicalGroup(2, [t2], -1, "t2")
    gmsh.model.addPhysicalGroup(1, [l1], -1, "l1")
    gmsh.model.mesh.generate(2)
    msh = Inti.import_mesh(; dim = 2)
    gmsh.finalize()

    # check that both Γ1 and Γ2 are oriented correctly by the divergence theorem
    # ∫ ∇ ⋅ f dΩ = ∫ f ⋅ n dΓ
    Ω1 = Inti.Domain(e -> "t1" ∈ Inti.labels(e), msh)
    Ω2 = Inti.Domain(e -> "t2" ∈ Inti.labels(e), msh)
    Γ1 = Inti.boundary(Ω1)
    Γ2 = Inti.boundary(Ω2)
    f_dot_n = (q) -> begin
        x, n = Inti.coords(q), Inti.normal(q)
        return (x[1] * n[1] + x[2] * n[2]) / 2
    end
    Q1 = Inti.Quadrature(msh[Γ1]; qorder = 3)
    Q2 = Inti.Quadrature(msh[Γ2]; qorder = 3)
    @test Inti.integrate(f_dot_n, Q1) ≈ 1
    @test Inti.integrate(f_dot_n, Q2) ≈ 1
    Q1v = Inti.Quadrature(view(msh, Γ1); qorder = 3)
    Q2v = Inti.Quadrature(view(msh, Γ2); qorder = 3)
    @test Inti.integrate(f_dot_n, Q1v) ≈ 1
    @test Inti.integrate(f_dot_n, Q2v) ≈ 1
end

@testset "Domain with hole" begin
    # geometry composed of a disk of radius two with a hole in the middle of radius one.
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)
    disk = gmsh.model.occ.addDisk(0, 0, 0, 2, 2)
    cut = gmsh.model.occ.addDisk(0.5, 0, 0, 1, 1)
    gmsh.model.occ.cut([(2, disk)], [(2, cut)])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    msh = Inti.import_mesh(; dim = 2)
    gmsh.finalize()

    Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
    Γ = Inti.boundary(Ω)
    f_dot_n = (q) -> begin
        x, n = Inti.coords(q), Inti.normal(q)
        return (x[1] * n[1] + x[2] * n[2]) / 2
    end # zero divergence, so the integral over the domain should be zero
    Q = Inti.Quadrature(msh[Γ]; qorder = 10)
    area = π * 2^2 - π * 1^2
    @test abs(Inti.integrate(f_dot_n, Q) - area) < 1e-4 # mesh is not exact, so approx. errors exist
    Qv = Inti.Quadrature(view(msh, Γ); qorder = 10)
    @test abs(Inti.integrate(f_dot_n, Qv) - area) < 1e-4 # mesh is not exact, so approx. errors exist

    # check that Green's identity holds on this domain
    op = Inti.Laplace(; dim = 2)
    c = 1.3
    xs = SVector(3, -5.2)
    u = (qnode) -> Inti.SingleLayerKernel(op)(qnode, xs) * c
    dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(op)(qnode, xs) * c
    γ₀u = map(u, Q)
    γ₁u = map(dudn, Q)
    γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
    for correction in ((method = :adaptive, maxdist = 0.5, atol = 1e-8), (method = :dim,))
        S, D = Inti.single_double_layer(;
            op,
            target      = Q,
            source      = Q,
            compression = (method = :none,),
            correction,
        )
        er = norm(S * γ₁u - D * γ₀u - 1 / 2 * γ₀u, Inf) / γ₀u_norm
        @test er < 1e-6
    end
end
