using Inti
using StaticArrays
using GLMakie
using Gmsh
using LinearAlgebra
using ForwardDiff
using Test

@testset begin
    function domain_and_mesh(; meshsize, meshorder = 1)
        Inti.clear_entities!()
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(meshorder)
        msh = Inti.import_mesh(; dim = 2)
        Ω = Inti.Domain(Inti.entities(msh)) do ent
            return Inti.geometric_dimension(ent) == 2
        end
        gmsh.finalize()
        return Ω, msh
    end

    meshsize = 0.1

    tmesh = @elapsed begin
        Ω, msh = domain_and_mesh(; meshsize)
    end
    @info "Mesh generation time: $tmesh"

    Γ = Inti.external_boundary(Ω)
    Ωₕ = view(msh, Ω)
    Γₕ = view(msh, Γ)

    ψ = (t) -> [cos(2 * π * t), sin(2 * π * t)]
    θ = 6 # smoothness order of curved elements
    crvmsh = Inti.curve_mesh(msh, ψ, θ)

    Γₕ = crvmsh[Γ]
    Ωₕ = crvmsh[Ω]

    qorder = 2
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), π, rtol = 1.0e-7)
    @test isapprox(Inti.integrate(q -> q.coords[1]^4, Ωₕ_quad), π / 8, rtol = 1.0e-5)

    qorder = 5
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), π, rtol = 1.0e-11)
    @test isapprox(Inti.integrate(q -> q.coords[1]^4, Ωₕ_quad), π / 8, rtol = 1.0e-10)

    qorder = 8
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), π, rtol = 1.0e-14)
    @test isapprox(Inti.integrate(q -> q.coords[1]^4, Ωₕ_quad), π / 8, rtol = 1.0e-14)

    Fvol = (x) -> x[2]^2 - 2 * x[2] * x[1]^3
    F = (x) -> [x[1] * x[2]^2, x[1]^3 * x[2]^2]
    #Fvol = (x) -> 1.0
    #F = (x) -> [1/2*x[1], 1/2*x[2]]
    divvol = Inti.integrate(q -> Fvol(q.coords), Ωₕ_quad)
    divline = Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad)
    @test isapprox(divline, divvol, rtol = 1.0e-13)
end

@testset begin
    meshsize = 2π / 4 / 8
    Inti.clear_entities!()

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)

    # Two kites
    kite = Inti.gmsh_curve(0, 1; meshsize) do s
        return SVector(0.25, 0.0) +
            SVector(cos(2π * s) + 0.65 * cos(4π * s[1]) - 0.65, 1.5 * sin(2π * s))
    end
    cl = gmsh.model.occ.addCurveLoop([kite])
    surf = gmsh.model.occ.addPlaneSurface([cl])
    kite_trans = Inti.gmsh_curve(0, 1; meshsize) do s
        return SVector(4.5, 0.0) +
            SVector(cos(2π * s) + 0.65 * cos(4π * s[1]) - 0.65, 1.5 * sin(2π * s))
    end
    cl_trans = gmsh.model.occ.addCurveLoop([kite_trans])
    surf_trans = gmsh.model.occ.addPlaneSurface([cl_trans])
    gmsh.model.occ.synchronize()

    # Add tags for stable identification of the entities
    gmsh.model.addPhysicalGroup(2, [surf], -1, "c1")
    gmsh.model.addPhysicalGroup(2, [surf_trans], -1, "c2")

    gmsh.model.mesh.generate(2)
    msh = Inti.import_mesh(; dim = 2)
    Ω = Inti.Domain(Inti.entities(msh)) do ent
        return Inti.geometric_dimension(ent) == 2
    end
    gmsh.finalize()

    Γ = Inti.external_boundary(Ω)
    Ωₕ = view(msh, Ω)
    Γₕ = view(msh, Γ)

    # Two kites
    ψ₁ = (t) -> [0.25 + cos(2π * t) + 0.65 * cos(4π * t) - 0.65, 1.5 * sin(2π * t)]
    ψ₂ = (t) -> [4.5 + cos(2π * t) + 0.65 * cos(4π * t) - 0.65, 1.5 * sin(2π * t)]
    entity_parametrizations = Dict{Inti.EntityKey, Function}()
    for e in Inti.entities(Ω)
        l = Inti.labels(e)
        if "c1" in l
            entity_parametrizations[e] = ψ₁
        elseif "c2" in l
            entity_parametrizations[e] = ψ₂
        elseif "c3" in l
            entity_parametrizations[e] = ψ₃
        end
    end
    θ = 6 # smoothness order of curved elements
    crvmsh = Inti.curve_mesh(msh, entity_parametrizations, θ)

    Γₕ = crvmsh[Γ]
    Ωₕ = crvmsh[Ω]

    qorder = 8
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)

    Fvol = (x) -> x[2]^2 - 2 * x[2] * x[1]^3
    F = (x) -> [x[1] * x[2]^2, x[1]^3 * x[2]^2]
    #Fvol = (x) -> 1.0
    #F = (x) -> [1/2*x[1], 1/2*x[2]]
    divvol = Inti.integrate(q -> Fvol(q.coords), Ωₕ_quad)
    divline = Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad)
    @test isapprox(divline, divvol, rtol = 1.0e-11)
end

@testset begin
    meshsize = 0.075
    Inti.clear_entities!()

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)

    # Three circles
    c1 = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    c2 = gmsh.model.occ.addDisk(0, 3.0, 0, 1, 1)
    c3 = gmsh.model.occ.addDisk(0, 8.0, 0, 2, 2)
    gmsh.model.occ.synchronize()

    # Add tags for stable identification of the entities
    gmsh.model.addPhysicalGroup(2, [c1], -1, "c1")
    gmsh.model.addPhysicalGroup(2, [c2], -1, "c2")
    gmsh.model.addPhysicalGroup(2, [c3], -1, "c3")

    gmsh.model.mesh.generate(2)
    msh = Inti.import_mesh(; dim = 2)
    Ω = Inti.Domain(Inti.entities(msh)) do ent
        return Inti.geometric_dimension(ent) == 2
    end
    gmsh.finalize()

    Γ = Inti.external_boundary(Ω)
    Ωₕ = view(msh, Ω)
    Γₕ = view(msh, Γ)

    # Three circles
    ψ₁ = (t) -> [cos(2 * π * t), sin(2 * π * t)]
    ψ₂ = (t) -> [cos(2 * π * t), 3.0 + sin(2 * π * t)]
    ψ₃ = (t) -> [2 * cos(2 * π * t), 8.0 + 2 * sin(2 * π * t)]
    entity_parametrizations = Dict{Inti.EntityKey, Function}()
    for e in Inti.entities(Ω)
        l = Inti.labels(e)
        if "c1" in l
            entity_parametrizations[e] = ψ₁
        elseif "c2" in l
            entity_parametrizations[e] = ψ₂
        elseif "c3" in l
            entity_parametrizations[e] = ψ₃
        end
    end

    θ = 6 # smoothness order of curved elements
    crvmsh = Inti.curve_mesh(msh, entity_parametrizations, θ)

    Γₕ = crvmsh[Γ]
    Ωₕ = crvmsh[Ω]

    qorder = 2
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), 6π, rtol = 1.0e-6)

    qorder = 5
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), 6π, rtol = 1.0e-11)

    qorder = 8
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), 6π, rtol = 1.0e-14)

    # The third circle has larger volume than others, so check that we can pick sub-domains correctly
    Ω_sub = Inti.Domain(collect(keys(Ω))[3])
    Ωₕ_sub = crvmsh[Ω_sub]
    Ωₕ_sub_quad = Inti.Quadrature(Ωₕ_sub; qorder = qorder)
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_sub_quad), 4π, rtol = 1.0e-14)

    Fvol = (x) -> x[2]^2 - 2 * x[2] * x[1]^3
    F = (x) -> [x[1] * x[2]^2, x[1]^3 * x[2]^2]
    #Fvol = (x) -> 1.0
    #F = (x) -> [1/2*x[1], 1/2*x[2]]
    divvol = Inti.integrate(q -> Fvol(q.coords), Ωₕ_quad)
    divline = Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad)
    @test isapprox(divline, divvol, rtol = 1.0e-13)
end

# Atlases are the basic input that `curve_mesh` accepts in both 2D and 3D, but
# test bare functions as inputs too
@testset "2D charts and atlases" begin
    function disk_mesh(; meshsize = 0.1)
        Inti.clear_entities!()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        msh = Inti.import_mesh(; dim = 2)
        Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
        gmsh.finalize()
        return Ω, msh
    end

    Ω, msh = disk_mesh()
    ψ = (t) -> SVector(cos(2π * t), sin(2π * t))
    θ = 6

    @testset "a bare ψ is exactly the one-chart atlas" begin
        # the 3D analogue of this check lives in `curved_multipatch_test.jl`
        chart = Inti.ParametricChart(
            α -> ψ(α[1]),
            SVector(0.0),
            SVector(1.0);
            period = SVector(1.0),
        )
        q_fun = Inti.Quadrature(Inti.curve_mesh(msh, ψ, θ)[Ω]; qorder = 5)
        q_atlas = Inti.Quadrature(Inti.curve_mesh(msh, [chart], θ)[Ω]; qorder = 5)
        @test length(q_fun) == length(q_atlas)
        @test maximum(norm(a.coords - b.coords) for (a, b) in zip(q_fun, q_atlas)) < 1.0e-14
        @test maximum(abs(a.weight - b.weight) for (a, b) in zip(q_fun, q_atlas)) < 1.0e-14
    end

    @testset "a bare ψ may take a scalar or an SVector{1}" begin
        q1 = Inti.Quadrature(Inti.curve_mesh(msh, ψ, θ)[Ω]; qorder = 5)
        q2 = Inti.Quadrature(
            Inti.curve_mesh(msh, s -> SVector(cos(2π * s[1]), sin(2π * s[1])), θ)[Ω];
            qorder = 5,
        )
        @test maximum(norm(a.coords - b.coords) for (a, b) in zip(q1, q2)) < 1.0e-14
    end

    @testset "several charts per entity" begin
        # two overlapping, non-periodic charts covering the circle between them
        atlas = [
            Inti.ParametricChart(α -> ψ(α[1]), SVector(-0.1), SVector(0.6)),
            Inti.ParametricChart(α -> ψ(α[1]), SVector(0.4), SVector(1.1)),
        ]
        crv = Inti.curve_mesh(msh, atlas, θ)
        quad = Inti.Quadrature(crv[Ω]; qorder = 5)
        @test isapprox(Inti.integrate(x -> 1, quad), π, rtol = 1.0e-11)
    end

    @testset "every `Dict` shape normalizes to the same mesh" begin
        ent = only(Inti.entities(Ω))
        chart = Inti.ParametricChart(
            α -> ψ(α[1]),
            SVector(0.0),
            SVector(1.0);
            period = SVector(1.0),
        )
        ref = Inti.Quadrature(Inti.curve_mesh(msh, ψ, θ)[Ω]; qorder = 5)
        for d in (
                Dict{Inti.EntityKey, Function}(ent => ψ),   # the historical form
                Dict(ent => ψ),                             # concrete closure type
                Dict(ent => [chart]),                       # entity => atlas
                Dict{Inti.EntityKey, Any}(ent => [chart]),  # heterogeneous values
            )
            q = Inti.Quadrature(Inti.curve_mesh(msh, d, θ)[Ω]; qorder = 5)
            @test maximum(norm(a.coords - b.coords) for (a, b) in zip(ref, q)) < 1.0e-14
        end
    end

    @testset "keywords are accepted or rejected by name, never by MethodError" begin
        # `face_map` and `projection_iterations` describe how a curved face is
        # laid over a surface, which is not necessary in 2D 
        @test_throws ErrorException Inti.curve_mesh(msh, ψ, θ; face_map = :projection)
        @test_throws ErrorException Inti.curve_mesh(msh, ψ, θ; projection_iterations = 8)
        # these two are as meaningful in 2D as in 3D
        @test Inti.curve_mesh(msh, ψ, θ; chart_atol = 0.05) isa Inti.Mesh
        @test Inti.curve_mesh(msh, ψ, θ; check_jacobian = :none) isa Inti.Mesh
    end

    @testset "a chart of the wrong parameter dimension is rejected" begin
        # keep a mismatched chart from falling back onto the normalizing wrapper
        # and recursing forever
        surface = Inti.ParametricChart(
            α -> SVector(α[1], α[2], 0.0),
            SVector(0.0, 0.0),
            SVector(1.0, 1.0),
        )
        @test_throws ErrorException Inti.curve_mesh(msh, [surface], θ)
    end
end

# An atlas must satisfy the property that every boundary edge lies inside a
# single chart. This can be achieved with overlapping charts, or abutting
# charts. The latter case enables curved volume meshing of a piece-wise smooth
# curve. The latter in turn needs a volume entity to be allowed several boundary
# entities.
@testset "2D charts that abut instead of overlap" begin
    ms = 0.1
    ψ = (t) -> SVector(cos(2π * t), sin(2π * t))
    Inti.clear_entities!()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", ms)
    gmsh.option.setNumber("Mesh.MeshSizeMin", ms)
    # one curve entity per chart, so the mesher puts a vertex at each seam
    top = Inti.gmsh_curve(0.0, 0.5; meshsize = ms) do s
        return ψ(s)
    end
    bot = Inti.gmsh_curve(0.5, 1.0; meshsize = ms) do s
        return ψ(s)
    end
    cl = gmsh.model.occ.addCurveLoop([top, bot])
    gmsh.model.occ.addPlaneSurface([cl])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    msh = Inti.import_mesh(; dim = 2)
    Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, Inti.entities(msh))
    gmsh.finalize()
    Γ = Inti.external_boundary(Ω)
    @test length(Inti.entities(Γ)) == 2

    chart(lc, hc) = Inti.ParametricChart(α -> ψ(α[1]), SVector(lc), SVector(hc))
    # the two charts share only their endpoints: no overlap whatsoever
    crv = Inti.curve_mesh(msh, [chart(0.0, 0.5), chart(0.5, 1.0)], 6;
                          check_jacobian = :error)

    Ωq = Inti.Quadrature(crv[Ω]; qorder = 8)
    Γq = Inti.Quadrature(crv[Γ]; qorder = 8)
    @test isapprox(Inti.integrate(x -> 1, Ωq), π, rtol = 1.0e-14)
    F = (x) -> [x[1] / 2, x[2] / 2]
    @test isapprox(Inti.integrate(q -> dot(F(q.coords), q.normal), Γq), π, rtol = 1.0e-14)
    @test maximum(q -> abs(norm(q.coords) - 1), Γq) < 1.0e-14
    @test all(q -> dot(q.normal, q.coords) > 0, Γq)
    # each boundary entity must have received its own curved elements
    for e in Inti.entities(Γ)
        @test sum(length, values(crv.ent2etags[e]); init = 0) > 0
    end

    # a seam falling mid-edge still cannot work, and must say so
    off = 0.5 + 0.3 / length(Γq)
    @test_throws ErrorException Inti.curve_mesh(msh, [chart(0.0, off), chart(off, 1.0)], 6)
end
