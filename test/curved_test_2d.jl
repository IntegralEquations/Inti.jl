using Inti
using Meshes
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

    ψ = (t) -> [cos(2*π*t), sin(2*π*t)]
    θ = 6 # smoothness order of curved elements
    crvmsh = Inti.curve_mesh(msh, ψ, θ)

    Γₕ = crvmsh[Γ]
    Ωₕ = crvmsh[Ω]

    qorder = 2
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder);
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder);
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), π, rtol = 1e-7)
    @test isapprox(Inti.integrate(q -> q.coords[1]^4, Ωₕ_quad), π/8, rtol = 1e-5)

    qorder = 5
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder);
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder);
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), π, rtol = 1e-11)
    @test isapprox(Inti.integrate(q -> q.coords[1]^4, Ωₕ_quad), π/8, rtol = 1e-10)

    qorder = 8
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder);
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder);
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), π, rtol = 1e-14)
    @test isapprox(Inti.integrate(q -> q.coords[1]^4, Ωₕ_quad), π/8, rtol = 1e-14)

    Fvol = (x) -> x[2]^2 - 2*x[2]*x[1]^3
    F = (x) -> [x[1]*x[2]^2, x[1]^3*x[2]^2]
    #Fvol = (x) -> 1.0
    #F = (x) -> [1/2*x[1], 1/2*x[2]]
    divvol = Inti.integrate(q -> Fvol(q.coords), Ωₕ_quad)
    divline = Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad)
    @test isapprox(divline, divvol, rtol = 1e-13)
end

@testset begin
    meshsize = 2π / 4/8
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
    entity_parametrizations = Dict{Inti.EntityKey,Function}()
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
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder);
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder);

    Fvol = (x) -> x[2]^2 - 2*x[2]*x[1]^3
    F = (x) -> [x[1]*x[2]^2, x[1]^3*x[2]^2]
    #Fvol = (x) -> 1.0
    #F = (x) -> [1/2*x[1], 1/2*x[2]]
    divvol = Inti.integrate(q -> Fvol(q.coords), Ωₕ_quad)
    divline = Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad)
    @test isapprox(divline, divvol, rtol = 1e-11)
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
    entity_parametrizations = Dict{Inti.EntityKey,Function}()
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
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder);
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder);
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), 6π, rtol = 1e-6)

    qorder = 5
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder);
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder);
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), 6π, rtol = 1e-11)

    qorder = 8
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder);
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder);
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), 6π, rtol = 1e-14)

    # The third circle has larger volume than others, so check that we can pick sub-domains correctly
    Ω_sub = Inti.Domain(collect(keys(Ω))[3])
    Ωₕ_sub = crvmsh[Ω_sub]
    Ωₕ_sub_quad = Inti.Quadrature(Ωₕ_sub; qorder = qorder);
    @test isapprox(Inti.integrate(x -> 1, Ωₕ_sub_quad), 4π, rtol = 1e-14)

    Fvol = (x) -> x[2]^2 - 2*x[2]*x[1]^3
    F = (x) -> [x[1]*x[2]^2, x[1]^3*x[2]^2]
    #Fvol = (x) -> 1.0
    #F = (x) -> [1/2*x[1], 1/2*x[2]]
    divvol = Inti.integrate(q -> Fvol(q.coords), Ωₕ_quad)
    divline = Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad)
    @test isapprox(divline, divvol, rtol = 1e-13)
end
