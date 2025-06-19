using Inti
using Meshes
using StaticArrays
using GLMakie
using Gmsh
using LinearAlgebra
using NearestNeighbors
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
    crvmsh = Inti.curve_mesh(msh, ψ, θ, 500*round(Int, 1/meshsize))

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
