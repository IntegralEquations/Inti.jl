using Test
using LinearAlgebra
using Gmsh
using Inti
using StaticArrays

@testset "Native mesh generation" begin
    Inti.clear_entities!()
    r = rx = ry = 0.5
    f = let r = r
        (x) -> SVector(r * cos(2π * x[1]), r * sin(2π * x[1]))
    end
    arc1 = Inti.parametric_curve(f, 0, 0.5)
    arc2 = Inti.parametric_curve(f, 0.5, 1)
    Γ = Inti.Domain(arc1, arc2)
    msh = Inti.meshgen(Γ, 100)
    quad = Inti.Quadrature(msh; qorder = 2)
    @test Inti.integrate(x -> 1, quad) ≈ 2 * π * r
    @test Inti.measure(arc1) ≈ π * r
    @test Inti.measure(arc2) ≈ π * r

    # transfinite patch
    Inti.clear_entities!()
    l1 = Inti.parametric_curve(0, 1) do x
        return SVector(x[1], 0.1 * sin(2π * x[1]))
    end
    l2 = Inti.line(SVector(1.0, 0.0), SVector(1.0, 1.0))
    l3 = Inti.line(SVector(1.0, 1.0), SVector(0.0, 1.0))
    l4 = Inti.line(SVector(0.0, 1.0), (0.0, 0.0))
    sq = Inti.transfinite_square(l1, l2, l3, l4)
    Ω = Inti.Domain(sq)
    msh = Inti.meshgen(Ω, Dict(l1 => 10, l2 => 20))
    Γ = Inti.external_boundary(Ω)
    msh[Γ]
    Γ_msh = Inti.meshgen(Γ, 100)
end
