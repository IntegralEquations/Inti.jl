using Test
using StaticArrays
using Inti
using LinearAlgebra

@testset "Lagrange elements" begin
    @testset "LagrangeLine" begin
        d = Inti.ReferenceLine()
        f = x -> x[1]^2
        x̂ = Inti.reference_nodes(Inti.LagrangeLine{3})
        vals = f.(x̂)
        p = Inti.LagrangeLine(vals)
        @test p(0) ≈ 0
        @test p(1) ≈ 1
        @test p(0.1) ≈ 0.1^2
        ## line in 3d
        vtx = SVector(
            SVector(0.0, 0.0, 0.0),
            SVector(1.0, 1.0, 1.0)
        )
        l = Inti.LagrangeLine(vtx)
        @test Inti.domain(l) == Inti.ReferenceLine()
        @test l(0.1) ≈ SVector(0.1,0.1,0.1)
        ## line in 2d
        a = SVector(0.0, 0.0)
        b = SVector(1.0, 1.0)
        l = Inti.LagrangeLine((a, b))
        @test Inti.domain(l) == Inti.ReferenceLine()
    end
    @testset "LagrangeTriangle" begin
        # triangle in 2d
        vtx = SVector(
            SVector(0.0, 0.0),
            SVector(0.0, 1.0),
            SVector(-1.0, 0)
        )
        t = Inti.LagrangeTriangle(vtx)
        @test Inti.domain_dimension(t) == 2
        @test Inti.range_dimension(t) == 2
        # triangle in 3d
        vtx = SVector(
            SVector(0.0, 0.0, 0.0),
            SVector(0.0, 1.0, 0.0),
            SVector(-1.0, 0, 0.0)
        )
        t = Inti.LagrangeTriangle(vtx)
        @test Inti.range_dimension(t) == 3
        @test Inti.domain_dimension(t) == 2
    end
    @testset "Tetrahedron" begin
        # TODO: add tests
    end
end
