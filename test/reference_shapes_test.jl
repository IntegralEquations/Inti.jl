using Test
using Inti
using StaticArrays

@testset "Line" begin
    l = Inti.ReferenceLine()
    @test Inti.ambient_dimension(l) == 1
    @test Inti.geometric_dimension(l) == 1
    x = SVector(0.5)
    @test x ∈ l
    x = SVector(1.0)
    @test x ∈ l
    x = SVector(1.1)
    @test !in(x, l)
    @test Inti.center(l) == SVector(1/2)
    @test Inti.vertices(l) == (SVector(0), SVector(1))
end
@testset "Triangle" begin
    t = Inti.ReferenceTriangle()
    @test Inti.ambient_dimension(t) == 2
    @test Inti.geometric_dimension(t) == 2
    x = SVector(0.5, 0.5)
    @test x ∈ t
    x = SVector(1.0, 0.0)
    @test x ∈ t
    x = SVector(1.1, 0.0)
    @test !in(x, t)
    @test Inti.center(t) == SVector(1/3, 1/3)
    @test Inti.vertices(t) == SVector([0, 0], [1, 0], [0, 1])
end
@testset "Tetrahedron" begin
    t = Inti.ReferenceTetrahedron()
    @test Inti.ambient_dimension(t) == 3
    @test Inti.geometric_dimension(t) == 3
    x = SVector(0.5, 0.5, 0.0)
    @test x ∈ t
    x = SVector(1.0, 0.0, 0.0) # point on edge
    @test x ∈ t
    x = SVector(1.1, 0.0, 0.0)
    @test !in(x, t)
    @test Inti.center(t) == SVector(1/4, 1/4, 1/4)
    @test Inti.vertices(t) == SVector([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
end
@testset "NSimplex" begin
    t = Inti.ReferenceSimplex{4}()
    @test Inti.ambient_dimension(t) == 4
    @test Inti.geometric_dimension(t) == 4
    x = SVector(0.5, 0.4, 0.0, 0.05)
    @test x ∈ t
    x = SVector(1.0, 0.0, 0.0, 0.0)
    @test x ∈ t
    x = SVector(1.1, 0.0, 0.0, 0.0)
    @test !in(x, t)
    @test Inti.center(t) == SVector(1/5, 1/5, 1/5, 1/5)
    @test Inti.vertices(t) == SVector([0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
end
