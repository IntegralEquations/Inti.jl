using Test
using Inti
using StaticArrays

@testset "ReferenceLine" begin
    d = Inti.ReferenceLine()
    k = 3
    sp = Inti.PolynomialSpace(d, k)
    @test Inti.dimension(sp) == k + 1
    b = Inti.monomial_basis(sp)
    x0 = 0.2
    @test length(b(x0)) == Inti.dimension(sp)

    # P0 basis
    sp = Inti.PolynomialSpace(d, 0)
    nodes = (0.5,)
    b = Inti.lagrange_basis(nodes, sp)
    @assert length(b(x0)) == length(nodes)
    @test b(0.5) ≈ SVector(1.0)

    # P1 basis
    sp = Inti.PolynomialSpace(d, 1)
    m = Inti.monomial_basis(sp)
    nodes = (0.0, 1.0)
    b = Inti.lagrange_basis(nodes, sp)
    @assert length(b(x0)) == length(nodes)
    @test @inferred(b(nodes[1])) ≈ SVector(1, 0)
    @test @inferred(b(nodes[2])) ≈ SVector(0, 1)

    # P5 basis
    sp = Inti.PolynomialSpace(d, 5)
    m = Inti.monomial_basis(sp)
    nodes = ntuple(6) do i
        return (i - 1) / 5
    end
    b = Inti.lagrange_basis(nodes, sp)
    @assert length(b(x0)) == length(nodes)
    for (i, xi) in enumerate(nodes)
        v = Inti.svector(j -> j == i ? 1 : 0, 6)
        @test @inferred(b(nodes[i])) ≈ v
    end
end

@testset "ReferenceTriangle" begin
    d = Inti.ReferenceTriangle()
    k = 3
    sp = Inti.PolynomialSpace(d, k)
    @test Inti.dimension(sp) == (k + 1) * (k + 2) / 2
    b = Inti.monomial_basis(sp)
    x₀ = (1 / 3, 1 / 3)
    @test length(b(x₀)) == Inti.dimension(sp)

    # P0 basis over triangle
    sp = Inti.PolynomialSpace(d, 0)
    nodes = [SVector(1 / 3, 1 / 3)]
    b = Inti.lagrange_basis(nodes, sp)
    @assert length(b(x₀)) == length(nodes)
    @test b(nodes[1]) ≈ SVector(1.0)

    # P1 basis over triangle
    sp = Inti.PolynomialSpace(d, 1)
    nodes = (SVector(0, 0), SVector(0, 1), SVector(1, 0))
    b = Inti.lagrange_basis(nodes, sp)
    @assert length(b(x₀)) == length(nodes)
    @test b(nodes[1]) ≈ SVector(1, 0, 0)
    @test b(nodes[2]) ≈ SVector(0, 1, 0)
    @test b(nodes[3]) ≈ SVector(0, 0, 1)

    # Test on VR quadrature
    q = Inti.VioreanuRokhlin(; domain = :triangle, order = 5)
    k = Inti.interpolation_order(q)
    sp = Inti.PolynomialSpace(d, k)
    nodes = Inti.qcoords(q)
    b = Inti.lagrange_basis(nodes, sp)
    check_alloc = (b, x) -> @allocated b(x)
    @test check_alloc(b, nodes[1]) == 0
    for (i, xi) in enumerate(nodes)
        v = Inti.svector(j -> j == i ? 1 : 0, length(nodes))
        @test @inferred(b(nodes[i])) ≈ v
    end
end

# @testset "ReferenceSquare" begin
#     d = Inti.ReferenceSquare()
#     k = 3
#     sp = Inti.PolynomialSpace(d, k)
#     @test Inti.dimension(sp) == (k + 1)^2
#     b = Inti.monomial_basis(sp)
#     @test length(b) == Inti.dimension(sp)
# end
