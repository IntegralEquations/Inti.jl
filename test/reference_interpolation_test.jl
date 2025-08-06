using Test
using StaticArrays
using Inti
using LinearAlgebra

@testset "Lagrange elements" begin
    @testset "ℚₖ" begin
        for dim in 1:3
            D = Inti.ReferenceHyperCube{dim}
            @testset "$D" begin
                for n in 1:5 # points per dimension
                    Np = n^dim
                    T = Inti.LagrangeElement{D,Np}
                    x = Inti.reference_nodes(T)
                    # make sure it is exact on polynomials of degree n-1
                    xtest = rand(SVector{dim}, 100)
                    p = (x) -> sum(xd -> xd^(n - 1), x)
                    vals = p.(x)
                    el = Inti.LagrangeElement{D}(vals)
                    @test all(norm(el(x) - p(x)) < 1e-12 for x in xtest)
                    # and not exact on polynomials of degree n
                    p = (x) -> sum(xd -> xd^(n), x)
                    vals = p.(x)
                    el = Inti.LagrangeElement{D}(vals)
                    @test !all(el(x) ≈ p(x) for x in xtest)
                end
            end
        end
    end
    @testset "ℙₖ" begin
        for dim in 1:3
            D = Inti.ReferenceSimplex{dim}
            @testset "$D" begin
                for k in 0:5
                    Np = binomial(dim + k, k) # number of points in the simplex
                    T = Inti.LagrangeElement{D,Np}
                    Inti.order(T)
                    x = Inti.reference_nodes(T)
                    p = (x) -> sum(xd -> xd^k, x)
                    vals = p.(x)
                    el = Inti.LagrangeElement{D}(vals)
                    xtest = rand(SVector{dim}, 10)
                    @test all(norm(el(x) - p(x)) < 1e-12 for x in xtest)
                    # test inexactness on polynomials of degree k+1
                    p = (x) -> sum(xd -> xd^(k + 1), x)
                    vals = p.(x)
                    el = Inti.LagrangeElement{D}(vals)
                    @test !all(el(x) ≈ p(x) for x in xtest)
                end
            end
        end
    end
end
