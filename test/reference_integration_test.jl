using Test
using LinearAlgebra
using Inti

@testset "Fejer quadrature" begin
    N = 5
    q = Inti.Fejer{N}()
    x, w = q()
    D = Inti.domain(q)
    @test D == Inti.ReferenceLine()
    @test all(qnode ∈ D for qnode in x)
    @test sum(w) ≈ 1
    # integrate all polynomial of degree N-1 exactly
    for n in 1:(N - 1)
        @test Inti.integrate(x -> x[1]^n, q) ≈ 1 / (n + 1)
    end
end

@testset "Gauss quad on triangle" begin
    d = Inti.ReferenceTriangle()
    # exact value for x^a*y^b integrate over reference triangle
    exa = (a, b) -> factorial(a) * factorial(b) / factorial(a + b + 2)
    # check all quadrature implemented
    orders = keys(Inti.TRIANGLE_GAUSS_ORDER_TO_NPTS)
    for p in orders
        q = Inti.Gauss(; domain=d, order=p)
        x, w = q()
        @test Inti.domain(q) == d
        @test all(qnode ∈ d for qnode in x)
        for i in 0:p, j in 0:p
            i + p > p && continue
            @test Inti.integrate(x -> x[1]^i * x[2]^j, q) ≈ exa(i, j)
        end
    end
end

@testset "Gauss quad on tetrahedron" begin
    d = Inti.ReferenceTetrahedron()
    # exact value for x^a*y^b*z^c integrate over reference tetrahedron
    exa = (a, b, c) -> factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3)
    # check all quadrature implemented
    orders = keys(Inti.TETRAHEDRON_GAUSS_ORDER_TO_NPTS)
    for p in orders
        q = Inti.Gauss(; domain=d, order=p)
        x, w = q()
        @test Inti.domain(q) == d
        @test all(qnode ∈ d for qnode in x)
        for i in 0:p, j in 0:p, k in 0:p
            i + j + k > p && continue
            @test Inti.integrate(x -> x[1]^i * x[2]^j * x[3]^k, q) ≈ exa(i, j, k)
        end
    end
end

@testset "Tensor product quad on square" begin
    px = 10
    py = 12
    qx = Inti.Fejer(;order=px)
    qy = Inti.Fejer(;order=py)
    q = Inti.TensorProductQuadrature(qx, qy)
    x, w = q()
    D = Inti.domain(q)
    @test D == Inti.ReferenceSquare()
    @test all(qnode ∈ D for qnode in x)
    for i in 0:px, j in 0:py
        @test Inti.integrate(x -> x[1]^i * x[2]^j, q) ≈ 1 / (i + 1) * 1 / (j + 1)
    end
end
