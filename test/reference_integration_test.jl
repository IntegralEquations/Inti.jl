using Test
using LinearAlgebra
using Inti

@testset "Fejer quadrature" begin
    N = 6
    q = Inti.Fejer{N}()
    x, w = q()
    D = Inti.domain(q)
    @test D == Inti.ReferenceLine()
    @test all(qnode in D for qnode in x)
    @test sum(w) ≈ 1
    @test Inti.order(q) == N - 1
    # integrate all polynomial of degree N-1 exactly
    for n in 1:(N-1)
        @test Inti.integrate(x -> x[1]^n, q) ≈ 1 / (n + 1)
    end
    # check that our quadrature order is maximal
    @test Inti.integrate(x -> x[1]^N, q) ≉ 1 / (N + 1)
end

@testset "Gauss quad on triangle" begin
    d = Inti.ReferenceTriangle()
    # exact value for x^a*y^b integrate over reference triangle
    exa = (a, b) -> factorial(a) * factorial(b) / factorial(a + b + 2)
    # check all quadrature implemented
    orders = keys(Inti.TRIANGLE_GAUSS_ORDER_TO_NPTS)
    for p in orders
        q = Inti.Gauss(; domain = d, order = p)
        x, w = q()
        @test Inti.domain(q) == d
        @test all(qnode in d for qnode in x)
        @test Inti.order(q) == p
        for i in 0:p, j in 0:p
            i + j > p && continue
            @test Inti.integrate(x -> x[1]^i * x[2]^j, q) ≈ exa(i, j)
        end
        # check that our declared quadrature order is maximal
        allintegrated = true
        for i in 0:(p+1), j in 0:(p+1)
            i + j > (p + 1) && continue
            Inti.integrate(x -> x[1]^i * x[2]^j, q) ≈ exa(i, j) ||
                (allintegrated = false; break)
        end
        @test allintegrated == false
    end
end

@testset "Gauss quad on tetrahedron" begin
    d = Inti.ReferenceTetrahedron()
    # exact value for x^a*y^b*z^c integrate over reference tetrahedron
    exa = (a, b, c) -> factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3)
    # check all quadrature implemented
    orders = keys(Inti.TETRAHEDRON_GAUSS_ORDER_TO_NPTS)
    for p in orders
        q = Inti.Gauss(; domain = d, order = p)
        x, w = q()
        @test Inti.domain(q) == d
        @test all(qnode in d for qnode in x)
        @test Inti.order(q) == p
        for i in 0:p, j in 0:p, k in 0:p
            i + j + k > p && continue
            @test Inti.integrate(x -> x[1]^i * x[2]^j * x[3]^k, q) ≈ exa(i, j, k)
        end
        # check that our declared quadrature order is maximal
        allintegrated = true
        for i in 0:(p+1), j in 0:(p+1), k in 0:(p+1)
            i + j + k > (p + 1) && continue
            Inti.integrate(x -> x[1]^i * x[2]^j * x[3]^k, q) ≈ exa(i, j, k) ||
                (allintegrated = false; break)
        end
        @test allintegrated == false
    end
end

@testset "Vioreanu-Rokhlin quad on triangle" begin
    d = Inti.ReferenceTriangle()
    # exact value for x^a*y^b integrate over reference triangle
    exa = (a, b) -> factorial(a) * factorial(b) / factorial(a + b + 2)
    # check all quadrature implemented
    orders = keys(Inti.TRIANGLE_VR_ORDER_TO_NPTS)
    for p in orders
        q = Inti.VioreanuRokhlin(; domain = d, order = p)
        x, w = q()
        @test Inti.domain(q) == d
        @test all(qnode in d for qnode in x)
        @test Inti.order(q) == p
        for i in 0:p, j in 0:p
            i + j > p && continue
            @test Inti.integrate(x -> x[1]^i * x[2]^j, q) ≈ exa(i, j)
        end
        # check that our declared quadrature order is maximal
        allintegrated = true
        for i in 0:(p+1), j in 0:(p+1)
            i + j > (p + 1) && continue
            Inti.integrate(x -> x[1]^i * x[2]^j, q) ≈ exa(i, j) ||
                (allintegrated = false; break)
        end
        @test allintegrated == false
    end
end

@testset "Vioreanu-Rokhlin quad on tetrahedron" begin
    d = Inti.ReferenceTetrahedron()
    # exact value for x^a*y^b*z^c integrate over reference tetrahedron
    exa = (a, b, c) -> factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3)
    # check all quadrature implemented
    orders = keys(Inti.TETRAHEDRON_VR_ORDER_TO_NPTS)
    for p in orders
        q = Inti.VioreanuRokhlin(; domain = d, order = p)
        x, w = q()
        @test Inti.domain(q) == d
        @test all(qnode in d for qnode in x)
        @test Inti.order(q) == p
        for i in 0:p, j in 0:p, k in 0:p
            i + j + k > p && continue
            @test Inti.integrate(x -> x[1]^i * x[2]^j * x[3]^k, q) ≈ exa(i, j, k)
        end
        # check that our declared quadrature order is maximal
        allintegrated = true
        for i in 0:(p+1), j in 0:(p+1), k in 0:(p+1)
            i + j + k > (p + 1) && continue
            Inti.integrate(x -> x[1]^i * x[2]^j * x[3]^k, q) ≈ exa(i, j, k) ||
                (allintegrated = false; break)
        end
        @test allintegrated == false
    end
end

@testset "Tensor product quad on square" begin
    px = 10
    py = 12
    qx = Inti.Fejer(; order = px)
    qy = Inti.Fejer(; order = py)
    q = Inti.TensorProductQuadrature(qx, qy)
    x, w = q()
    D = Inti.domain(q)
    @test D == Inti.ReferenceSquare()
    @test all(qnode in D for qnode in x)
    # TODO write an order() for TensorProductQuadrature
    for i in 0:px, j in 0:py
        @test Inti.integrate(x -> x[1]^i * x[2]^j, q) ≈ 1 / (i + 1) * 1 / (j + 1)
    end
end
