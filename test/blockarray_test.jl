using StaticArrays
using LinearAlgebra
using Inti
using Test
using HMatrices

@testset "Basic functionality" begin
    T = SVector{3,Float64}
    data = rand(30, 10)
    B = Inti.BlockArray{T}(data)
    @test Inti.one_padding((1, 2), (1, 2, 3)) == ((1, 2, 1), (1, 2, 3))
    @test Inti.one_padding((3, 2, 1), (1,)) == ((3, 2, 1), (1, 1, 1))
    @test Inti.blocksize(B) == (3,)
    @test Inti._blocksize_normalized(B) == (3, 1)
    @test size(B) == (10, 10)
    @test @inferred(B[1, 1]) == data[1:3, 1]
    @test @inferred(B[1, 2]) == data[1:3, 2]
    @test B[2, 1] == data[4:6, 1]
    @test B[2, 2] == data[4:6, 2]
    @test eltype(B) == T
    v = rand(T)
    B[1, 1] = v
    @test B[1, 1] == v
end

@testset "Undef constructor" begin
    T = SVector{3,Float64}
    B = Inti.BlockArray{T}(undef, (3, 3))
    @test Inti.blocksize(B) == (3,)
    @test Inti._blocksize_normalized(B) == (3, 1)
    @test size(B) == (3, 3)
    @test eltype(B) == T
end

@testset "Subset indexing" begin
    T = SVector{3,Float64}
    B = Inti.BlockArray{T}(undef, (3, 3))
    @test B[1:2, 1] isa Inti.BlockVector
    @test B[:, 1] isa Inti.BlockVector
    @test B[1:2, 1:2] isa Inti.BlockMatrix
    T = SMatrix{3,3,Float64,9}
    B = Inti.BlockArray{T}(undef, (3, 3))
    @test B[:, 1] isa Inti.BlockVector
    @test B[:, 1:1] isa Inti.BlockMatrix
    @test B[1:2, 1:2] isa Inti.BlockMatrix
end

@testset "Cross constructors" begin
    A = rand(SMatrix{3,3,Float64,9}, 3, 3)
    B = Inti.BlockArray(A)
    @test 2 * B isa Inti.BlockArray
    @test A ≈ B
    x = rand(SVector{3,Float64}, 3)
    @test A * x ≈ B * x
    X = rand(SMatrix{3,3,Float64,9}, 3)
    @test A * X ≈ B * X

    # zero it out
    fill!(B.data, 0)
    Azero = Array(B)
    @test norm(Azero * x) < 1e-14
end

@testset "Composition with HMatrix" begin
    op = Inti.Stokes(; dim = 3, μ = 2.0)
    Ω = Inti.ellipsoid() |> Inti.Domain
    Γ = Inti.boundary(Ω)
    Q = Inti.Quadrature(Γ; meshsize = 0.2, qorder = 2)
    K = Inti.SingleLayerKernel(op)
    iop = Inti.IntegralOperator(K, Q, Q)
    M = Matrix(iop)
    H = Inti.assemble_hmatrix(iop)
    T = eltype(iop)
    X = rand(T, size(H, 2), 3)
    @test M * X ≈ H * X
    B = Inti.BlockArray(X)
    @test M * B ≈ H * B
    # this type of product is used in e.g. the bdim correction
end
