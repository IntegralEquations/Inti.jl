using Test
using LinearAlgebra
using StaticArrays
using Random
using Inti
using KernelAbstractions
using Metal
using CUDA

Random.seed!(1)

const KM_KERNELS = (
    Inti.SingleLayerKernel,
    Inti.DoubleLayerKernel,
    Inti.AdjointDoubleLayerKernel,
    Inti.HyperSingularKernel,
)

_km_tol(::Type{Float64}) = 1.0e-10
_km_tol(::Type{Float32}) = 1.0e-5

function km_quadrature(dim, ::Type{T}) where {T}
    Inti.clear_entities!()
    if dim == 2
        Γ = Inti.parametric_curve(t -> SVector(cos(t), sin(t)), 0, 2π) |> Inti.Domain
        msh = Inti.meshgen(Γ; meshsize = 0.05, T)
        return Inti.Quadrature(view(msh, Γ); qorder = 5)
    else
        Γ = Inti.boundary(Inti.ellipsoid() |> Inti.Domain)
        msh = Inti.meshgen(Γ; meshsize = 0.4, T)
        return Inti.Quadrature(view(msh, Γ); qorder = 4)
    end
end

function km_operators(::Type{T}, dim) where {T}
    ops = [
        ("Laplace", Inti.Laplace(; dim)),
        ("Stokes", Inti.Stokes(; dim, μ = T(1.5))),
        ("Elastostatic", Inti.Elastostatic(; dim, μ = T(1.0), λ = T(2.0))),
    ]
    if dim == 3
        push!(ops, ("Helmholtz", Inti.Helmholtz(; dim, k = T(1.3))))
        push!(ops, ("Yukawa", Inti.Yukawa(; dim, λ = T(1.2))))
    end
    return ops
end

function km_density_eltype(op, ::Type{T}) where {T}
    D = Inti.default_density_eltype(op)
    return D <: SVector ? SVector{Inti.ambient_dimension(op), T} :
        D <: Complex ? Complex{T} : T
end

function test_assemble_kernelmatrix(backend, ::Type{T}) where {T}
    tol = _km_tol(T)

    @testset "$name $(dim)D $(nameof(Ker))" for dim in (2, 3),
            (name, op) in km_operators(T, dim), Ker in KM_KERNELS

        quad = km_quadrature(dim, T)
        n = length(quad)
        x = rand(km_density_eltype(op, T), n)
        iop = Inti.IntegralOperator(Ker(op), quad, quad)
        yref = Inti.assemble_matrix(iop) * x
        A = Inti.assemble_kernelmatrix(iop; backend)
        @test size(A) == (n, n)
        @test norm(A * x - yref) / norm(yref) < tol
    end

    @testset "5-arg mul!" begin
        quad = km_quadrature(3, T)
        n = length(quad)
        op = Inti.Stokes(; dim = 3, μ = T(1.0))
        iop = Inti.IntegralOperator(Inti.SingleLayerKernel(op), quad, quad)
        A = Inti.assemble_kernelmatrix(iop; backend)
        D = km_density_eltype(op, T)
        x = rand(D, n)
        y0 = rand(D, n)
        α, β = T(2), T(-0.5)
        y = copy(y0)
        mul!(y, A, x, α, β)
        @test norm(y - (α * (A * x) + β * y0)) / norm(y) < tol
    end

    @testset "rectangular target != source" begin
        quad = km_quadrature(2, T)
        trg = [SVector(2cos(t), 2sin(t)) for t in range(T(0), T(2π); length = 37)]
        iop = Inti.IntegralOperator(Inti.DoubleLayerKernel(Inti.Laplace(; dim = 2)), trg, quad)
        x = rand(T, length(quad))
        yref = Inti.assemble_matrix(iop) * x
        A = Inti.assemble_kernelmatrix(iop; backend)
        @test size(A) == (length(trg), length(quad))
        @test norm(A * x - yref) / norm(yref) < tol
    end

    @testset "device-resident vectors" begin
        quad = km_quadrature(3, T)
        n = length(quad)
        iop = Inti.IntegralOperator(Inti.SingleLayerKernel(Inti.Laplace(; dim = 3)), quad, quad)
        A = Inti.assemble_kernelmatrix(iop; backend)
        x = rand(T, n)
        y0 = rand(T, n)
        yref = Inti.assemble_matrix(iop) * x
        xd = KernelAbstractions.adapt(backend, copy(x))
        # fully on-device 5-arg mul!
        yd = KernelAbstractions.adapt(backend, copy(y0))
        α, β = T(2), T(-0.5)
        mul!(yd, A, xd, α, β)
        @test norm(Array(yd) - (α * yref + β * y0)) / norm(yref) < tol
        # device * returns a device vector
        yd = A * xd
        @test KernelAbstractions.get_backend(yd) == backend
        @test norm(Array(yd) - yref) / norm(yref) < tol
        # mixed: device x, host y
        y = similar(x)
        mul!(y, A, xd)
        @test norm(y - yref) / norm(yref) < tol
    end

    @testset "non-default tile parameters" begin
        quad = km_quadrature(2, T)
        n = length(quad)
        iop = Inti.IntegralOperator(Inti.SingleLayerKernel(Inti.Laplace(; dim = 2)), quad, quad)
        x = rand(T, n)
        yref = Inti.assemble_matrix(iop) * x
        for (tg, tb) in ((32, 2), (128, 1))
            A = Inti.assemble_kernelmatrix(iop; backend, workgroupsize = tg, targets_per_lane = tb)
            @test norm(A * x - yref) / norm(yref) < tol
        end
    end

    @testset "complex density" begin
        quad = km_quadrature(3, T)
        n = length(quad)
        iop = Inti.IntegralOperator(Inti.SingleLayerKernel(Inti.Laplace(; dim = 3)), quad, quad)
        A = Inti.assemble_kernelmatrix(iop; backend)
        x = rand(Complex{T}, n)
        y = A * x
        @test eltype(y) == Complex{T}
        @test norm(y - Inti.assemble_matrix(iop) * x) / norm(y) < tol
    end
    return nothing
end

@testset "assemble_kernelmatrix (CPU, Float64)" begin
    test_assemble_kernelmatrix(KernelAbstractions.CPU(), Float64)
end

@testset "assemble_kernelmatrix (CUDA, Float64)" begin
    if !CUDA.functional()
        @test_skip "no functional CUDA device"
    else
        test_assemble_kernelmatrix(CUDA.CUDABackend(), Float64)
    end
end

@testset "assemble_kernelmatrix (Metal, Float32)" begin
    if !Metal.functional()
        @test_skip "no functional Metal device"
    else
        test_assemble_kernelmatrix(Metal.MetalBackend(), Float32)
    end
end
