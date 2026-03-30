const PREDEFINED_OPERATORS = ["Laplace", "Helmholtz", "Stokes", "Yukawa"]

"""
    abstract type AbstractKernel{T}

A kernel functions `K` with the signature `K(target,source)::T`.

See also: [`SingleLayerKernel`](@ref),
[`DoubleLayerKernel`](@ref), [`AdjointDoubleLayerKernel`](@ref),
[`HyperSingularKernel`](@ref)
"""
abstract type AbstractKernel{T} end

return_type(::AbstractKernel{T}, args...) where {T} = T

"""
    singularity_order(K)

Given a kernel `K` with signature `K(target,source)::T`, return the order of the singularity
of `K` at `target = source`. Order `n` means that `K(x,y) ∼ (x - y)^n` as `x -> y`.
"""
singularity_order(K) = nothing

"""
    abstract type AbstractDifferentialOperator{N}

A partial differential operator in dimension `N`.

`AbstractDifferentialOperator` types are used to define [`AbstractKernel`s](@ref
AbstractKernel) related to fundamental solutions of differential operators.
"""
abstract type AbstractDifferentialOperator{N} end

ambient_dimension(::AbstractDifferentialOperator{N}) where {N} = N

function range_dimension(op::AbstractDifferentialOperator)
    T = default_density_eltype(op)
    if T <: Number
        return 1
    elseif hasmethod(length, Tuple{Type{T}})
        return length(T)
    else
        error("default_density_eltype($(typeof(op))) = $T does not define length(::Type{$T}); cannot determine range dimension")
    end
end

# convenient constructor for e.g. SingleLayerKernel(op,Float64) or DoubleLayerKernel(op,ComplexF64)
function (::Type{K})(
        op::Op,
        ::Type{T} = default_kernel_eltype(op),
    ) where {T, Op, K <: AbstractKernel}
    return K{T, Op}(op)
end

operator(K::AbstractKernel) = K.op

"""
    struct SingleLayerKernel{T,Op} <: AbstractKernel{T}

The free-space single-layer kernel (i.e. the fundamental solution) of an `Op <:
AbstractDifferentialOperator`.
"""
struct SingleLayerKernel{T, Op} <: AbstractKernel{T}
    op::Op
end

function singularity_order(K::SingleLayerKernel)
    N = ambient_dimension(K.op)
    return 2 - N
end

"""
    struct DoubleLayerKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space double-layer kernel. This
corresponds to the `γ₁` trace of the [`SingleLayerKernel`](@ref). For operators
such as [`Laplace`](@ref) or [`Helmholtz`](@ref), this is simply the normal
derivative of the fundamental solution with respect to the source variable.
"""
struct DoubleLayerKernel{T, Op} <: AbstractKernel{T}
    op::Op
end

function singularity_order(K::DoubleLayerKernel)
    N = ambient_dimension(K.op)
    return 1 - N
end

"""
    struct AdjointDoubleLayerKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space adjoint double-layer kernel.
This corresponds to the `transpose(γ₁,ₓ[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative of the fundamental
solution respect to the target variable.
"""
struct AdjointDoubleLayerKernel{T, Op} <: AbstractKernel{T}
    op::Op
end

function singularity_order(K::AdjointDoubleLayerKernel)
    N = ambient_dimension(K.op)
    return 1 - N
end

"""
    struct HyperSingularKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space hypersingular kernel. This
corresponds to the `transpose(γ₁,ₓγ₁[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative respect to the target
variable of the `DoubleLayerKernel`.
"""
struct HyperSingularKernel{T, Op} <: AbstractKernel{T}
    op::Op
end

function singularity_order(K::HyperSingularKernel)
    N = ambient_dimension(K.op)
    return -N
end

################################################################################
################################# LAPLACE ######################################
################################################################################

struct Laplace{N} <: AbstractDifferentialOperator{N} end

"""
    Laplace(; dim)

Laplace's differential operator in `dim` dimension: ``-Δu``.
```

Note the **negative sign** in the definition.
"""
Laplace(; dim) = Laplace{dim}()

function Base.show(io::IO, op::Laplace{N}) where {N}
    return print(io, "Laplace operator in $N dimensions: -Δu")
end

default_kernel_eltype(::Laplace) = Float64
default_density_eltype(::Laplace) = Float64

function (SL::SingleLayerKernel{T, Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    )::T where {N, T}
    d = norm(r)
    (d ≤ SAME_POINT_TOLERANCE) && return zero(T)
    if N == 2
        return -1 / (2π) * log(d)
    elseif N == 3
        return 1 / (4π) / d
    else
        notimplemented()
    end
end

function (DL::DoubleLayerKernel{T, Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    )::T where {N, T}
    ny = normal(source)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return 1 / (2π) / (d^2) * dot(r, ny)
    elseif N == 3
        return 1 / (4π) / (d^3) * dot(r, ny)
    else
        notimplemented()
    end
end

function (ADL::AdjointDoubleLayerKernel{T, Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    )::T where {N, T}
    nx = normal(target)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return -1 / (2π) / (d^2) * dot(r, nx)
    elseif N == 3
        return -1 / (4π) / (d^3) * dot(r, nx)
    end
end

function (HS::HyperSingularKernel{T, Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    nx = normal(target)
    ny = normal(source)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    id2 = inv(d * d)
    # nxᵀ(I - N*rrᵀ/d²)ny = nxdny - N*rdnx*rdny/d²
    nxdny = dot(nx, ny)
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    if N == 2
        return id2 / (2π) * (nxdny - 2 * rdnx * rdny * id2)
    elseif N == 3
        return id2 / (4π * d) * (nxdny - 3 * rdnx * rdny * id2)
    end
end

################################################################################
################################# Yukawa #######################################
################################################################################

struct Yukawa{N, K <: Real} <: AbstractDifferentialOperator{N}
    λ::K
end

"""
    Yukawa(; λ, dim)

Yukawa operator, also known as modified Helmholtz, in `dim` dimensions: ``-Δu + λ²u``.

The parameter `λ` is a positive number. Note the **negative sign** in front of
the Laplacian.
"""
function Yukawa(; λ, dim)
    @assert λ > 0 "λ must be a positive number"
    return Yukawa{dim, typeof(λ)}(λ)
end

"""
    const ModifiedHelmholtz

Type alias for the [`Yukawa`](@ref) operator.
"""
const ModifiedHelmholtz = Yukawa

function Base.show(io::IO, ::Yukawa{N}) where {N}
    return print(io, "Yukawa operator in $N dimensions: -Δu + λ²u")
end

default_kernel_eltype(::Yukawa) = Float64
default_density_eltype(::Yukawa) = Float64

function (SL::SingleLayerKernel{T, <:Yukawa{N, K}})(target, source)::T where {N, T, K}
    x = coords(target)
    y = coords(source)
    λ = SL.op.λ
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return 1 / (2π) * Bessels.besselk(0, λ * d)
    elseif N == 3
        return 1 / (4π) / d * exp(-λ * d)
    end
end

function (DL::DoubleLayerKernel{T, Yukawa{N, K}})(target, source)::T where {N, T, K}
    x, y, ny = coords(target), coords(source), normal(source)
    λ = DL.op.λ
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return λ / (2π * d) * Bessels.besselk(1, λ * d) * dot(r, ny)
    elseif N == 3
        return inv(4π * d^2) * exp(-λ * d) * (λ + inv(d)) * dot(r, ny)
    end
end

function (ADL::AdjointDoubleLayerKernel{T, <:Yukawa{N, K}})(target, source)::T where {N, T, K}
    x, y, nx = coords(target), coords(source), normal(target)
    λ = ADL.op.λ
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return -λ / (2 * π * d) * Bessels.besselk(1, λ * d) * dot(r, nx)
    elseif N == 3
        return -1 / (4π) / d^2 * exp(-λ * d) * (λ + 1 / d) * dot(r, nx)
    end
end

function (HS::HyperSingularKernel{T, <:Yukawa{N, K}})(target, source)::T where {N, T, K}
    nx, ny = normal(target), normal(source)
    λ = HS.op.λ
    r = coords(target) - coords(source)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    # nxᵀ(a*rrᵀ + b*I)ny = a*rdnx*rdny + b*nxdny
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    nxdny = dot(nx, ny)
    if N == 2
        k1 = Bessels.besselk(1, λ * d)
        k2 = Bessels.besselk(2, λ * d)
        a = -λ^2 / (2π * d^2) * k2
        b = λ / (2π * d) * k1
        return a * rdnx * rdny + b * nxdny
    elseif N == 3
        id2 = inv(d * d)
        emld = exp(-λ * d)
        b = emld * id2 / (4π) * (λ + inv(d))
        a = emld * id2 * id2 / (4π * d) * (-3 * (d * λ + 1) - d^2 * λ^2)
        return a * rdnx * rdny + b * nxdny
    end
end

################################################################################
################################# Helmholtz ####################################
################################################################################

struct Helmholtz{N, K} <: AbstractDifferentialOperator{N}
    k::K
end

"""
    Helmholtz(; k, dim)

Helmholtz operator in `dim` dimensions: `-Δu - k²u`.

The parameter `k` can be a real or complex number. For purely imaginary
wavenumbers, consider using the [`Yukawa`](@ref) kernel.
"""
function Helmholtz(; k, dim)
    if k isa Complex
        @assert imag(k) ≥ 0 "k must have a non-negative imaginary part"
        if iszero(real(k))
            msg = """Purely imaginary wavenumber detected in Helmholtz operator.
            Creating a modified Helmholtz (Yukawa) op instead."""
            @warn msg
            return Yukawa(; λ = imag(k), dim = dim)
        elseif iszero(imag(k))
            return Helmholtz(; k = real(k), dim = dim)
        end
    end
    return Helmholtz{dim, typeof(k)}(k)
end

function Base.show(io::IO, ::Helmholtz{N}) where {N}
    return print(io, "Helmholtz operator in $N dimensions: -Δu - k²u")
end

default_kernel_eltype(::Helmholtz) = ComplexF64
default_density_eltype(::Helmholtz) = ComplexF64

hankelh1(n, x::Real) = Bessels.hankelh1(n, x)
hankelh1(n, x::Complex) = SpecialFunctions.hankelh1(n, x)

function (SL::SingleLayerKernel{T, <:Helmholtz{N}})(target, source)::T where {N, T}
    x = coords(target)
    y = coords(source)
    k = SL.op.k
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return im / 4 * hankelh1(0, k * d)
    elseif N == 3
        return 1 / (4π) / d * exp(im * k * d)
    end
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T, <:Helmholtz{N}})(target, source)::T where {N, T}
    x, y, ny = coords(target), coords(source), normal(source)
    k = DL.op.k
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return im * k / (4d) * hankelh1(1, k * d) * dot(r, ny)
    elseif N == 3
        return inv(4π * d^2) * exp(im * k * d) * (-im * k + inv(d)) * dot(r, ny)
    end
end

# Adjoint double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T, <:Helmholtz{N}})(target, source)::T where {N, T}
    x, y, nx = coords(target), coords(source), normal(target)
    k = ADL.op.k
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return -im * k / (4d) * hankelh1(1, k * d) * dot(r, nx)
    elseif N == 3
        return -inv(4π * d^2) * exp(im * k * d) * (-im * k + inv(d)) * dot(r, nx)
    end
end

# Hypersingular kernel
function (HS::HyperSingularKernel{T, <:Helmholtz{N}})(target, source)::T where {N, T}
    nx, ny = normal(target), normal(source)
    k = HS.op.k
    r = coords(target) - coords(source)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    # nxᵀ(a*rrᵀ + b*I)ny = a*rdnx*rdny + b*nxdny
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    nxdny = dot(nx, ny)
    if N == 2
        h1 = hankelh1(1, k * d)
        h2 = hankelh1(2, k * d)
        a = -im * k^2 / (4 * d^2) * h2
        b = im * k / (4 * d) * h1
        return a * rdnx * rdny + b * nxdny
    elseif N == 3
        id2 = inv(d * d)
        eikd = exp(im * k * d)
        b = eikd * id2 / (4π) * (-im * k + inv(d))
        a = eikd * id2 * id2 / (4π * d) * (3 * (d * im * k - 1) + d^2 * k^2)
        return a * rdnx * rdny + b * nxdny
    end
end

############################ STOKES ############################3
struct Stokes{N, T} <: AbstractDifferentialOperator{N}
    μ::T
end

"""
    Stokes(; μ, dim)

Stokes operator in `dim` dimensions: ``[-μΔu + ∇p, ∇⋅u]``.
"""
Stokes(; μ, dim = 3) = Stokes{dim}(μ)
Stokes{N}(μ::T) where {N, T} = Stokes{N, T}(μ)

function Base.show(io::IO, op::Stokes{N}) where {N}
    return println(io, "Stokes operator in $N dimensions: [-μΔu + ∇p, ∇⋅u]")
end

default_kernel_eltype(::Stokes{N}) where {N} = SMatrix{N, N, Float64, N * N}
default_density_eltype(::Stokes{N}) where {N} = SVector{N, Float64}

# Single Layer
function (SL::SingleLayerKernel{T, <:Stokes{N}})(target, source)::T where {N, T}
    μ = SL.op.μ
    x = coords(target)
    y = coords(source)
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        γ = -log(d)
    elseif N == 3
        γ = 1 / d
    end
    return 1 / (4π * (N - 1) * μ) * (γ * I + r * transpose(r) / d^N)
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T, <:Stokes{N}})(target, source)::T where {N, T}
    ny = normal(source)
    r = coords(target) - coords(source)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    id2 = inv(d * d)
    RRT = r * transpose(r)
    if N == 2
        return id2 * id2 / π * dot(r, ny) * RRT
    elseif N == 3
        return 3 * id2 * id2 / (4π * d) * dot(r, ny) * RRT
    end
end

# Adjoint Double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T, <:Stokes{N}})(target, source)::T where {N, T}
    nx = normal(target)
    r = coords(target) - coords(source)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    id2 = inv(d * d)
    RRT = r * transpose(r)
    if N == 2
        return -id2 * id2 / π * dot(r, nx) * RRT
    elseif N == 3
        return -3 * id2 * id2 / (4π * d) * dot(r, nx) * RRT
    end
end

# TODO: Stokes hypersingular kernel

################################################################################
################################# Elastostatic #################################
################################################################################

"""
    struct Elastostatic{N,T} <: AbstractDifferentialOperator{N}

Elastostatic operator in `N` dimensions: -μΔu - (μ+λ)∇(∇⋅u)

Note that the displacement ``u`` is a vector of length `N` since this is a
vectorial problem.
"""
struct Elastostatic{N, T} <: AbstractDifferentialOperator{N}
    μ::T
    λ::T
end
Elastostatic(; μ, λ, dim) = Elastostatic{dim}(promote(μ, λ)...)
Elastostatic{N}(μ::T, λ::T) where {N, T} = Elastostatic{N, T}(μ, λ)

function Base.show(io::IO, op::Elastostatic{N}) where {N}
    return print(io, "Elastostatic operator in $N dimensions: -μΔu - (μ+λ)∇(∇⋅u)")
end

default_kernel_eltype(::Elastostatic{N}) where {N} = SMatrix{N, N, Float64, N * N}
default_density_eltype(::Elastostatic{N}) where {N} = SVector{N, Float64}

function (SL::SingleLayerKernel{T, <:Elastostatic{N}})(target, source)::T where {N, T}
    μ, λ = SL.op.μ, SL.op.λ
    ν = λ / (2 * (μ + λ))
    x = coords(target)
    y = coords(source)
    r = x .- y
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r * transpose(r) # r ⊗ rᵗ
    if N == 2
        return 1 / (8π * μ * (1 - ν)) * (-(3 - 4 * ν) * log(d) * I + RRT / d^2)
    elseif N == 3
        return 1 / (16π * μ * (1 - ν) * d) * ((3 - 4 * ν) * I + RRT / d^2)
    end
end

function (DL::DoubleLayerKernel{T, <:Elastostatic{N}})(target, source)::T where {N, T}
    μ, λ = DL.op.μ, DL.op.λ
    ν = λ / (2 * (μ + λ))
    ny = normal(source)
    r = coords(target) .- coords(source)
    d = norm(r)
    d == 0 && return zero(T)
    id2 = inv(d * d)
    RRT = r * transpose(r)
    drdn = -dot(r, ny) / d
    ν1 = 1 - 2ν
    if N == 2
        return -inv(4π * (1 - ν) * d) * (
            drdn * (ν1 * I + 2 * id2 * RRT) +
                ν1 / d * (r * transpose(ny) - ny * transpose(r))
        )
    elseif N == 3
        return -id2 / (8π * (1 - ν)) * (
            drdn * (ν1 * I + 3 * id2 * RRT) +
                ν1 / d * (r * transpose(ny) - ny * transpose(r))
        )
    end
end

function (ADL::AdjointDoubleLayerKernel{T, <:Elastostatic{N}})(target, source)::T where {N, T}
    μ, λ = ADL.op.μ, ADL.op.λ
    ν = λ / (2 * (μ + λ))
    nx = normal(target)
    r = coords(target) .- coords(source)
    d = norm(r)
    d == 0 && return zero(T)
    id2 = inv(d * d)
    RRT = r * transpose(r)
    drdn = -dot(r, nx) / d
    ν1 = 1 - 2ν
    # ADL = -transpose(DL with ny→nx), which flips the sign of the antisymmetric part
    if N == 2
        return inv(4π * (1 - ν) * d) * (
            drdn * (ν1 * I + 2 * id2 * RRT) +
                ν1 / d * (nx * transpose(r) - r * transpose(nx))
        )
    elseif N == 3
        return id2 / (8π * (1 - ν)) * (
            drdn * (ν1 * I + 3 * id2 * RRT) +
                ν1 / d * (nx * transpose(r) - r * transpose(nx))
        )
    end
end

function (HS::HyperSingularKernel{T, <:Elastostatic{N}})(target, source) where {N, T}
    μ, λ = HS.op.μ, HS.op.λ
    ν = λ / (2 * (μ + λ))
    nx = normal(target)
    ny = normal(source)
    r = coords(target) .- coords(source)
    d = norm(r)
    d == 0 && return zero(T)
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    nxdny = dot(nx, ny)
    if N == 2
        c = μ / (2π * (1 - ν) * d^2)
        α = 2 * rdny / d^2
        # Decompose as: c * (r⊗vr + nx⊗vnx + ny⊗vny + a_diag * I)
        vr = (-4α * rdnx + 2ν * nxdny) / d^2 * r + α * ν * nx + (1 - 2ν) * 2 * rdnx / d^2 * ny
        vnx = (1 - 2ν) * α * r - (1 - 4ν) * ny
        vny = 2ν * rdnx / d^2 * r + (1 - 2ν) * nx
        a_diag = α * ν * rdnx + (1 - 2ν) * nxdny
    elseif N == 3
        c = μ / (4π * (1 - ν) * d^3)
        α = 3 * rdny / d^2
        vr = (-5α * rdnx + 3ν * nxdny) / d^2 * r + α * ν * nx + (1 - 2ν) * 3 * rdnx / d^2 * ny
        vnx = (1 - 2ν) * α * r - (1 - 4ν) * ny
        vny = 3ν * rdnx / d^2 * r + (1 - 2ν) * nx
        a_diag = α * ν * rdnx + (1 - 2ν) * nxdny
    end
    return c * (r * transpose(vr) + nx * transpose(vnx) + ny * transpose(vny) + a_diag * I)
end

################################################################################
################################# LAPLACE PERIODIC #############################
################################################################################

struct LaplacePeriodic1D{N, T <: Real} <: AbstractDifferentialOperator{N}
    period::T
end

"""
    LaplacePeriodic1D(; dim, period = 2π)

Laplace's differential operator `-Δu` in `dim` dimension with periodic boundary
conditions along the first dimension. The `period` is set to `2π` by default, and the
periodic cell is defined as `[-period/2, period/2]`.

The negative sign is used to match the convention of coercive operators.
"""
LaplacePeriodic1D(; dim, period = 2π) = LaplacePeriodic1D{dim, typeof(period)}(period)

function Base.show(io::IO, op::LaplacePeriodic1D{N}) where {N}
    return print(
        io,
        "Periodic Laplace operator -Δu in $N dimensions with periodic conditions along the first dimension",
    )
end

default_kernel_eltype(::LaplacePeriodic1D) = Float64
default_density_eltype(::LaplacePeriodic1D) = Float64

function (SL::SingleLayerKernel{T, <:LaplacePeriodic1D{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    l = SL.op.period
    if N == 2
        d2 = sin(π / l * r[1])^2 + sinh(π / l * r[2])^2
        out = -1 / 4π * log(d2)
        return d2 ≤ SAME_POINT_TOLERANCE ? zero(T) : out
    else
        error("Single layer kernel for LaplacePeriodic1D not implemented in $N dimensions")
    end
end

function (DL::DoubleLayerKernel{T, <:LaplacePeriodic1D{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    ny = normal(source)
    if N == 2
        l = DL.op.period
        s = sin(π / l * r[1])
        sh = sinh(π / l * r[2])
        d2 = s^2 + sh^2
        out = 1 / (4π * d2) * (2 * π / l * s * cos(π / l * r[1]) * ny[1] + 2 * π / l * sh * cosh(π / l * r[2]) * ny[2])
        return d2 ≤ SAME_POINT_TOLERANCE ? zero(T) : out
    else
        error("Double layer kernel for LaplacePeriodic1D not implemented in $N dimensions")
    end
end

function (ADL::AdjointDoubleLayerKernel{T, <:LaplacePeriodic1D{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    nx = normal(target)
    if N == 2
        l = ADL.op.period
        s = sin(π / l * r[1])
        sh = sinh(π / l * r[2])
        d2 = s^2 + sh^2
        out = -1 / (4π * d2) * (2 * π / l * s * cos(π / l * r[1]) * nx[1] + 2 * π / l * sh * cosh(π / l * r[2]) * nx[2])
        return d2 ≤ SAME_POINT_TOLERANCE ? zero(T) : out
    else
        error(
            "Adjoint double layer kernel for LaplacePeriodic1D not implemented in $N dimensions",
        )
    end
end

function (HS::HyperSingularKernel{T, <:LaplacePeriodic1D{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    x = coords(target)
    nx = normal(target)
    ny = normal(source)
    if N == 2
        dGdny = DoubleLayerKernel(HS.op)
        # TODO: in the case of the double- and a adjoint double-layer kernerls, I observed
        # that ForwardDiff is slighly slower than the analytical forms. That may still be
        # the case here, so we should consider implementing the analytical form.
        ForwardDiff.derivative(t -> dGdny(x + t * nx, source), 0)
    else
        return error(
            "Hypersingular kernel for LaplacePeriodic1D not implemented in $N dimensions",
        )
    end
end

################################################################################
################################# HELMHOLTZ PERIODIC ###########################
################################################################################

function HelmholtzPeriodic1D(args...; kwargs...)
    return error("HelmholtzPeriodic1D not found. Did you forget to import QPGreen?")
end
