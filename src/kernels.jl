const PREDEFINED_OPERATORS = ["Laplace", "Helmholtz", "Stokes", "Yukawa"]

"""
    abstract type AbstractKernel{T}

A kernel functions `K` with the signature `K(target,source)::T`.

See also: [`SingleLayerKernel`](@ref),
[`DoubleLayerKernel`](@ref), [`AdjointDoubleLayerKernel`](@ref),
[`HyperSingularKernel`](@ref)
"""
abstract type AbstractKernel{T} end

return_type(::AbstractKernel{T}) where {T} = T

"""
    abstract type AbstractDifferentialOperator{N}

A partial differential operator in dimension `N`.

`AbstractDifferentialOperator` types are used to define [`AbstractKernel`s](@ref
AbstractKernel) related to fundamental solutions of differential operators.
"""
abstract type AbstractDifferentialOperator{N} end

ambient_dimension(::AbstractDifferentialOperator{N}) where {N} = N

# convenient constructor for e.g. SingleLayerKernel(op,Float64) or DoubleLayerKernel(op,ComplexF64)
function (::Type{K})(
    op::Op;
    ::Type{T} = default_kernel_eltype(op),
) where {T,Op,K<:AbstractKernel}
    return K{T,Op}(op)
end

"""
    struct SingleLayerKernel{T,Op} <: AbstractKernel{T}

The free-space single-layer kernel (i.e. the fundamental solution) of an `Op <:
AbstractDifferentialOperator`.
"""
struct SingleLayerKernel{T,Op} <: AbstractKernel{T}
    op::Op
end

"""
    struct DoubleLayerKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space double-layer kernel. This
corresponds to the `γ₁` trace of the [`SingleLayerKernel`](@ref). For operators
such as [`Laplace`](@ref) or [`Helmholtz`](@ref), this is simply the normal
derivative of the fundamental solution respect to the source variable.
"""
struct DoubleLayerKernel{T,Op} <: AbstractKernel{T}
    op::Op
end

"""
    struct AdjointDoubleLayerKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space adjoint double-layer kernel.
This corresponds to the `transpose(γ₁,ₓ[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative of the fundamental
solution respect to the target variable.
"""
struct AdjointDoubleLayerKernel{T,Op} <: AbstractKernel{T}
    op::Op
end

"""
    struct HyperSingularKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space hypersingular kernel. This
corresponds to the `transpose(γ₁,ₓγ₁[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative respect to the target
variable of the `DoubleLayerKernel`.
"""
struct HyperSingularKernel{T,Op} <: AbstractKernel{T}
    op::Op
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

function (SL::SingleLayerKernel{T,Laplace{N}})(
    target,
    source;
    r = coords(target) - coords(source),
)::T where {N,T}
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

function (DL::DoubleLayerKernel{T,Laplace{N}})(
    target,
    source;
    r = coords(target) - coords(source),
)::T where {N,T}
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

function (ADL::AdjointDoubleLayerKernel{T,Laplace{N}})(
    target,
    source;
    r = coords(target) - coords(source),
)::T where {N,T}
    nx = normal(target)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return -1 / (2π) / (d^2) * dot(r, nx)
    elseif N == 3
        return -1 / (4π) / (d^3) * dot(r, nx)
    end
end

function (HS::HyperSingularKernel{T,Laplace{N}})(
    target,
    source;
    r = coords(target) - coords(source),
)::T where {N,T}
    nx = normal(target)
    ny = normal(source)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return 1 / (2π) / (d^2) * transpose(nx) * ((I - 2 * r * transpose(r) / d^2) * ny)
    elseif N == 3
        return 1 / (4π) / (d^3) * transpose(nx) * ((I - 3 * r * transpose(r) / d^2) * ny)
    end
end

################################################################################
################################# Yukawa #######################################
################################################################################

struct Yukawa{N,K<:Real} <: AbstractDifferentialOperator{N}
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
    return Yukawa{dim,typeof(λ)}(λ)
end

"""
    const ModifiedHelmholtz

Type alias for the [`Yukawa`](@ref) operator.
"""
const ModifiedHelmholtz = Yukawa

function Base.show(io::IO, ::Yukawa{N}) where {N}
    return print(io, "Yukawa operator in $N dimensions: -Δu + λ²u")
end

default_kernel_eltype(::Yukawa)  = Float64
default_density_eltype(::Yukawa) = Float64

function (SL::SingleLayerKernel{T,<:Yukawa{N,K}})(target, source)::T where {N,T,K}
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

function (DL::DoubleLayerKernel{T,Yukawa{N,K}})(target, source)::T where {N,T,K}
    x, y, ny = coords(target), coords(source), normal(source)
    λ = DL.op.λ
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        k = im * λ
        return λ / (2 * π * d) * Bessels.besselk(1, λ * d) .* dot(r, ny)
    elseif N == 3
        return 1 / (4π) / d^2 * exp(-λ * d) * (λ + 1 / d) * dot(r, ny)
    end
end

function (ADL::AdjointDoubleLayerKernel{T,<:Yukawa{N,K}})(target, source)::T where {N,T,K}
    x, y, nx = coords(target), coords(source), normal(target)
    λ = ADL.op.λ
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        k = im * λ
        return -λ / (2 * π * d) * Bessels.besselk(1, λ * d) .* dot(r, nx)
    elseif N == 3
        return -1 / (4π) / d^2 * exp(-λ * d) * (λ + 1 / d) * dot(r, nx)
    end
end

function (HS::HyperSingularKernel{T,<:Yukawa{N,K}})(target, source)::T where {N,T,K}
    x, y, nx, ny = coords(target), coords(source), normal(target), normal(source)
    λ = HS.op.λ
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    k = im * λ
    if N == 2
        RRT = r * transpose(r) # r ⊗ rᵗ
        # TODO: rewrite the operation below in a more clear/efficient way
        val =
            transpose(nx) * (
                (
                    -λ^2 / (2π) / d^2 * Bessels.besselk(2, λ * d) * RRT +
                    λ / (2 * π * d) * Bessels.besselk(1, λ * d) * I
                ) * ny
            )
        return val
    elseif N == 3
        term1 = 1 / (4π) / d^2 * exp(-λ * d) * (λ + 1 / d) * I
        term2 =
            r * transpose(r) / d * exp(-λ * d) / (4 * π * d^4) *
            (3 * (-d * λ - 1) - d^2 * λ^2)
        val = transpose(nx) * (term1 + term2) * ny
        return val
    end
end

################################################################################
################################# Helmholtz ####################################
################################################################################

struct Helmholtz{N,K} <: AbstractDifferentialOperator{N}
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
    return Helmholtz{dim,typeof(k)}(k)
end

function Base.show(io::IO, ::Helmholtz{N}) where {N}
    return print(io, "Helmholtz operator in $N dimensions: -Δu - k²u")
end

default_kernel_eltype(::Helmholtz) = ComplexF64
default_density_eltype(::Helmholtz) = ComplexF64

hankelh1(n, x::Real)    = Bessels.hankelh1(n, x)
hankelh1(n, x::Complex) = SpecialFunctions.hankelh1(n, x)

function (SL::SingleLayerKernel{T,<:Helmholtz{N}})(target, source)::T where {N,T}
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
function (DL::DoubleLayerKernel{T,<:Helmholtz{N}})(target, source)::T where {N,T}
    x, y, ny = coords(target), coords(source), normal(source)
    k = DL.op.k
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        val = im * k / 4 / d * hankelh1(1, k * d) .* dot(r, ny)
        return val
    elseif N == 3
        val = 1 / (4π) / d^2 * exp(im * k * d) * (-im * k + 1 / d) * dot(r, ny)
        return val
    end
end

# Adjoint double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T,<:Helmholtz{N}})(target, source)::T where {N,T}
    x, y, nx = coords(target), coords(source), normal(target)
    k = ADL.op.k
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        val = -im * k / 4 / d * hankelh1(1, k * d) .* dot(r, nx)
        return val
    elseif N == 3
        val = -1 / (4π) / d^2 * exp(im * k * d) * (-im * k + 1 / d) * dot(r, nx)
        return val
    end
end

# Hypersingular kernel
function (HS::HyperSingularKernel{T,<:Helmholtz{N}})(target, source)::T where {N,T}
    x, y, nx, ny = coords(target), coords(source), normal(target), normal(source)
    k = HS.op.k
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        val =
            transpose(nx) * (
                (
                    -im * k^2 / 4 / d^2 * hankelh1(2, k * d) * r * transpose(r) +
                    im * k / 4 / d * hankelh1(1, k * d) * I
                ) * ny
            )
        return val
    elseif N == 3
        RRT = r * transpose(r) # r ⊗ rᵗ
        term1 = 1 / (4π) / d^2 * exp(im * k * d) * (-im * k + 1 / d) * I
        term2 =
            RRT / d * exp(im * k * d) / (4 * π * d^4) * (3 * (d * im * k - 1) + d^2 * k^2)
        val = transpose(nx) * (term1 + term2) * ny
        return val
    end
end

############################ STOKES ############################3
struct Stokes{N,T} <: AbstractDifferentialOperator{N}
    μ::T
end

"""
    Stokes(; μ, dim)

Stokes operator in `dim` dimensions: ``[-μΔu + ∇p, ∇⋅u]``.
"""
Stokes(; μ, dim = 3) = Stokes{dim}(μ)
Stokes{N}(μ::T) where {N,T} = Stokes{N,T}(μ)

function Base.show(io::IO, op::Stokes{N}) where {N}
    return println(io, "Stokes operator in $N dimensions: [-μΔu + ∇p, ∇⋅u]")
end

default_kernel_eltype(::Stokes{N}) where {N} = SMatrix{N,N,Float64,N * N}
default_density_eltype(::Stokes{N}) where {N} = SVector{N,Float64}

# Single Layer
function (SL::SingleLayerKernel{T,<:Stokes{N}})(target, source)::T where {N,T}
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
function (DL::DoubleLayerKernel{T,<:Stokes{N}})(target, source)::T where {N,T}
    x = coords(target)
    y = coords(source)
    ny = normal(source)
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return 1 / π * dot(r, ny) / d^4 * r * transpose(r)
    elseif N == 3
        return 3 / (4π) * dot(r, ny) / d^5 * r * transpose(r)
    end
end

# Double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T,<:Stokes{N}})(target, source)::T where {N,T}
    x = coords(target)
    nx = normal(target)
    y = coords(source)
    r = x - y
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return -1 / π * dot(r, nx) / d^4 * r * transpose(r)
    elseif N == 3
        return -3 / (4π) * dot(r, nx) / d^5 * r * transpose(r)
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
struct Elastostatic{N,T} <: AbstractDifferentialOperator{N}
    μ::T
    λ::T
end
Elastostatic(; μ, λ, dim) = Elastostatic{dim}(promote(μ, λ)...)
Elastostatic{N}(μ::T, λ::T) where {N,T} = Elastostatic{N,T}(μ, λ)

function Base.show(io::IO, op::Elastostatic{N}) where {N}
    return print(io, "Elastostatic operator in $N dimensions: -μΔu - (μ+λ)∇(∇⋅u)")
end

default_kernel_eltype(::Elastostatic{N}) where {N} = SMatrix{N,N,Float64,N * N}
default_density_eltype(::Elastostatic{N}) where {N} = SVector{N,Float64}

function (SL::SingleLayerKernel{T,<:Elastostatic{N}})(target, source)::T where {N,T}
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

function (DL::DoubleLayerKernel{T,<:Elastostatic{N}})(target, source)::T where {N,T}
    μ, λ = DL.op.μ, DL.op.λ
    ν = λ / (2 * (μ + λ))
    x = coords(target)
    y = coords(source)
    ny = normal(source)
    ν = λ / (2 * (μ + λ))
    r = x .- y
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r * transpose(r) # r ⊗ rᵗ
    drdn = -dot(r, ny) / d
    if N == 2
        return -1 / (4π * (1 - ν) * d) * (
            drdn * ((1 - 2ν) * I + 2 * RRT / d^2) +
            (1 - 2ν) / d * (r * transpose(ny) - ny * transpose(r))
        )
    elseif N == 3
        return -1 / (8π * (1 - ν) * d^2) * (
            drdn * ((1 - 2 * ν) * I + 3 * RRT / d^2) +
            (1 - 2 * ν) / d * (r * transpose(ny) - ny * transpose(r))
        )
    end
end

function (ADL::AdjointDoubleLayerKernel{T,<:Elastostatic{N}})(target, source)::T where {N,T}
    μ, λ = ADL.op.μ, ADL.op.λ
    ν = λ / (2 * (μ + λ))
    x = coords(target)
    nx = normal(target)
    y = coords(source)
    ν = λ / (2 * (μ + λ))
    r = x .- y
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r * transpose(r) # r ⊗ rᵗ
    drdn = -dot(r, nx) / d
    if N == 2
        out =
            -1 / (4π * (1 - ν) * d) * (
                drdn * ((1 - 2ν) * I + 2 * RRT / d^2) +
                (1 - 2ν) / d * (r * transpose(nx) - nx * transpose(r))
            )
        return -transpose(out)
    elseif N == 3
        out =
            -1 / (8π * (1 - ν) * d^2) * (
                drdn * ((1 - 2 * ν) * I + 3 * RRT / d^2) +
                (1 - 2 * ν) / d * (r * transpose(nx) - nx * transpose(r))
            )
        return -transpose(out)
    end
end

function (HS::HyperSingularKernel{T,<:Elastostatic{N}})(target, source)::T where {N,T}
    μ, λ = HS.op.μ, HS.op.λ
    ν = λ / (2 * (μ + λ))
    x = coords(target)
    nx = normal(target)
    y = coords(source)
    ny = normal(source)
    r = x .- y
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r * transpose(r) # r ⊗ rᵗ
    drdn = dot(r, ny) / d
    if N == 2
        return μ / (2π * (1 - ν) * d^2) * (
            2 * drdn / d * (
                (1 - 2ν) * nx * transpose(r) + ν * (dot(r, nx) * I + r * transpose(nx)) -
                4 * dot(r, nx) * RRT / d^2
            ) +
            2 * ν / d^2 * (dot(r, nx) * ny * transpose(r) + dot(nx, ny) * RRT) +
            (1 - 2 * ν) * (
                2 / d^2 * dot(r, nx) * r * transpose(ny) +
                dot(nx, ny) * I +
                ny * transpose(nx)
            ) - (1 - 4ν) * nx * transpose(ny)
        )
    elseif N == 3
        return μ / (4π * (1 - ν) * d^3) * (
            3 * drdn / d * (
                (1 - 2ν) * nx * transpose(r) + ν * (dot(r, nx) * I + r * transpose(nx)) -
                5 * dot(r, nx) * RRT / d^2
            ) +
            3 * ν / d^2 * (dot(r, nx) * ny * transpose(r) + dot(nx, ny) * RRT) +
            (1 - 2 * ν) * (
                3 / d^2 * dot(r, nx) * r * transpose(ny) +
                dot(nx, ny) * I +
                ny * transpose(nx)
            ) - (1 - 4ν) * nx * transpose(ny)
        )
    end
end
