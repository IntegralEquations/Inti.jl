const PREDEFINED_KERNELS = ["Laplace", "Helmholtz", "Stokes"]
"""
    abstract type AbstractKernel{T}

A kernel functions `K` with the signature `K(target,source)::T`.

See also: [`GenericKernel`](@ref), [`SingleLayerKernel`](@ref), [`DoubleLayerKernel`](@ref), [`AdjointDoubleLayerKernel`](@ref), [`HyperSingularKernel`](@ref)
"""
abstract type AbstractKernel{T} end

return_type(::AbstractKernel{T}) where {T} = T

"""
    struct GenericKernel{T,F} <: AbstractKernel{T}

An [`AbstractKernel`](@ref) with `kernel` of type `F`.
"""
struct GenericKernel{T,F} <: AbstractKernel{T}
    kernel::F
end

"""
    abstract type AbstractPDE{N}

A partial differential equation in dimension `N`. `AbstractPDE` types are used
to define `AbstractPDEKernel`s.
"""
abstract type AbstractPDE{N} end

ambient_dimension(::AbstractPDE{N}) where {N} = N

"""
    abstract type AbstractPDEKernel{T,Op} <: AbstractKernel{T}

An [`AbstractKernel`](@ref) with an associated `pde::Op` field.
"""
abstract type AbstractPDEKernel{T,Op} <: AbstractKernel{T} end

"""
    pde(K::AbstractPDEKernel)

Return the underlying `AbstractPDE` when `K` correspond to the kernel related to
the underlying Greens function of a PDE.
"""
pde(k::AbstractPDEKernel) = k.pde

parameters(k::AbstractPDEKernel) = parameters(pde(k))

# convenient constructor for e.g. SingleLayerKernel(pde,Float64) or DoubleLayerKernel(pde,ComplexF64)
function (::Type{K})(
    pde::Op,
    ::Type{T} = default_kernel_eltype(pde),
) where {T,Op,K<:AbstractPDEKernel}
    return K{T,Op}(pde)
end

"""
    struct SingleLayerKernel{T,Op} <: AbstractPDEKernel{T,Op}

The free-space single-layer kernel (i.e. the fundamental solution) of an `OP <:
AbstractPDE`.
"""
struct SingleLayerKernel{T,Op} <: AbstractPDEKernel{T,Op}
    pde::Op
end

"""
    struct DoubleLayerKernel{T,Op} <: AbstractPDEKernel{T,Op}

Given an operator `Op`, construct its free-space double-layer kernel. This
corresponds to the `γ₁` trace of the [`SingleLayerKernel`](@ref). For operators
such as [`Laplace`](@ref) or [`Helmholtz`](@ref), this is simply the normal
derivative of the fundamental solution respect to the source variable.
"""
struct DoubleLayerKernel{T,Op} <: AbstractPDEKernel{T,Op}
    pde::Op
end

"""
    struct AdjointDoubleLayerKernel{T,Op} <: AbstractPDEKernel{T,Op}

Given an operator `Op`, construct its free-space adjoint double-layer kernel.
This corresponds to the `transpose(γ₁,ₓ[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative of the fundamental
solution respect to the target variable.
"""
struct AdjointDoubleLayerKernel{T,Op} <: AbstractPDEKernel{T,Op}
    pde::Op
end

"""
    struct HyperSingularKernel{T,Op} <: AbstractPDEKernel{T,Op}

Given an operator `Op`, construct its free-space hypersingular kernel. This
corresponds to the `transpose(γ₁,ₓγ₁[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative of the fundamental
solution respect to the target variable of the `DoubleLayerKernel`.
"""
struct HyperSingularKernel{T,Op} <: AbstractPDEKernel{T,Op}
    pde::Op
end

################################################################################
################################# LAPLACE ######################################
################################################################################

"""
    struct Laplace{N}

Laplace equation in `N` dimension: Δu = 0.
"""
struct Laplace{N} <: AbstractPDE{N} end

Laplace(; dim) = Laplace{dim}()

function Base.show(io::IO, pde::Laplace)
    return print(io, "Δu = 0")
end

default_kernel_eltype(::Laplace) = Float64
default_density_eltype(::Laplace) = Float64

parameters(::Laplace) = nothing

function (SL::SingleLayerKernel{T,Laplace{N}})(
    target,
    source,
    r = coords(target) - coords(source),
)::T where {N,T}
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (-1 / (2π) * log(d))
    elseif N == 3
        return filter * (1 / (4π) / d)
    else
        notimplemented()
    end
end

function (DL::DoubleLayerKernel{T,Laplace{N}})(
    target,
    source,
    r = coords(target) - coords(source),
)::T where {N,T}
    ny = normal(source)
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (1 / (2π) / (d^2) * dot(r, ny))
    elseif N == 3
        return filter * (1 / (4π) / (d^3) * dot(r, ny))
    else
        notimplemented()
    end
end

function (ADL::AdjointDoubleLayerKernel{T,Laplace{N}})(
    target,
    source,
    r = coords(target) - coords(source),
)::T where {N,T}
    nx = normal(target)
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (-1 / (2π) / (d^2) * dot(r, nx))
    elseif N == 3
        return filter * (-1 / (4π) / (d^3) * dot(r, nx))
    end
end

function (HS::HyperSingularKernel{T,Laplace{N}})(
    target,
    source,
    r = coords(target) - coords(source),
)::T where {N,T}
    nx = normal(target)
    ny = normal(source)
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter *
               (1 / (2π) / (d^2) * transpose(nx) * ((I - 2 * r * transpose(r) / d^2) * ny))
    elseif N == 3
        ID = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        RRT = r * transpose(r) # r ⊗ rᵗ
        return filter * (1 / (4π) / (d^3) * transpose(nx) * ((ID - 3 * RRT / d^2) * ny))
    end
end

################################################################################
################################# Helmholtz ####################################
################################################################################

"""
    struct Helmholtz{N,T}

Helmholtz equation in `N` dimensions: Δu + k²u = 0.
"""
struct Helmholtz{N,K} <: AbstractPDE{N}
    k::K
end

Helmholtz(; k, dim) = Helmholtz{dim,typeof(k)}(k)

function Base.show(io::IO, ::Helmholtz)
    # k = parameters(pde)
    return print(io, "Δu + k² u = 0")
end

parameters(pde::Helmholtz) = pde.k

default_kernel_eltype(::Helmholtz) = ComplexF64
default_density_eltype(::Helmholtz) = ComplexF64

function (SL::SingleLayerKernel{T,<:Helmholtz{N}})(target, source)::T where {N,T}
    x = coords(target)
    y = coords(source)
    k = parameters(SL)
    r = x - y
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (im / 4 * hankelh1(0, k * d))
    elseif N == 3
        return filter * (1 / (4π) / d * exp(im * k * d))
    end
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T,<:Helmholtz{N}})(target, source)::T where {N,T}
    x, y, ny = coords(target), coords(source), normal(source)
    k = parameters(DL)
    r = x - y
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        val = im * k / 4 / d * hankelh1(1, k * d) .* dot(r, ny)
        return filter * val
    elseif N == 3
        val = 1 / (4π) / d^2 * exp(im * k * d) * (-im * k + 1 / d) * dot(r, ny)
        return filter * val
    end
end

# Adjoint double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T,<:Helmholtz{N}})(target, source)::T where {N,T}
    x, y, nx = coords(target), coords(source), normal(target)
    k = parameters(ADL)
    r = x - y
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        val = -im * k / 4 / d * hankelh1(1, k * d) .* dot(r, nx)
        return filter * val
    elseif N == 3
        val = -1 / (4π) / d^2 * exp(im * k * d) * (-im * k + 1 / d) * dot(r, nx)
        return filter * val
    end
end

# Hypersingular kernel
function (HS::HyperSingularKernel{T,S})(target, source)::T where {T,S<:Helmholtz}
    x, y, nx, ny = coords(target), coords(source), normal(target), normal(source)
    N = ambient_dimension(pde(HS))
    k = parameters(pde(HS))
    r = x - y
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        RRT = r * transpose(r) # r ⊗ rᵗ
        # TODO: rewrite the operation below in a more clear/efficient way
        val =
            transpose(nx) * (
                (
                    -im * k^2 / 4 / d^2 * hankelh1(2, k * d) * RRT +
                    im * k / 4 / d * hankelh1(1, k * d) * I
                ) * ny
            )
        return filter * val
    elseif N == 3
        RRT = r * transpose(r) # r ⊗ rᵗ
        term1 = 1 / (4π) / d^2 * exp(im * k * d) * (-im * k + 1 / d) * I
        term2 =
            RRT / d * exp(im * k * d) / (4 * π * d^4) * (3 * (d * im * k - 1) + d^2 * k^2)
        val = transpose(nx) * (term1 + term2) * ny
        return filter * val
    end
end

############################ STOKES ############################3
struct Stokes{N,T} <: AbstractPDE{N}
    μ::T
end
Stokes(; μ, dim = 3) = Stokes{dim}(μ)
Stokes{N}(μ::T) where {N,T} = Stokes{N,T}(μ)

function Base.show(io::IO, pde::Stokes)
    return println(io, "μΔu -∇p = 0, ∇⋅u = 0")
end

parameters(s::Stokes) = s.μ

default_kernel_eltype(::Stokes{N}) where {N} = SMatrix{N,N,Float64,N * N}
default_density_eltype(::Stokes{N}) where {N} = SVector{N,Float64}

# Single Layer
function (SL::SingleLayerKernel{T,<:Stokes{N}})(target, source)::T where {N,T}
    μ = parameters(SL)
    x = coords(target)
    y = coords(source)
    r = x - y
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        γ = -log(d)
    elseif N == 3
        γ = 1 / d
    end
    return filter * (1 / (4π * (N - 1) * μ) * (γ * I + r * transpose(r) / d^N))
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T,<:Stokes{N}})(target, source)::T where {N,T}
    μ = parameters(DL)
    x = coords(target)
    y = coords(source)
    ny = normal(source)
    r = x - y
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (1 / π * dot(r, ny) / d^4 * r * transpose(r))
    elseif N == 3
        return filter * (3 / (4π) * dot(r, ny) / d^5 * r * transpose(r))
    end
end

# Double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T,<:Stokes{N}})(target, source)::T where {N,T}
    μ = parameters(ADL)
    x = coords(target)
    nx = normal(target)
    y = coords(source)
    r = x - y
    d = norm(r)
    filter = !(d ≤ SAME_POINT_TOLERANCE)
    if N == 2
        return filter * (-1 / π * dot(r, nx) / d^4 * r * transpose(r))
    elseif N == 3
        return filter * (-3 / (4π) * dot(r, nx) / d^5 * r * transpose(r))
    end
end

# TODO: Stokes hypersingular kernel

################################################################################
################################# Elastostatic #################################
################################################################################

"""
    struct Elastostatic{N,T} <: AbstractPDE{N}

Elastostatic equation in `N` dimensions: μΔu + (μ+λ)∇(∇⋅u) = 0. Note that the
displacement u is a vector of length `N` since this is a vectorial problem.
"""
struct Elastostatic{N,T} <: AbstractPDE{N}
    μ::T
    λ::T
end
Elastostatic(; μ, λ, dim) = Elastostatic{dim}(promote(μ, λ)...)
Elastostatic{N}(μ::T, λ::T) where {N,T} = Elastostatic{N,T}(μ, λ)

function Base.show(io::IO, pde::Elastostatic)
    return print(io, "μΔu + (μ+λ)∇(∇⋅u) = 0")
end

parameters(pde::Elastostatic) = pde.μ, pde.λ

default_kernel_eltype(::Elastostatic{N}) where {N} = SMatrix{N,N,Float64,N * N}
default_density_eltype(::Elastostatic{N}) where {N} = SVector{N,Float64}

function (SL::SingleLayerKernel{T,<:Elastostatic{N}})(target, source)::T where {N,T}
    μ, λ = parameters(pde(SL))
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
    μ, λ = parameters(pde(DL))
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
    μ, λ = parameters(pde(ADL))
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
    μ, λ = parameters(pde(HS))
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
