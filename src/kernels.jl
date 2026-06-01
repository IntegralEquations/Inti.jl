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

"""
    struct GradientSingleLayerKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space gradient single-layer kernel.
This evaluates the gradient of the fundamental solution with respect to the target variable.
"""
struct GradientSingleLayerKernel{T, Op} <: AbstractKernel{T}
    op::Op
end

function GradientSingleLayerKernel(op::AbstractDifferentialOperator{N}, ::Type{T} = SVector{N, default_kernel_eltype(op)}) where {N, T}
    return GradientSingleLayerKernel{T, typeof(op)}(op)
end

function singularity_order(K::GradientSingleLayerKernel)
    N = ambient_dimension(K.op)
    return 1 - N
end

"""
    struct GradientDoubleLayerKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space gradient double-layer kernel.
This evaluates the gradient of the double-layer kernel with respect to the target variable.
"""
struct GradientDoubleLayerKernel{T, Op} <: AbstractKernel{T}
    op::Op
end

function GradientDoubleLayerKernel(op::AbstractDifferentialOperator{N}, ::Type{T} = SVector{N, default_kernel_eltype(op)}) where {N, T}
    return GradientDoubleLayerKernel{T, typeof(op)}(op)
end

function singularity_order(K::GradientDoubleLayerKernel)
    N = ambient_dimension(K.op)
    return -N
end

"""
    struct SourceGradientSingleLayerKernel{T,Op} <: AbstractKernel{T}

Gradient of the single-layer kernel `G(x,y)` with respect to the source variable `y`,
i.e. ``\\nabla_y G(x,y)``. The value is returned as the (row) covector
`transpose(∇_yG)`, so that `K(x,y) * g(y)` contracts an `SVector` density to a
scalar. It is the source-variable counterpart of [`GradientSingleLayerKernel`](@ref)
(which is the target gradient ``\\nabla_x G``), and since ``\\nabla_y G = -\\nabla_x
G`` it is built by negating the latter's evaluation.

The associated volume integral operator is ``\\mathcal{W}[g](x) = -\\int_\\Omega
\\nabla_y G(x,y) \\cdot g(y)\\,dy``; note the leading minus sign, so the
discrete operator that evaluates ``\\mathcal{W}`` is the negative of
`IntegralOperator(SourceGradientSingleLayerKernel(op), ...)`.
"""
struct SourceGradientSingleLayerKernel{T, Op} <: AbstractKernel{T}
    op::Op
end

function SourceGradientSingleLayerKernel(
        op::AbstractDifferentialOperator{N},
        ::Type{S} = default_kernel_eltype(op),
    ) where {N, S}
    T = Transpose{S, SVector{N, S}}
    return SourceGradientSingleLayerKernel{T, typeof(op)}(op)
end

function singularity_order(K::SourceGradientSingleLayerKernel)
    N = ambient_dimension(K.op)
    return 1 - N
end

function (K::SourceGradientSingleLayerKernel)(
        target,
        source,
        r = coords(target) - coords(source),
    )
    # ∇_y G(x,y) = -∇_x G(x,y); returned as a row covector to contract a vector density
    gx = GradientSingleLayerKernel(K.op)(target, source, r)
    return transpose(-gx)
end

# `zero` for the covector element type, needed when assembling the sparse VDIM
# correction whose entries map an `SVector` density to a scalar.
Base.zero(::Type{Transpose{T, SVector{N, T}}}) where {N, T} = transpose(zero(SVector{N, T}))

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
    ) where {N, T}
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
    ) where {N, T}
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
    ) where {N, T}
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
    if N == 2
        return 1 / (2π) / (d^2) * transpose(nx) * ((I - 2 * r * transpose(r) / d^2) * ny)
    elseif N == 3
        return 1 / (4π) / (d^3) * transpose(nx) * ((I - 3 * r * transpose(r) / d^2) * ny)
    end
end

function (GSL::GradientSingleLayerKernel{T, Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return -1 / (2π) / (d^2) * r
    elseif N == 3
        return -1 / (4π) / (d^3) * r
    end
end

function (GDL::GradientDoubleLayerKernel{T, Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    ny = normal(source)
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return 1 / (2π) / (d^2) * (ny - 2 * dot(r, ny) / d^2 * r)
    elseif N == 3
        return 1 / (4π) / (d^3) * (ny - 3 * dot(r, ny) / d^2 * r)
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
        return λ / (2 * π * d) * Bessels.besselk(1, λ * d) .* dot(r, ny)
    elseif N == 3
        return 1 / (4π) / d^2 * exp(-λ * d) * (λ + 1 / d) * dot(r, ny)
    end
end

function (ADL::AdjointDoubleLayerKernel{T, <:Yukawa{N, K}})(target, source)::T where {N, T, K}
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

function (HS::HyperSingularKernel{T, <:Yukawa{N, K}})(target, source)::T where {N, T, K}
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
        val = im * k / 4 / d * hankelh1(1, k * d) .* dot(r, ny)
        return val
    elseif N == 3
        val = 1 / (4π) / d^2 * exp(im * k * d) * (-im * k + 1 / d) * dot(r, ny)
        return val
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
        val = -im * k / 4 / d * hankelh1(1, k * d) .* dot(r, nx)
        return val
    elseif N == 3
        val = -1 / (4π) / d^2 * exp(im * k * d) * (-im * k + 1 / d) * dot(r, nx)
        return val
    end
end

# Hypersingular kernel
function (HS::HyperSingularKernel{T, <:Helmholtz{N}})(target, source)::T where {N, T}
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

function (GSL::GradientSingleLayerKernel{T, <:Helmholtz{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    k = GSL.op.k
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    if N == 2
        return -im * k / 4 / d * hankelh1(1, k * d) * r
    elseif N == 3
        return 1 / (4π) / d^2 * exp(im * k * d) * (im * k - 1 / d) * r
    end
end

function (GDL::GradientDoubleLayerKernel{T, <:Helmholtz{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    ny = normal(source)
    k = GDL.op.k
    d = norm(r)
    d ≤ SAME_POINT_TOLERANCE && return zero(T)
    rdotny = dot(r, ny)
    if N == 2
        return im * k / (4 * d) * hankelh1(1, k * d) * ny -
            im * k^2 / (4 * d^2) * hankelh1(2, k * d) * r * rdotny
    elseif N == 3
        pref = 1 / (4π) / d^3 * exp(im * k * d)
        return pref * ((1 - im * k * d) * ny + (k^2 * d^2 + 3 * im * k * d - 3) / d^2 * r * rdotny)
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
function (ADL::AdjointDoubleLayerKernel{T, <:Stokes{N}})(target, source)::T where {N, T}
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

function (ADL::AdjointDoubleLayerKernel{T, <:Elastostatic{N}})(target, source)::T where {N, T}
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

function (HS::HyperSingularKernel{T, <:Elastostatic{N}})(target, source) where {N, T}
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

################################################################################
############################ ELASTOSTATIC GRADIENT ############################
################################################################################

# Algebra needed for SVector{N, <:SMatrix} output type.
# These products arise when gradient kernels for vector-valued operators (Elastostatic, Stokes)
# act on matrix-valued densities (DIM method) or vector-valued densities (final application).
function Base.:*(t::SVector{N, M}, m::SMatrix{P, Q}) where {N, P, Q, M <: SMatrix{P, Q}}
    return typeof(t)(ntuple(k -> t[k] * m, N))
end
function Base.:*(t::SVector{N, M}, v::SVector{P}) where {N, P, M <: SMatrix{P, P}}
    return SMatrix{P, N}(hcat(ntuple(k -> t[k] * v, N)...))
end

function GradientSingleLayerKernel(op::Elastostatic{N}) where {N}
    T = SVector{N, SMatrix{N, N, Float64, N * N}}
    return GradientSingleLayerKernel{T, typeof(op)}(op)
end

function GradientDoubleLayerKernel(op::Elastostatic{N}) where {N}
    T = SVector{N, SMatrix{N, N, Float64, N * N}}
    return GradientDoubleLayerKernel{T, typeof(op)}(op)
end

function (K::GradientSingleLayerKernel{T, <:Elastostatic{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    μ, λ = K.op.μ, K.op.λ
    ν = λ / (2 * (μ + λ))
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r * r'
    SM = SMatrix{N, N, Float64, N * N}
    if N == 2
        C = 1 / (8π * μ * (1 - ν))
        return SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (-(3 - 4ν) * r[k] / d^2 * I + (ek * r' + r * ek') / d^2 - 2 * r[k] * RRT / d^4))
            end
        )
    elseif N == 3
        C = 1 / (16π * μ * (1 - ν))
        return SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (-(3 - 4ν) * r[k] / d^3 * I + (ek * r' + r * ek') / d^3 - 3 * r[k] * RRT / d^5))
            end
        )
    end
end

function (K::GradientDoubleLayerKernel{T, <:Elastostatic{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    μ, λ = K.op.μ, K.op.λ
    ν = λ / (2 * (μ + λ))
    ny = normal(source)
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r * r'
    qr = dot(ny, r)
    B = r * ny' - ny * r'
    SM = SMatrix{N, N, Float64, N * N}
    if N == 2
        C = 1 / (4π * (1 - ν))
        A = (1 - 2ν) * I + 2 * RRT / d^2
        return SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(
                    C * (
                        (ny[k] / d^2 - 2 * qr * r[k] / d^4) * A
                            + 2 * qr / d^4 * (ek * r' + r * ek')
                            - 4 * qr * r[k] * RRT / d^6
                            - (1 - 2ν) / d^2 * (ek * ny' - ny * ek')
                            + 2 * r[k] * (1 - 2ν) / d^4 * B
                    )
                )
            end
        )
    elseif N == 3
        C = 1 / (8π * (1 - ν))
        A = (1 - 2ν) * I + 3 * RRT / d^2
        return SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(
                    C * (
                        (ny[k] / d^3 - 3 * qr * r[k] / d^5) * A
                            + 3 * qr / d^5 * (ek * r' + r * ek')
                            - 6 * qr * r[k] * RRT / d^7
                            - (1 - 2ν) / d^3 * (ek * ny' - ny * ek')
                            + 3 * r[k] * (1 - 2ν) / d^5 * B
                    )
                )
            end
        )
    end
end

################################################################################
################################### STOKES GRADIENT ############################
################################################################################

function GradientSingleLayerKernel(op::Stokes{N}) where {N}
    T = SVector{N, SMatrix{N, N, Float64, N * N}}
    return GradientSingleLayerKernel{T, typeof(op)}(op)
end

function GradientDoubleLayerKernel(op::Stokes{N}) where {N}
    T = SVector{N, SMatrix{N, N, Float64, N * N}}
    return GradientDoubleLayerKernel{T, typeof(op)}(op)
end

function (K::GradientSingleLayerKernel{T, <:Stokes{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    μ = K.op.μ
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r * r'
    SM = SMatrix{N, N, Float64, N * N}
    if N == 2
        C = 1 / (4π * μ)
        return SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (-r[k] / d^2 * I + (ek * r' + r * ek') / d^2 - 2 * r[k] * RRT / d^4))
            end
        )
    elseif N == 3
        C = 1 / (8π * μ)
        return SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (-r[k] / d^3 * I + (ek * r' + r * ek') / d^3 - 3 * r[k] * RRT / d^5))
            end
        )
    end
end

function (K::GradientDoubleLayerKernel{T, <:Stokes{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N, T}
    ny = normal(source)
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r * r'
    qr = dot(ny, r)
    SM = SMatrix{N, N, Float64, N * N}
    if N == 2
        C = 1 / π
        return SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (ny[k] * RRT / d^4 + qr * (ek * r' + r * ek') / d^4 - 4 * qr * r[k] * RRT / d^6))
            end
        )
    elseif N == 3
        C = 3 / (4π)
        return SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (ny[k] * RRT / d^5 + qr * (ek * r' + r * ek') / d^5 - 5 * qr * r[k] * RRT / d^7))
            end
        )
    end
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
