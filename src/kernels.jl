const PREDEFINED_OPERATORS = ["Laplace", "Helmholtz", "Stokes", "Yukawa"]

"""
    abstract type AbstractKernel

A kernel function `K`, callable as `K(target,source)`.

See also: [`SingleLayerKernel`](@ref),
[`DoubleLayerKernel`](@ref), [`AdjointDoubleLayerKernel`](@ref),
[`HyperSingularKernel`](@ref)
"""
abstract type AbstractKernel end

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

# convenient constructor for e.g. SingleLayerKernel(op) or DoubleLayerKernel(op)
function (::Type{K})(op::Op) where {Op, K <: AbstractKernel}
    return K{Op}(op)
end

operator(K::AbstractKernel) = K.op

"""
    apply_kernel(K::AbstractKernel, target, source, v)

Return `K(target, source) * v`, the action of the kernel on a density value `v`,
computed as `kernel_prefactor(K) * apply_kernel_unscaled(K, target, source, v)`.
Kernels should specialize [`apply_kernel_unscaled`](@ref) (and
[`kernel_prefactor`](@ref)), not this function.
"""
function apply_kernel(K::AbstractKernel, target, source, v)
    out = apply_kernel_unscaled(K, target, source, v)
    # `oftype` keeps the working precision of the data: prefactors may be exact
    # Float64 constants (e.g. 1/4π for Laplace) that must not promote Float32 data.
    return oftype(out, kernel_prefactor(K) * out)
end

"""
    kernel_prefactor(K::AbstractKernel)

A constant (independent of `target` and `source`, but possibly depending on kernel
parameters such as `μ`) that can be factored out of the kernel action:

    apply_kernel(K, target, source, v) ≈ kernel_prefactor(K) * apply_kernel_unscaled(K, target, source, v)

Matrix-free backends use this to scale the density by `kernel_prefactor` **once** per
matvec and evaluate only [`apply_kernel_unscaled`](@ref) in the `O(N²)` inner loop;
this also keeps kernel parameters like `μ` out of the device code, where per-pair
loads and divisions are costly. The fallback is `true` (the multiplicative identity),
paired with `apply_kernel_unscaled` falling back to `K(target, source) * v`; the two
must always be specialized as a consistent pair. The returned value must be a real
scalar, convertible to the precision of the quadrature weights.
"""
kernel_prefactor(K::AbstractKernel) = true

"""
    apply_kernel_unscaled(K::AbstractKernel, target, source, v)

The kernel action with the constant [`kernel_prefactor`](@ref) factored out; see its
docstring for the contract.

The generic fallback forms the kernel value and multiplies. Kernels whose value has
exploitable structure (e.g. the identity-plus-rank-one / rank-one Stokes kernels)
specialize this to compute the action **without ever assembling the matrix**, which
is both cheaper and lighter on registers — useful for matrix-free mat-vecs.
Specializations must agree with the fallback up to the prefactor, to machine
precision.
"""
apply_kernel_unscaled(K::AbstractKernel, target, source, v) = K(target, source) * v

"""
    struct SingleLayerKernel{Op} <: AbstractKernel

The free-space single-layer kernel (i.e. the fundamental solution) of an `Op <:
AbstractDifferentialOperator`.
"""
struct SingleLayerKernel{Op} <: AbstractKernel
    op::Op
end

function singularity_order(K::SingleLayerKernel)
    N = ambient_dimension(K.op)
    return 2 - N
end

"""
    struct DoubleLayerKernel{Op} <: AbstractKernel

Given an operator `Op`, construct its free-space double-layer kernel. This
corresponds to the `γ₁` trace of the [`SingleLayerKernel`](@ref). For operators
such as [`Laplace`](@ref) or [`Helmholtz`](@ref), this is simply the normal
derivative of the fundamental solution with respect to the source variable.
"""
struct DoubleLayerKernel{Op} <: AbstractKernel
    op::Op
end

function singularity_order(K::DoubleLayerKernel)
    N = ambient_dimension(K.op)
    return 1 - N
end

"""
    struct AdjointDoubleLayerKernel{Op} <: AbstractKernel

Given an operator `Op`, construct its free-space adjoint double-layer kernel.
This corresponds to the `transpose(γ₁,ₓ[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative of the fundamental
solution respect to the target variable.
"""
struct AdjointDoubleLayerKernel{Op} <: AbstractKernel
    op::Op
end

function singularity_order(K::AdjointDoubleLayerKernel)
    N = ambient_dimension(K.op)
    return 1 - N
end

"""
    struct HyperSingularKernel{Op} <: AbstractKernel

Given an operator `Op`, construct its free-space hypersingular kernel. This
corresponds to the `transpose(γ₁,ₓγ₁[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative respect to the target
variable of the `DoubleLayerKernel`.
"""
struct HyperSingularKernel{Op} <: AbstractKernel
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

function (SL::SingleLayerKernel{Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        v = @fastmath -log(d2) / 4 / π
    elseif N == 3
        v = @fastmath one(d2) / sqrt(d2) / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (DL::DoubleLayerKernel{Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    ny = normal(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        v = @fastmath dot(r, ny) / d2 / 2 / π
    elseif N == 3
        # single rsqrt: `1/d2` followed by `sqrt` costs two SFU ops on GPUs
        invd = @fastmath one(d2) / sqrt(d2)
        v = @fastmath dot(r, ny) * invd * invd * invd / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (ADL::AdjointDoubleLayerKernel{Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    nx = normal(target)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        v = @fastmath -dot(r, nx) / d2 / 2 / π
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        v = @fastmath -dot(r, nx) * invd * invd * invd / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (HS::HyperSingularKernel{Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    nx = normal(target)
    ny = normal(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    # nxᵀ(I - N*rrᵀ/d²)ny = nxdny - N*rdnx*rdny/d²
    nxdny = dot(nx, ny)
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    if N == 2
        id2 = @fastmath one(d2) / d2
        v = @fastmath id2 * (nxdny - 2 * rdnx * rdny * id2) / 2 / π
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        id2 = invd * invd
        v = @fastmath id2 * invd * (nxdny - 3 * rdnx * rdny * id2) / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

# Prefactored actions (see `kernel_prefactor`): same formulas as the kernels above
# with the constant pulled out; the sign of the adjoint double layer stays in the
# unscaled part so each body mirrors its kernel.
kernel_prefactor(::SingleLayerKernel{Laplace{N}}) where {N} = 1 / (4π)
kernel_prefactor(::DoubleLayerKernel{Laplace{N}}) where {N} = N == 2 ? 1 / (2π) : 1 / (4π)
function kernel_prefactor(::AdjointDoubleLayerKernel{Laplace{N}}) where {N}
    return N == 2 ? 1 / (2π) : 1 / (4π)
end
kernel_prefactor(::HyperSingularKernel{Laplace{N}}) where {N} = N == 2 ? 1 / (2π) : 1 / (4π)

function apply_kernel_unscaled(
        SL::SingleLayerKernel{Laplace{N}}, target, source, v,
    ) where {N}
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        out = @fastmath -log(d2) * v
    elseif N == 3
        out = @fastmath one(d2) / sqrt(d2) * v
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(out) : out
end

function apply_kernel_unscaled(
        DL::DoubleLayerKernel{Laplace{N}}, target, source, v,
    ) where {N}
    ny = normal(source)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        out = @fastmath dot(r, ny) / d2 * v
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        out = @fastmath dot(r, ny) * invd * invd * invd * v
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(out) : out
end

function apply_kernel_unscaled(
        ADL::AdjointDoubleLayerKernel{Laplace{N}}, target, source, v,
    ) where {N}
    nx = normal(target)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        out = @fastmath -dot(r, nx) / d2 * v
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        out = @fastmath -dot(r, nx) * invd * invd * invd * v
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(out) : out
end

function apply_kernel_unscaled(
        HS::HyperSingularKernel{Laplace{N}}, target, source, v,
    ) where {N}
    nx = normal(target)
    ny = normal(source)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    nxdny = dot(nx, ny)
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    if N == 2
        id2 = @fastmath one(d2) / d2
        out = @fastmath id2 * (nxdny - 2 * rdnx * rdny * id2) * v
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        id2 = invd * invd
        out = @fastmath id2 * invd * (nxdny - 3 * rdnx * rdny * id2) * v
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(out) : out
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

# same precision-preserving wrapper as `hankelh1`
besselk(n, x::T) where {T <: Real} = float(T)(Bessels.besselk(n, x))

function (SL::SingleLayerKernel{<:Yukawa{N, K}})(target, source) where {N, K}
    λ = SL.op.λ
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        d = sqrt(d2)
        v = besselk(0, λ * d) / 2 / π
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        d = d2 * invd
        v = @fastmath exp(-λ * d) * invd / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (DL::DoubleLayerKernel{<:Yukawa{N, K}})(target, source) where {N, K}
    ny = normal(source)
    λ = DL.op.λ
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    rdny = dot(r, ny)
    if N == 2
        d = sqrt(d2)
        v = λ * besselk(1, λ * d) * rdny / d / 2 / π
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        d = d2 * invd
        id2 = invd * invd
        v = @fastmath exp(-λ * d) * (λ + invd) * rdny * id2 / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (ADL::AdjointDoubleLayerKernel{<:Yukawa{N, K}})(target, source) where {N, K}
    nx = normal(target)
    λ = ADL.op.λ
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    rdnx = dot(r, nx)
    if N == 2
        d = sqrt(d2)
        v = -λ * besselk(1, λ * d) * rdnx / d / 2 / π
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        d = d2 * invd
        id2 = invd * invd
        v = @fastmath -exp(-λ * d) * (λ + invd) * rdnx * id2 / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (HS::HyperSingularKernel{<:Yukawa{N, K}})(target, source) where {N, K}
    nx, ny = normal(target), normal(source)
    λ = HS.op.λ
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    # nxᵀ(a*rrᵀ + b*I)ny = a*rdnx*rdny + b*nxdny
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    nxdny = dot(nx, ny)
    if N == 2
        d = sqrt(d2)
        k1 = besselk(1, λ * d)
        k2 = besselk(2, λ * d)
        # avoid `2π`: it is a Float64 and would promote single-precision inputs
        a = -λ^2 / (2 * d2) / π * k2
        b = λ / (2 * d) / π * k1
        v = a * rdnx * rdny + b * nxdny
    elseif N == 3
        @fastmath begin
            invd = one(d2) / sqrt(d2)
            d = d2 * invd
            id2 = invd * invd
            emld = exp(-λ * d)
            b = emld * id2 / 4 / π * (λ + invd)
            a = emld * id2 * id2 * invd / 4 / π * (-3 * (d * λ + 1) - d2 * λ^2)
            v = a * rdnx * rdny + b * nxdny
        end
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
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

# Bessels.jl evaluates some order/argument ranges in Float64, so convert back to the
# input precision to keep kernel evaluations type-stable
hankelh1(n, x::T) where {T <: Real} = Complex{float(T)}(Bessels.hankelh1(n, x))
hankelh1(n, x::Complex) = SpecialFunctions.hankelh1(n, x)

function (SL::SingleLayerKernel{<:Helmholtz{N}})(target, source) where {N}
    k = SL.op.k
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        d = sqrt(d2)
        # leading with `im / 4` (a ComplexF64) would promote single-precision inputs
        v = im * hankelh1(0, k * d) / 4
        return d2 ≤ tol * tol ? zero(v) : v
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        d = d2 * invd
        v = @fastmath cis(k * d) * invd / 4 / π
        return d2 ≤ tol * tol ? zero(v) : v
    else
        notimplemented()
    end
end

function (DL::DoubleLayerKernel{<:Helmholtz{N}})(target, source) where {N}
    ny = normal(source)
    k = DL.op.k
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    rdny = dot(r, ny)
    if N == 2
        d = sqrt(d2)
        v = im * k / (4d) * hankelh1(1, k * d) * rdny
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        d = d2 * invd
        id2 = invd * invd
        v = @fastmath cis(k * d) * (-im * k + invd) * rdny * id2 / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (ADL::AdjointDoubleLayerKernel{<:Helmholtz{N}})(target, source) where {N}
    nx = normal(target)
    k = ADL.op.k
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    rdnx = dot(r, nx)
    if N == 2
        d = sqrt(d2)
        v = -im * k / (4d) * hankelh1(1, k * d) * rdnx
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        d = d2 * invd
        id2 = invd * invd
        v = @fastmath -cis(k * d) * (-im * k + invd) * rdnx * id2 / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (HS::HyperSingularKernel{<:Helmholtz{N}})(target, source) where {N}
    nx, ny = normal(target), normal(source)
    k = HS.op.k
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    # nxᵀ(a*rrᵀ + b*I)ny = a*rdnx*rdny + b*nxdny
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    nxdny = dot(nx, ny)
    if N == 2
        d = sqrt(d2)
        h1 = hankelh1(1, k * d)
        h2 = hankelh1(2, k * d)
        a = -im * k^2 / (4 * d2) * h2
        b = im * k / (4 * d) * h1
        v = a * rdnx * rdny + b * nxdny
    elseif N == 3
        @fastmath begin
            invd = one(d2) / sqrt(d2)
            d = d2 * invd
            id2 = invd * invd
            eikd = cis(k * d)
            b = eikd * id2 / 4 / π * (-im * k + invd)
            a = eikd * id2 * id2 * invd / 4 / π * (3 * (d * im * k - 1) + d2 * k^2)
            v = a * rdnx * rdny + b * nxdny
        end
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
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

function (SL::SingleLayerKernel{<:Stokes{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    μ = SL.op.μ
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    RRT = r * transpose(r)
    if N == 2
        γ = @fastmath -log(d2) / 2
        invd2 = @fastmath one(d2) / d2
        v = (γ * I + RRT * invd2) / (μ * 4 * π * (N - 1))
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        invd3 = @fastmath invd * invd * invd
        v = (invd * I + RRT * invd3) / (μ * 4 * π * (N - 1))
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (DL::DoubleLayerKernel{<:Stokes{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    ny = normal(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    RRT = r * transpose(r)
    rdny = dot(r, ny)
    if N == 2
        id2 = @fastmath one(d2) / d2
        c = @fastmath id2 * id2 / π * rdny
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        id2 = invd * invd
        c = @fastmath 3 * id2 * id2 * invd / 4 / π * rdny
    else
        notimplemented()
    end
    v = c * RRT
    return d2 ≤ tol * tol ? zero(v) : v
end

function (ADL::AdjointDoubleLayerKernel{<:Stokes{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    nx = normal(target)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    RRT = r * transpose(r)
    rdnx = dot(r, nx)
    if N == 2
        id2 = @fastmath one(d2) / d2
        c = @fastmath -id2 * id2 / π * rdnx
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        id2 = invd * invd
        c = @fastmath -3 * id2 * id2 * invd / 4 / π * rdnx
    else
        notimplemented()
    end
    v = c * RRT
    return d2 ≤ tol * tol ? zero(v) : v
end

function (HS::HyperSingularKernel{<:Stokes{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    μ = HS.op.μ
    nx = normal(target)
    ny = normal(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    invd = @fastmath one(d2) / sqrt(d2)
    id2 = invd * invd
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    nxdny = dot(nx, ny)
    # Stokes is the incompressible limit (ν = 1/2) of the Elastostatic operator; the
    # hypersingular kernel below is the Elastostatic one specialized to ν = 1/2 (the
    # (1 - 2ν) terms vanish and 1/(1 - ν) stays finite). The value is assembled as
    # c * (r⊗vr + nx⊗vnx + ny⊗vny + a_diag*I).
    if N == 2
        c = μ * id2 / π                                  # μ/(π d²)
        α = 2 * rdny * id2
        vr = (-4α * rdnx + nxdny) * id2 * r + α * nx / 2
        vnx = ny
        vny = rdnx * id2 * r
        a_diag = α * rdnx / 2
    elseif N == 3
        c = μ * id2 * invd / 2 / π                       # μ/(2π d³)
        α = 3 * rdny * id2
        vr = (-5α * rdnx + 3 * nxdny / 2) * id2 * r + α * nx / 2
        vnx = ny
        vny = 3 * rdnx * id2 * r / 2
        a_diag = α * rdnx / 2
    else
        notimplemented()
    end
    v = c * (r * transpose(vr) + nx * transpose(vnx) + ny * transpose(vny) + a_diag * I)
    return d2 ≤ tol * tol ? zero(v) : v
end

# Prefactored actions (see `kernel_prefactor`). Beyond saving a few operations, this
# keeps `μ` out of the inner loop of matrix-free backends: a division by (or even a
# multiplication with) a kernel-struct field per pair measurably slows GPU kernels.
# NOTE: a runtime float (`μ`) must sit lexically first in each product so the
# `Int`/`Irrational` literals are demoted to its precision (else `4 * π::Float64`
# would silently promote Float32 data).
function kernel_prefactor(SL::SingleLayerKernel{<:Stokes{N}}) where {N}
    μ = SL.op.μ
    return inv(μ * 4 * (N - 1) * π)
end
kernel_prefactor(::DoubleLayerKernel{<:Stokes{N}}) where {N} = N == 2 ? 1 / π : 3 / (4π)
function kernel_prefactor(::AdjointDoubleLayerKernel{<:Stokes{N}}) where {N}
    return N == 2 ? 1 / π : 3 / (4π)
end
function kernel_prefactor(HS::HyperSingularKernel{<:Stokes{N}}) where {N}
    μ = HS.op.μ
    return N == 2 ? μ / π : μ / 2 / π
end

function apply_kernel_unscaled(
        SL::SingleLayerKernel{<:Stokes{N}}, target, source, v,
    ) where {N}
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    rdv = dot(r, v)
    if N == 2
        γ = @fastmath -log(d2) / 2
        invd2 = @fastmath one(d2) / d2
        out = γ * v + (invd2 * rdv) * r
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        invd3 = @fastmath invd * invd * invd
        out = invd * v + (invd3 * rdv) * r
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(out) : out
end

function apply_kernel_unscaled(
        _DL::DoubleLayerKernel{<:Stokes{N}}, target, source, v,
    ) where {N}
    ny = normal(source)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    rdny = dot(r, ny)
    rdv = dot(r, v)
    if N == 2
        id2 = @fastmath one(d2) / d2
        c = @fastmath id2 * id2 * rdny * rdv
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        id2 = invd * invd
        c = @fastmath id2 * id2 * invd * rdny * rdv
    else
        notimplemented()
    end
    out = c * r
    return d2 ≤ tol * tol ? zero(out) : out
end

function apply_kernel_unscaled(
        ADL::AdjointDoubleLayerKernel{<:Stokes{N}}, target, source, v,
    ) where {N}
    nx = normal(target)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    rdnx = dot(r, nx)
    rdv = dot(r, v)
    if N == 2
        id2 = @fastmath one(d2) / d2
        c = @fastmath -id2 * id2 * rdnx * rdv
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        id2 = invd * invd
        c = @fastmath -id2 * id2 * invd * rdnx * rdv
    else
        notimplemented()
    end
    out = c * r
    return d2 ≤ tol * tol ? zero(out) : out
end

function apply_kernel_unscaled(
        HS::HyperSingularKernel{<:Stokes{N}}, target, source, v,
    ) where {N}
    nx = normal(target)
    ny = normal(source)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    invd = @fastmath one(d2) / sqrt(d2)
    id2 = invd * invd
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    nxdny = dot(nx, ny)
    if N == 2
        c = id2
        α = 2 * rdny * id2
        vr = (-4α * rdnx + nxdny) * id2 * r + α * nx / 2
        vnx = ny
        vny = rdnx * id2 * r
        a_diag = α * rdnx / 2
    elseif N == 3
        c = id2 * invd
        α = 3 * rdny * id2
        vr = (-5α * rdnx + 3 * nxdny / 2) * id2 * r + α * nx / 2
        vnx = ny
        vny = 3 * rdnx * id2 * r / 2
        a_diag = α * rdnx / 2
    else
        notimplemented()
    end
    out = c * (dot(vr, v) * r + dot(vnx, v) * nx + dot(vny, v) * ny + a_diag * v)
    return d2 ≤ tol * tol ? zero(out) : out
end

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

function (SL::SingleLayerKernel{<:Elastostatic{N}})(target, source) where {N}
    μ, λ = SL.op.μ, SL.op.λ
    ν = λ / (2 * (μ + λ))
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    RRT = r * transpose(r) # r ⊗ rᵗ
    if N == 2
        id2 = @fastmath one(d2) / d2
        v = (-(3 - 4 * ν) * log(d2) / 2 * I + RRT * id2) / (μ * 8 * π * (1 - ν))
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        id2 = invd * invd
        v = invd * ((3 - 4 * ν) * I + RRT * id2) / (μ * 16 * π * (1 - ν))
    else
        notimplemented()
    end
    return iszero(d2) ? zero(v) : v
end

function (DL::DoubleLayerKernel{<:Elastostatic{N}})(target, source) where {N}
    μ, λ = DL.op.μ, DL.op.λ
    ν = λ / (2 * (μ + λ))
    ny = normal(source)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    invd = @fastmath one(d2) / sqrt(d2)
    id2 = invd * invd
    RRT = r * transpose(r)
    drdn = -dot(r, ny) * invd
    ν1 = 1 - 2ν
    asym = r * transpose(ny) - ny * transpose(r)
    if N == 2
        v = -invd * (drdn * (ν1 * I + 2 * id2 * RRT) + (ν1 * invd) * asym) / 4 / π / (1 - ν)
    elseif N == 3
        v = -id2 * (drdn * (ν1 * I + 3 * id2 * RRT) + (ν1 * invd) * asym) / 8 / π / (1 - ν)
    else
        notimplemented()
    end
    return iszero(d2) ? zero(v) : v
end

function (ADL::AdjointDoubleLayerKernel{<:Elastostatic{N}})(target, source) where {N}
    μ, λ = ADL.op.μ, ADL.op.λ
    ν = λ / (2 * (μ + λ))
    nx = normal(target)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    invd = @fastmath one(d2) / sqrt(d2)
    id2 = invd * invd
    RRT = r * transpose(r)
    drdn = -dot(r, nx) * invd
    ν1 = 1 - 2ν
    # ADL = -transpose(DL with ny→nx), which flips the sign of the antisymmetric part
    asym = nx * transpose(r) - r * transpose(nx)
    if N == 2
        v = invd * (drdn * (ν1 * I + 2 * id2 * RRT) + (ν1 * invd) * asym) / 4 / π / (1 - ν)
    elseif N == 3
        v = id2 * (drdn * (ν1 * I + 3 * id2 * RRT) + (ν1 * invd) * asym) / 8 / π / (1 - ν)
    else
        notimplemented()
    end
    return iszero(d2) ? zero(v) : v
end

function (HS::HyperSingularKernel{<:Elastostatic{N}})(target, source) where {N}
    μ, λ = HS.op.μ, HS.op.λ
    ν = λ / (2 * (μ + λ))
    nx = normal(target)
    ny = normal(source)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    invd = @fastmath one(d2) / sqrt(d2)
    id2 = invd * invd
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    nxdny = dot(nx, ny)
    if N == 2
        c = μ * id2 / 2 / π / (1 - ν)
        α = 2 * rdny * id2
        # Decompose as: c * (r⊗vr + nx⊗vnx + ny⊗vny + a_diag * I)
        vr = (-4α * rdnx + 2ν * nxdny) * id2 * r + α * ν * nx + (1 - 2ν) * 2 * rdnx * id2 * ny
        vnx = (1 - 2ν) * α * r - (1 - 4ν) * ny
        vny = 2ν * rdnx * id2 * r + (1 - 2ν) * nx
        a_diag = α * ν * rdnx + (1 - 2ν) * nxdny
    elseif N == 3
        c = μ * id2 * invd / 4 / π / (1 - ν)
        α = 3 * rdny * id2
        vr = (-5α * rdnx + 3ν * nxdny) * id2 * r + α * ν * nx + (1 - 2ν) * 3 * rdnx * id2 * ny
        vnx = (1 - 2ν) * α * r - (1 - 4ν) * ny
        vny = 3ν * rdnx * id2 * r + (1 - 2ν) * nx
        a_diag = α * ν * rdnx + (1 - 2ν) * nxdny
    else
        notimplemented()
    end
    v = c * (r * transpose(vr) + nx * transpose(vnx) + ny * transpose(vny) + a_diag * I)
    return iszero(d2) ? zero(v) : v
end

# Prefactored actions (see `kernel_prefactor`). The Poisson-ratio coefficients
# (1 - 2ν etc.) appear non-multiplicatively, so only the overall constant factors out;
# ν itself is still computed in the unscaled action. NOTE: the runtime floats (`μ`,
# `1 - ν`) sit lexically first in each product so the `Int`/`Irrational` literals are
# demoted to their precision.
function kernel_prefactor(SL::SingleLayerKernel{<:Elastostatic{N}}) where {N}
    μ, λ = SL.op.μ, SL.op.λ
    ν = λ / (2 * (μ + λ))
    return N == 2 ? inv(μ * 8 * (1 - ν) * π) : inv(μ * 16 * (1 - ν) * π)
end
function kernel_prefactor(DL::DoubleLayerKernel{<:Elastostatic{N}}) where {N}
    μ, λ = DL.op.μ, DL.op.λ
    ν = λ / (2 * (μ + λ))
    return N == 2 ? inv((1 - ν) * 4 * π) : inv((1 - ν) * 8 * π)
end
function kernel_prefactor(ADL::AdjointDoubleLayerKernel{<:Elastostatic{N}}) where {N}
    μ, λ = ADL.op.μ, ADL.op.λ
    ν = λ / (2 * (μ + λ))
    return N == 2 ? inv((1 - ν) * 4 * π) : inv((1 - ν) * 8 * π)
end
function kernel_prefactor(HS::HyperSingularKernel{<:Elastostatic{N}}) where {N}
    μ, λ = HS.op.μ, HS.op.λ
    ν = λ / (2 * (μ + λ))
    return N == 2 ? μ / (2 * (1 - ν) * π) : μ / (4 * (1 - ν) * π)
end

function apply_kernel_unscaled(SL::SingleLayerKernel{<:Elastostatic{N}}, target, source, v) where {N}
    μ, λ = SL.op.μ, SL.op.λ
    ν = λ / (2 * (μ + λ))
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    rdv = dot(r, v)
    if N == 2
        id2 = @fastmath one(d2) / d2
        out = -(3 - 4 * ν) * log(d2) / 2 * v + (id2 * rdv) * r
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        id2 = invd * invd
        out = invd * ((3 - 4 * ν) * v + (id2 * rdv) * r)
    else
        notimplemented()
    end
    return iszero(d2) ? zero(out) : out
end

function apply_kernel_unscaled(DL::DoubleLayerKernel{<:Elastostatic{N}}, target, source, v) where {N}
    μ, λ = DL.op.μ, DL.op.λ
    ν = λ / (2 * (μ + λ))
    ny = normal(source)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    invd = @fastmath one(d2) / sqrt(d2)
    id2 = invd * invd
    rdv = dot(r, v)
    drdn = -dot(r, ny) * invd
    ν1 = 1 - 2ν
    asym = dot(ny, v) * r - rdv * ny                # (r⊗ny − ny⊗r)·v
    if N == 2
        out = -invd * (drdn * (ν1 * v + 2 * id2 * rdv * r) + (ν1 * invd) * asym)
    elseif N == 3
        out = -id2 * (drdn * (ν1 * v + 3 * id2 * rdv * r) + (ν1 * invd) * asym)
    else
        notimplemented()
    end
    return iszero(d2) ? zero(out) : out
end

function apply_kernel_unscaled(ADL::AdjointDoubleLayerKernel{<:Elastostatic{N}}, target, source, v) where {N}
    μ, λ = ADL.op.μ, ADL.op.λ
    ν = λ / (2 * (μ + λ))
    nx = normal(target)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    invd = @fastmath one(d2) / sqrt(d2)
    id2 = invd * invd
    rdv = dot(r, v)
    drdn = -dot(r, nx) * invd
    ν1 = 1 - 2ν
    asym = rdv * nx - dot(nx, v) * r                # (nx⊗r − r⊗nx)·v
    if N == 2
        out = invd * (drdn * (ν1 * v + 2 * id2 * rdv * r) + (ν1 * invd) * asym)
    elseif N == 3
        out = id2 * (drdn * (ν1 * v + 3 * id2 * rdv * r) + (ν1 * invd) * asym)
    else
        notimplemented()
    end
    return iszero(d2) ? zero(out) : out
end

function apply_kernel_unscaled(HS::HyperSingularKernel{<:Elastostatic{N}}, target, source, v) where {N}
    μ, λ = HS.op.μ, HS.op.λ
    ν = λ / (2 * (μ + λ))
    nx = normal(target)
    ny = normal(source)
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    invd = @fastmath one(d2) / sqrt(d2)
    id2 = invd * invd
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    nxdny = dot(nx, ny)
    if N == 2
        c = id2
        α = 2 * rdny * id2
        vr = (-4α * rdnx + 2ν * nxdny) * id2 * r + α * ν * nx + (1 - 2ν) * 2 * rdnx * id2 * ny
        vnx = (1 - 2ν) * α * r - (1 - 4ν) * ny
        vny = 2ν * rdnx * id2 * r + (1 - 2ν) * nx
        a_diag = α * ν * rdnx + (1 - 2ν) * nxdny
    elseif N == 3
        c = id2 * invd
        α = 3 * rdny * id2
        vr = (-5α * rdnx + 3ν * nxdny) * id2 * r + α * ν * nx + (1 - 2ν) * 3 * rdnx * id2 * ny
        vnx = (1 - 2ν) * α * r - (1 - 4ν) * ny
        vny = 3ν * rdnx * id2 * r + (1 - 2ν) * nx
        a_diag = α * ν * rdnx + (1 - 2ν) * nxdny
    else
        notimplemented()
    end
    out = c * (dot(vr, v) * r + dot(vnx, v) * nx + dot(vny, v) * ny + a_diag * v)
    return iszero(d2) ? zero(out) : out
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

function (SL::SingleLayerKernel{<:LaplacePeriodic1D{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    l = SL.op.period
    if N == 2
        d2 = sin(π / l * r[1])^2 + sinh(π / l * r[2])^2
        out = -1 / 4π * log(d2)
        return d2 ≤ SAME_POINT_TOLERANCE ? zero(out) : out
    else
        error("Single layer kernel for LaplacePeriodic1D not implemented in $N dimensions")
    end
end

function (DL::DoubleLayerKernel{<:LaplacePeriodic1D{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    ny = normal(source)
    if N == 2
        l = DL.op.period
        s = sin(π / l * r[1])
        sh = sinh(π / l * r[2])
        d2 = s^2 + sh^2
        out = 1 / (4π * d2) * (2 * π / l * s * cos(π / l * r[1]) * ny[1] + 2 * π / l * sh * cosh(π / l * r[2]) * ny[2])
        return d2 ≤ SAME_POINT_TOLERANCE ? zero(out) : out
    else
        error("Double layer kernel for LaplacePeriodic1D not implemented in $N dimensions")
    end
end

function (ADL::AdjointDoubleLayerKernel{<:LaplacePeriodic1D{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    nx = normal(target)
    if N == 2
        l = ADL.op.period
        s = sin(π / l * r[1])
        sh = sinh(π / l * r[2])
        d2 = s^2 + sh^2
        out = -1 / (4π * d2) * (2 * π / l * s * cos(π / l * r[1]) * nx[1] + 2 * π / l * sh * cosh(π / l * r[2]) * nx[2])
        return d2 ≤ SAME_POINT_TOLERANCE ? zero(out) : out
    else
        error(
            "Adjoint double layer kernel for LaplacePeriodic1D not implemented in $N dimensions",
        )
    end
end

function (HS::HyperSingularKernel{<:LaplacePeriodic1D{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
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
