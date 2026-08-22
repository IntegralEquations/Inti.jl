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

# convenient constructor for e.g. SingleLayerKernel(op) or DoubleLayerKernel(op)
function (::Type{K})(op::Op) where {Op, K <: AbstractKernel}
    return K{Op}(op)
end

operator(K::AbstractKernel) = K.op

"""
    apply_kernel(K::AbstractKernel, target, source, v)

Return `K(target, source) * v`, the action of the kernel on a density value `v`.

The generic fallback simply forms the kernel value and multiplies. Kernels whose
value has exploitable structure (e.g. the identity-plus-rank-one / rank-one Stokes
kernels) specialize this to compute the action **without ever assembling the
matrix**, which is both cheaper and lighter on registers — useful for matrix-free
mat-vecs. Specializations must agree with the fallback to machine precision.
"""
apply_kernel(K::AbstractKernel, target, source, v) = K(target, source) * v

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

"""
    struct GradientSingleLayerKernel{Op} <: AbstractKernel

Given an operator `Op`, construct its free-space gradient single-layer kernel.
This evaluates the gradient of the fundamental solution with respect to the target variable.
"""
struct GradientSingleLayerKernel{Op} <: AbstractKernel
    op::Op
end

function singularity_order(K::GradientSingleLayerKernel)
    N = ambient_dimension(K.op)
    return 1 - N
end

"""
    struct GradientDoubleLayerKernel{Op} <: AbstractKernel

Given an operator `Op`, construct its free-space gradient double-layer kernel.
This evaluates the gradient of the double-layer kernel with respect to the target variable.
"""
struct GradientDoubleLayerKernel{Op} <: AbstractKernel
    op::Op
end

function singularity_order(K::GradientDoubleLayerKernel)
    N = ambient_dimension(K.op)
    return -N
end

"""
    struct SourceGradientSingleLayerKernel{Op} <: AbstractKernel

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
struct SourceGradientSingleLayerKernel{Op} <: AbstractKernel
    op::Op
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

"""
    struct HessianKernel{Op} <: AbstractKernel

The Hessian of the single-layer kernel `G(x,y)` with respect to the *target*
variable `x`, i.e. ``\\nabla_x\\nabla_x G(x,y)``, returned as an `N×N`
`SMatrix`.

This is the kernel of the strongly-singular volume integral operator
``\\mathcal{X}[g](x) = \\nabla\\mathcal{W}[g](x) = \\mathsf{S}\\,g(x) -
\\mathrm{p.v.}\\!\\int_\\Omega \\nabla_x\\nabla_y G(x,y)\\cdot g(y)\\,dy`` (eq.
(2.24) of [anderson2026general](@cite)) acting on a vector density `g`. Since
``\\nabla_x\\nabla_y G = -\\nabla_x\\nabla_x G``, the principal-value integral
equals ``+\\int_\\Omega \\nabla_x\\nabla_x G \\cdot g``, so this (target) Hessian
is the kernel assembled for the forward map of ``\\mathcal{X}``; the free-term
tensor ``\\mathsf{S}`` is handled by the density-interpolation regularization.
"""
struct HessianKernel{Op} <: AbstractKernel
    op::Op
    charge_dipole::Symbol
end

function HessianKernel(op::AbstractDifferentialOperator, charge_dipole::Symbol = :charge)
    if !(charge_dipole == :charge || charge_dipole == :dipole)
        error("Invalid charge/dipole selection")
    end
    return HessianKernel{typeof(op)}(op, charge_dipole)
end

function singularity_order(K::HessianKernel)
    N = ambient_dimension(K.op)
    return -N
end

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
        id2 = @fastmath one(d2) / d2
        v = @fastmath dot(r, ny) * id2 * sqrt(id2) / 4 / π
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
        id2 = @fastmath one(d2) / d2
        v = @fastmath -dot(r, nx) * id2 * sqrt(id2) / 4 / π
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
    id2 = @fastmath one(d2) / d2
    # nxᵀ(I - N*rrᵀ/d²)ny = nxdny - N*rdnx*rdny/d²
    nxdny = dot(nx, ny)
    rdnx = dot(r, nx)
    rdny = dot(r, ny)
    if N == 2
        v = @fastmath id2 * (nxdny - 2 * rdnx * rdny * id2) / 2 / π
    elseif N == 3
        v = @fastmath id2 * sqrt(id2) * (nxdny - 3 * rdnx * rdny * id2) / 4 / π
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(v) : v
end

function (GSL::GradientSingleLayerKernel{Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    d = norm(r)
    if N == 2
        v = -1 / (2π) / (d^2) * r
    elseif N == 3
        v = -1 / (4π) / (d^3) * r
    else
        notimplemented()
    end
    return d ≤ SAME_POINT_TOLERANCE ? zero(v) : v
end

function (GDL::GradientDoubleLayerKernel{Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    ny = normal(source)
    d = norm(r)
    if N == 2
        v = 1 / (2π) / (d^2) * (ny - 2 * dot(r, ny) / d^2 * r)
    elseif N == 3
        v = 1 / (4π) / (d^3) * (ny - 3 * dot(r, ny) / d^2 * r)
    else
        notimplemented()
    end
    return d ≤ SAME_POINT_TOLERANCE ? zero(v) : v
end

function (HSL::HessianKernel{Laplace{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    d = norm(r)
    # ∇ₓ∇ₓG: for Laplace, ∂ᵢ∂ⱼG = c/dᴺ (k·r̂ᵢr̂ⱼ - δᵢⱼ) with (c,k) = (1/2π,2) in 2D
    # and (1/4π,3) in 3D.
    if N == 2
        v = 1 / (2π) / d^2 * (2 * r * transpose(r) / d^2 - I)
    elseif N == 3
        v = 1 / (4π) / d^3 * (3 * r * transpose(r) / d^2 - I)
    else
        notimplemented()
    end
    return d ≤ SAME_POINT_TOLERANCE ? zero(v) : v
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

function (SL::SingleLayerKernel{<:Yukawa{N, K}})(target, source) where {N, K}
    λ = SL.op.λ
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        d = sqrt(d2)
        v = Bessels.besselk(0, λ * d) / 2 / π
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
        v = λ * Bessels.besselk(1, λ * d) * rdny / d / 2 / π
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
        v = -λ * Bessels.besselk(1, λ * d) * rdnx / d / 2 / π
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
        k1 = Bessels.besselk(1, λ * d)
        k2 = Bessels.besselk(2, λ * d)
        a = -λ^2 / (2π * d^2) * k2
        b = λ / (2π * d) * k1
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

hankelh1(n, x::Real) = Bessels.hankelh1(n, x)
hankelh1(n, x::Complex) = SpecialFunctions.hankelh1(n, x)

# ---------------------------------------------------------------------------
# `H₀⁽¹⁾(z)` and `H₁⁽¹⁾(z)/z` at small argument
# ---------------------------------------------------------------------------
#
# The 2D Helmholtz kernels evaluated the way a *near-field* caller needs them. Two
# structural facts pay for a separate routine:
#
#  * both are **even** in `z` up to a single `log z`, so `u = z²/4 = (k²/4)|x-y|²` is the
#    natural variable — no square root is taken at all, and `ln(z/2) = ½ln u` is the only
#    transcendental left;
#  * `J` and `Y` share every power of `u`, and so do orders `0` and `1`, so one pass over
#    the series produces the whole pair a layer operator wants.
#
# This is an evaluation strategy, not an approximation: outside the series' range the
# general routine is called instead, and the branch is on `u` alone. The two term counts
# below hold the relative error, against `Bessels`, at 5e-16 for `u ≤ 1` and 4e-15 for
# `u ≤ 25/4`; past that the ascending series loses digits to cancellation faster than added
# terms recover them, which is where the fallback starts. In `lvdim` terms the fast path
# covers `k·|x-y| ≤ 5`, and `|x-y|` there is at most a patch across.

const HANKEL_SERIES_U_LO = 1.0     # z ≤ 2
const HANKEL_SERIES_U_HI = 6.25    # z ≤ 5

# `J₀ = Σ aₘuᵐ`, `Y₀ = (2/π)[(ln(z/2)+γ)J₀ + Σ bₘuᵐ]`,
# `J₁/z = Σ cₘuᵐ`, `Y₁/z = (2/π)[ln(z/2)·J₁/z - 1/(4u)] - (1/2π)Σ dₘuᵐ`   (DLMF 10.8.1)
_hankel_a(m) = (-1)^m // factorial(big(m))^2
_hankel_c(m) = (-1)^m // (2 * factorial(big(m)) * factorial(big(m + 1)))
_harmonic(n) = sum(1 // big(j) for j in 1:n; init = 0 // big(1))
_hankel_series(f, nt) = ntuple(i -> Float64(f(i - 1)), nt)

for (lo, nt) in ((:LO, 12), (:HI, 20))
    @eval begin
        const $(Symbol(:HANKEL_J0_, lo)) = _hankel_series(_hankel_a, $nt)
        const $(Symbol(:HANKEL_Y0_, lo)) =
            _hankel_series(m -> -_hankel_a(m) * _harmonic(m), $nt)
        const $(Symbol(:HANKEL_J1_, lo)) = _hankel_series(_hankel_c, $nt)
        const $(Symbol(:HANKEL_Y1_, lo)) = _hankel_series($nt) do m
            2 * _hankel_c(m) * (_harmonic(m) + _harmonic(m + 1) -
                                2 * big(MathConstants.eulergamma))
        end
    end
end

@inline function _h0_series(u, aj, ay)
    j = evalpoly(u, aj)
    lnz2 = log(u) / 2                              # `ln(z/2)`, since `u = (z/2)²`
    return complex(j, (2 / π) * ((lnz2 + MathConstants.eulergamma) * j + evalpoly(u, ay)))
end

@inline function _h01_series(u, aj0, ay0, aj1, ay1)
    j0, j1 = evalpoly(u, aj0), evalpoly(u, aj1)    # `J₀(z)` and `J₁(z)/z`
    lnz2 = log(u) / 2
    h0 = complex(j0, (2 / π) * ((lnz2 + MathConstants.eulergamma) * j0 + evalpoly(u, ay0)))
    h1 = complex(j1, (2 / π) * (lnz2 * j1 - 1 / (4u)) - evalpoly(u, ay1) / (2π))
    return h0, h1
end

"""
    _hankelh1_0(k, d²)    -> H₀⁽¹⁾(k√d²)
    _hankelh1_01(k, d²)   -> (H₀⁽¹⁾(z), H₁⁽¹⁾(z)/z),  z = k√d²

The 2D Helmholtz kernels' Hankel functions in terms of the *squared* distance, which is
what a kernel has and what the series above wants. `H₁⁽¹⁾/z` rather than `H₁⁽¹⁾` because
that is the even combination, and the `1/d` the double layer divides by is exactly the one
this cancels.

Complex `k` (a lossy medium) takes the general routine: the series' branch is an inequality
on `u`.
"""
@inline function _hankelh1_0(k::Real, d2::Real)
    u = (k * k / 4) * d2
    u ≤ HANKEL_SERIES_U_LO && return _h0_series(u, HANKEL_J0_LO, HANKEL_Y0_LO)
    u ≤ HANKEL_SERIES_U_HI && return _h0_series(u, HANKEL_J0_HI, HANKEL_Y0_HI)
    return hankelh1(0, k * sqrt(d2))
end

_hankelh1_0(k, d2) = hankelh1(0, k * sqrt(d2))

@doc (@doc _hankelh1_0)
@inline function _hankelh1_01(k::Real, d2::Real)
    u = (k * k / 4) * d2
    u ≤ HANKEL_SERIES_U_LO &&
        return _h01_series(u, HANKEL_J0_LO, HANKEL_Y0_LO, HANKEL_J1_LO, HANKEL_Y1_LO)
    u ≤ HANKEL_SERIES_U_HI &&
        return _h01_series(u, HANKEL_J0_HI, HANKEL_Y0_HI, HANKEL_J1_HI, HANKEL_Y1_HI)
    z = k * sqrt(d2)
    return hankelh1(0, z), hankelh1(1, z) / z
end

function _hankelh1_01(k, d2)
    z = k * sqrt(d2)
    return hankelh1(0, z), hankelh1(1, z) / z
end

function (SL::SingleLayerKernel{<:Helmholtz{N}})(target, source) where {N}
    k = SL.op.k
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    if N == 2
        v = im / 4 * _hankelh1_0(k, d2)
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
        # `H₁⁽¹⁾(z)/z` absorbs the `1/d`: `im·k/(4d)·H₁⁽¹⁾(kd) = im·k²/4·H₁⁽¹⁾(z)/z`
        v = im * k^2 / 4 * last(_hankelh1_01(k, d2)) * rdny
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

"""
    layer_pair_value(S, D, target, source) -> (s, d)

`(S(target, source), D(target, source))` as one call, for a caller that always wants both at
the same pair of points — `lvdim`'s quadrature over `∂Ωτ` is the whole reason this exists.
The fallback is the two calls, so any kernel pair may be passed; a pair that shares work
overrides it.

For 2D Helmholtz the sharing is nearly total: one `|x-y|²`, one series pass, and one
logarithm serve both orders (see [`_hankelh1_01`](@ref)), where two independent kernel calls
repeat all three.
"""
@inline layer_pair_value(S, D, target, source) = (S(target, source), D(target, source))

@inline function layer_pair_value(
        S::SingleLayerKernel{<:Helmholtz{2}}, D::DoubleLayerKernel{<:Helmholtz{2}},
        target, source,
    )
    k = S.op.k
    # the fused form reads one wavenumber; a mismatched pair is a caller error, not ours
    k == D.op.k || return (S(target, source), D(target, source))
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    h0, h1z = _hankelh1_01(k, d2)
    sv = im / 4 * h0
    dv = im * k^2 / 4 * h1z * dot(r, normal(source))
    return d2 ≤ tol * tol ? (zero(sv), zero(dv)) : (sv, dv)
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

function (GSL::GradientSingleLayerKernel{<:Helmholtz{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    k = GSL.op.k
    d = norm(r)
    if N == 2
        v = -im * k / 4 / d * hankelh1(1, k * d) * r
    elseif N == 3
        v = 1 / (4π) / d^2 * exp(im * k * d) * (im * k - 1 / d) * r
    else
        notimplemented()
    end
    return d ≤ SAME_POINT_TOLERANCE ? zero(v) : v
end

function (GDL::GradientDoubleLayerKernel{<:Helmholtz{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    ny = normal(source)
    k = GDL.op.k
    d = norm(r)
    rdotny = dot(r, ny)
    if N == 2
        v = im * k / (4 * d) * hankelh1(1, k * d) * ny -
            im * k^2 / (4 * d^2) * hankelh1(2, k * d) * r * rdotny
    elseif N == 3
        pref = 1 / (4π) / d^3 * exp(im * k * d)
        v = pref * ((1 - im * k * d) * ny + (k^2 * d^2 + 3 * im * k * d - 3) / d^2 * r * rdotny)
    else
        notimplemented()
    end
    return d ≤ SAME_POINT_TOLERANCE ? zero(v) : v
end

function (HSL::HessianKernel{<:Helmholtz{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    k = HSL.op.k
    d = norm(r)
    # ∇ₓ∇ₓG = (g″−g′/d) r̂r̂ᵀ + (g′/d) I for the isotropic G = g(d).
    if N == 2
        # 2D: G = (i/4)H₀⁽¹⁾(kd);  recurrence leads to H₂.
        v = im * k^2 / 4 / d^2 * hankelh1(2, k * d) * (r * transpose(r)) -
            im * k / 4 / d * hankelh1(1, k * d) * I
    elseif N == 3
        # 3D: G = eⁱᵏᵈ/(4πd).
        pref = exp(im * k * d) / (4π)
        cI = pref * (im * k / d^2 - 1 / d^3)
        cR = pref * (3 / d^5 - 3 * im * k / d^4 - k^2 / d^3)
        v = cR * (r * transpose(r)) + cI * I
    else
        notimplemented()
    end
    return d ≤ SAME_POINT_TOLERANCE ? zero(v) : v
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
        id2 = @fastmath one(d2) / d2
        c = @fastmath 3 * id2 * id2 * sqrt(id2) / 4 / π * rdny
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
        id2 = @fastmath one(d2) / d2
        c = @fastmath -3 * id2 * id2 * sqrt(id2) / 4 / π * rdnx
    else
        notimplemented()
    end
    v = c * RRT
    return d2 ≤ tol * tol ? zero(v) : v
end

function apply_kernel(
        SL::SingleLayerKernel{<:Stokes{N}}, target, source, v,
    ) where {N}
    μ = SL.op.μ
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    tol = oftype(d2, SAME_POINT_TOLERANCE)
    rdv = dot(r, v)
    if N == 2
        γ = @fastmath -log(d2) / 2
        invd2 = @fastmath one(d2) / d2
        out = (γ * v + (invd2 * rdv) * r) / (μ * 4 * π * (N - 1))
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        invd3 = @fastmath invd * invd * invd
        out = (invd * v + (invd3 * rdv) * r) / (μ * 4 * π * (N - 1))
    else
        notimplemented()
    end
    return d2 ≤ tol * tol ? zero(out) : out
end

function apply_kernel(
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
        c = @fastmath id2 * id2 / π * rdny * rdv
    elseif N == 3
        id2 = @fastmath one(d2) / d2
        c = @fastmath 3 * id2 * id2 * sqrt(id2) / 4 / π * rdny * rdv
    else
        notimplemented()
    end
    out = c * r
    return d2 ≤ tol * tol ? zero(out) : out
end

function apply_kernel(
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
        c = @fastmath -id2 * id2 / π * rdnx * rdv
    elseif N == 3
        id2 = @fastmath one(d2) / d2
        c = @fastmath -3 * id2 * id2 * sqrt(id2) / 4 / π * rdnx * rdv
    else
        notimplemented()
    end
    out = c * r
    return d2 ≤ tol * tol ? zero(out) : out
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

function apply_kernel(HS::HyperSingularKernel{<:Stokes{N}}, target, source, v) where {N}
    μ = HS.op.μ
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
        c = μ * id2 / π
        α = 2 * rdny * id2
        vr = (-4α * rdnx + nxdny) * id2 * r + α * nx / 2
        vnx = ny
        vny = rdnx * id2 * r
        a_diag = α * rdnx / 2
    elseif N == 3
        c = μ * id2 * invd / 2 / π
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

function apply_kernel(SL::SingleLayerKernel{<:Elastostatic{N}}, target, source, v) where {N}
    μ, λ = SL.op.μ, SL.op.λ
    ν = λ / (2 * (μ + λ))
    r = coords(target) - coords(source)
    d2 = dot(r, r)
    rdv = dot(r, v)
    if N == 2
        id2 = @fastmath one(d2) / d2
        out = (-(3 - 4 * ν) * log(d2) / 2 * v + (id2 * rdv) * r) / (μ * 8 * π * (1 - ν))
    elseif N == 3
        invd = @fastmath one(d2) / sqrt(d2)
        id2 = invd * invd
        out = invd * ((3 - 4 * ν) * v + (id2 * rdv) * r) / (μ * 16 * π * (1 - ν))
    else
        notimplemented()
    end
    return iszero(d2) ? zero(out) : out
end

function apply_kernel(DL::DoubleLayerKernel{<:Elastostatic{N}}, target, source, v) where {N}
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
        out = -invd * (drdn * (ν1 * v + 2 * id2 * rdv * r) + (ν1 * invd) * asym) / 4 / π / (1 - ν)
    elseif N == 3
        out = -id2 * (drdn * (ν1 * v + 3 * id2 * rdv * r) + (ν1 * invd) * asym) / 8 / π / (1 - ν)
    else
        notimplemented()
    end
    return iszero(d2) ? zero(out) : out
end

function apply_kernel(ADL::AdjointDoubleLayerKernel{<:Elastostatic{N}}, target, source, v) where {N}
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
        out = invd * (drdn * (ν1 * v + 2 * id2 * rdv * r) + (ν1 * invd) * asym) / 4 / π / (1 - ν)
    elseif N == 3
        out = id2 * (drdn * (ν1 * v + 3 * id2 * rdv * r) + (ν1 * invd) * asym) / 8 / π / (1 - ν)
    else
        notimplemented()
    end
    return iszero(d2) ? zero(out) : out
end

function apply_kernel(HS::HyperSingularKernel{<:Elastostatic{N}}, target, source, v) where {N}
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
    out = c * (dot(vr, v) * r + dot(vnx, v) * nx + dot(vny, v) * ny + a_diag * v)
    return iszero(d2) ? zero(out) : out
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

# The `ntuple` construction below defeats `promote_op`, so declare the value type
# explicitly rather than let `IntegralOperator` fall back to `Any`.
function return_type(
        ::Union{
            GradientSingleLayerKernel{<:Elastostatic{N}},
            GradientDoubleLayerKernel{<:Elastostatic{N}},
        },
        args...,
    ) where {N}
    return SVector{N, SMatrix{N, N, Float64, N * N}}
end

function (K::GradientSingleLayerKernel{<:Elastostatic{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    μ, λ = K.op.μ, K.op.λ
    ν = λ / (2 * (μ + λ))
    d = norm(r)
    RRT = r * r'
    SM = SMatrix{N, N, Float64, N * N}
    if N == 2
        C = 1 / (8π * μ * (1 - ν))
        v = SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (-(3 - 4ν) * r[k] / d^2 * I + (ek * r' + r * ek') / d^2 - 2 * r[k] * RRT / d^4))
            end
        )
    elseif N == 3
        C = 1 / (16π * μ * (1 - ν))
        v = SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (-(3 - 4ν) * r[k] / d^3 * I + (ek * r' + r * ek') / d^3 - 3 * r[k] * RRT / d^5))
            end
        )
    else
        notimplemented()
    end
    return d == 0 ? zero(v) : v
end

function (K::GradientDoubleLayerKernel{<:Elastostatic{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    μ, λ = K.op.μ, K.op.λ
    ν = λ / (2 * (μ + λ))
    ny = normal(source)
    d = norm(r)
    RRT = r * r'
    qr = dot(ny, r)
    B = r * ny' - ny * r'
    SM = SMatrix{N, N, Float64, N * N}
    if N == 2
        C = 1 / (4π * (1 - ν))
        A = (1 - 2ν) * I + 2 * RRT / d^2
        v = SVector{N}(
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
        v = SVector{N}(
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
    else
        notimplemented()
    end
    return d == 0 ? zero(v) : v
end

################################################################################
################################### STOKES GRADIENT ############################
################################################################################

# See the note on the Elastostatic gradient `return_type` above.
function return_type(
        ::Union{
            GradientSingleLayerKernel{<:Stokes{N}},
            GradientDoubleLayerKernel{<:Stokes{N}},
        },
        args...,
    ) where {N}
    return SVector{N, SMatrix{N, N, Float64, N * N}}
end

function (K::GradientSingleLayerKernel{<:Stokes{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    μ = K.op.μ
    d = norm(r)
    RRT = r * r'
    SM = SMatrix{N, N, Float64, N * N}
    if N == 2
        C = 1 / (4π * μ)
        v = SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (-r[k] / d^2 * I + (ek * r' + r * ek') / d^2 - 2 * r[k] * RRT / d^4))
            end
        )
    elseif N == 3
        C = 1 / (8π * μ)
        v = SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (-r[k] / d^3 * I + (ek * r' + r * ek') / d^3 - 3 * r[k] * RRT / d^5))
            end
        )
    else
        notimplemented()
    end
    return d == 0 ? zero(v) : v
end

function (K::GradientDoubleLayerKernel{<:Stokes{N}})(
        target,
        source,
        r = coords(target) - coords(source),
    ) where {N}
    ny = normal(source)
    d = norm(r)
    RRT = r * r'
    qr = dot(ny, r)
    SM = SMatrix{N, N, Float64, N * N}
    if N == 2
        C = 1 / π
        v = SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (ny[k] * RRT / d^4 + qr * (ek * r' + r * ek') / d^4 - 4 * qr * r[k] * RRT / d^6))
            end
        )
    elseif N == 3
        C = 3 / (4π)
        v = SVector{N}(
            ntuple(N) do k
                ek = SVector{N}(ntuple(i -> i == k ? 1.0 : 0.0, N))
                SM(C * (ny[k] * RRT / d^5 + qr * (ek * r' + r * ek') / d^5 - 5 * qr * r[k] * RRT / d^7))
            end
        )
    else
        notimplemented()
    end
    return d == 0 ? zero(v) : v
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
