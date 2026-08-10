"""
    monomial_exponents(N, order)

Exponents `I ∈ ℕᴺ` of the monomials `x^I` of total degree `|I| ≤ order` — the simplex-shaped
set, not the `(order+1)^N` box — graded by total degree, ties lexicographic (`I[N]` fastest).
For `N = 2, order = 2`: `(0,0)`, `(0,1)`, `(1,0)`, `(0,2)`, `(1,1)`, `(2,0)`.

Two consequences the rest of the file leans on: `I - eᵢ` precedes `I`, and the set for `order`
is a prefix of the set for any larger degree.
"""
function monomial_exponents(N, order)
    # reversing the product index makes `I[N]` the fastest axis, i.e. lexicographic in `I`
    I = [NTuple{N, Int}(reverse(t)) for t in Iterators.product(ntuple(_ -> 0:order, N)...)]
    I = filter(I -> sum(I) ≤ order, vec(I))
    return sort!(I; by = sum, alg = MergeSort) # stable, so ties keep lexicographic order
end

"""
    struct MonomialBasis{N}

Callable structure for the monomial basis of total degree `≤ order` in `N` dimensions:
`m(x)` returns the vector of *all* monomial values `x^I` at `x`, ordered by
[`monomial_exponents`](@ref).

Evaluation costs one multiplication per monomial: the grading guarantees `x^{I-eᵢ}` is
already available when `x^I` is reached, so `parent`/`axis` name it instead of recomputing
coordinate powers.

The `N` derivative matrices are built once here rather than on demand: they are
element-independent, and [`derivative_matrix`](@ref) is called from inside the per-element
loops of `lvdim`, where rebuilding the `Dict` behind each one showed up as pure overhead.
"""
struct MonomialBasis{N}
    exponents::Vector{NTuple{N, Int}}
    degrees::Vector{Int}                   # total degree `|I|`, non-decreasing
    parent::Vector{Int}                    # index of `I - e_{axis[j]}`
    axis::Vector{Int}                      # coordinate `parent` steps along
    derivatives::NTuple{N, SparseMatrixCSC{Float64, Int}}   # `∂/∂x_d`, cached
end

MonomialBasis(dim::Int, order) = MonomialBasis(Val(dim), order)

function MonomialBasis(::Val{N}, order) where {N}
    exponents = monomial_exponents(N, order)
    pos = Dict(I => j for (j, I) in enumerate(exponents))
    parent, axis = zeros(Int, length(exponents)), ones(Int, length(exponents))
    for (j, I) in enumerate(exponents)
        i = findfirst(>(0), I)
        isnothing(i) && continue                 # the constant, `m[1] = 1`
        parent[j] = pos[ntuple(l -> l == i ? I[l] - 1 : I[l], Val(N))]
        axis[j] = i
    end
    derivatives = ntuple(d -> _derivative_matrix(exponents, d, Float64), Val(N))
    return MonomialBasis{N}(
        exponents, [sum(I) for I in exponents], parent, axis, derivatives
    )
end

Base.length(m::MonomialBasis) = length(m.exponents)
degree(m::MonomialBasis) = sum(last(m.exponents))   # graded, so the last is of maximal degree
ambient_dimension(::MonomialBasis{N}) where {N} = N
exponents(m::MonomialBasis) = m.exponents

Base.show(io::IO, m::MonomialBasis{N}) where {N} =
    print(io, "MonomialBasis: $(length(m)) monomials of degree ≤ $(degree(m)) in $N variables")

"""
    derivative_matrix(m::MonomialBasis, d[, T]) -> D

`∂/∂x_d` in coefficient space over `m`, from `∂_d x^I = I_d x^{I-e_d}`: if row `β` of `C`
holds the coefficients of `p_β` over `m`, row `β` of `C * D` holds those of `∂_d p_β`.

Exact, not truncated: differentiating lowers the degree, so every column written to is in the
graded set.

The two-argument form returns the `Float64` matrix cached on `m`; only a caller wanting
another element type pays for a build.
"""
derivative_matrix(m::MonomialBasis, d) = m.derivatives[d]

derivative_matrix(m::MonomialBasis, d, ::Type{T}) where {T} =
    _derivative_matrix(m.exponents, d, T)

function _derivative_matrix(exponents::Vector{NTuple{N, Int}}, d, ::Type{T}) where {N, T}
    pos = Dict(I => j for (j, I) in enumerate(exponents))
    rows, cols, vals = Int[], Int[], T[]
    for (j, I) in enumerate(exponents)
        iszero(I[d]) && continue
        push!(rows, j)
        push!(cols, pos[ntuple(l -> l == d ? I[l] - 1 : I[l], Val(N))])
        push!(vals, I[d])
    end
    return sparse(rows, cols, vals, length(exponents), length(exponents))
end

# unit vector e_d ∈ ℝᴺ, as the one-column sparse matrix `kron` needs to place a block
unit_matrix(N, d, ::Type{T}) where {T} = sparse([d], [1], [one(T)], N, 1)

"""
    gradient_matrix(m::MonomialBasis, T = Float64) -> G

`∇` in coefficient space over `m`, of size `N·nb × nb` with `nb = length(m)`, laid out with
the direction varying fastest: `(G * m(x))[N*(j-1) + d] = ∂_d m_j(x)`, so each monomial's
gradient is contiguous. Stacked from the [`derivative_matrix`](@ref) blocks as
`Σ_d kron(D_d, e_d)`.
"""
function gradient_matrix(m::MonomialBasis{N}, ::Type{T} = Float64) where {N, T}
    return sum(kron(derivative_matrix(m, d, T), unit_matrix(N, d, T)) for d in 1:N)
end

"""
    divergence_matrix(m::MonomialBasis, T = Float64) -> Div

`∇·` in coefficient space over `m`, of size `nb × N·nb`, consuming the interleaved layout
[`gradient_matrix`](@ref) produces, so that the two compose:

    divergence_matrix(m) * gradient_matrix(m) == Δ == sum(D_d^2)
"""
function divergence_matrix(m::MonomialBasis{N}, ::Type{T} = Float64) where {N, T}
    return sum(kron(derivative_matrix(m, d, T), transpose(unit_matrix(N, d, T))) for d in 1:N)
end

"""
    laplace_matrix(m::MonomialBasis, T = Float64) -> Λ

`Δ` in coefficient space over `m`, as the composition `∇·∘∇` of the two primitives above.
Lowers degree by two and is therefore nilpotent.
"""
laplace_matrix(m::MonomialBasis{N}, ::Type{T} = Float64) where {N, T} =
    divergence_matrix(m, T) * gradient_matrix(m, T)

"""
    r2_matrix(m::MonomialBasis, T = Float64) -> M

Multiplication by `r² = Σ_d x_d²` in coefficient space over `m`, from `r² x^I = Σ_d
x^{I+2e_d}`.  Raises the degree by two, so it truncates at the top of the graded set;
[`laplace_inverse_matrix`](@ref), its only caller, visits no truncated row.
"""
function r2_matrix(m::MonomialBasis{N}, ::Type{T} = Float64) where {N, T}
    pos = Dict(I => j for (j, I) in enumerate(m.exponents))
    rows, cols, vals = Int[], Int[], T[]
    for (j, I) in enumerate(m.exponents), d in 1:N
        K = ntuple(l -> I[l] + 2 * (l == d), Val(N))
        haskey(pos, K) || continue
        push!(rows, j)
        push!(cols, pos[K])
        push!(vals, one(T))
    end
    return sparse(rows, cols, vals, length(m), length(m))
end

"""
    laplace_inverse_matrix(m::MonomialBasis, T = Float64) -> R

A right inverse of [`laplace_matrix`](@ref): row `β` of `R` holds the coefficients of a
polynomial `Ψ_β` with `Δ Ψ_β = m_β`, so `Ψ = R * m₊(x)` evaluates the whole family.

`Δ` lowers the degree by two, so its right inverse raises it by two: `m` of degree `n` is the
source and the solutions live in `m₊ = MonomialBasis(N, n + 2)`, making `R` rectangular,
`length(m) × length(m₊)`. `R * Λ₊ == [I 0]` exactly, since `Δ` maps degree `d+2` *onto* degree
`d`; the reverse composition is only a projector, solving then differentiating recovering a
polynomial up to the harmonic ones.

The right inverse is not unique, but rotation equivariance and homogeneity pin it down — a
minimum-norm `pinv` would pick another, the coefficient `ℓ²` norm not being rotation
invariant. It is the classical Almansi series, with `M₂` from [`r2_matrix`](@ref):

    Ψ = Σ_{k=0}^{⌊n/2⌋} c_k r^{2k+2} Δ^k m,   c_0 = 1/γ(0,n),  c_k = -c_{k-1}/γ(k, n-2k)

`γ(k,p) = 2(k+1)(2k+2p+N)`, terminating because `Δ^k` annihilates a degree-`n` monomial once
`2k > n`. No `pinv`, no least squares, no thresholds.
"""
function laplace_inverse_matrix(m::MonomialBasis{N}, ::Type{T} = Float64) where {N, T}
    n = degree(m)
    m₊ = MonomialBasis(Val(N), n + 2)
    Λ, M₂ = laplace_matrix(m₊, T), r2_matrix(m₊, T)
    nb, deg = length(m), [sum(I) for I in m.exponents]
    γ(k, p) = 2 * (k + 1) * (2k + 2p + N)
    cs = zeros(T, n ÷ 2 + 1, n + 1)           # cs[k+1, d+1], the recursion for source degree d
    for d in 0:n
        cs[1, d + 1] = one(T) / γ(0, d)
        for k in 1:(d ÷ 2)
            cs[k + 1, d + 1] = -cs[k, d + 1] / γ(k, d - 2k)
        end
    end
    R = spzeros(T, nb, length(m₊))
    A = M₂                                    # A_k = Λ^k M₂^{k+1}, so A_0 = M₂
    for k in 0:(n ÷ 2)
        R += Diagonal([cs[k + 1, d + 1] for d in deg]) * A[1:nb, :]
        A = Λ * A * M₂                        # A_{k+1} = Λ A_k M₂
    end
    return R
end

"""
    (m::MonomialBasis)(x, nb = length(m)) -> v

The value of the first `nb` monomials at `x`. The element type follows `x`, so this serves
plain points as well as the `ForwardDiff.Dual`s a caller differentiating through sends in.

Truncating is legitimate for any `nb`: the set is graded, so every monomial's parent precedes
it and the first `nb` entries are exactly the basis they would form on their own. That is how
a lower-degree family is evaluated without a second basis.
"""
function (m::MonomialBasis)(x, nb::Integer = length(m))
    v = Vector{float(eltype(x))}(undef, nb)
    @inbounds v[1] = one(eltype(v))
    @inbounds for j in 2:nb
        v[j] = v[m.parent[j]] * x[m.axis[j]]
    end
    return v
end

"""
    (m::MonomialBasis)(x, ν::AbstractVector, nb = length(m)) -> (v, dv)

The values *and* the directional derivative `∂_ν` of the first `nb` monomials at `x`.

Differentiating the recursion rather than the result: from `x^I = x^{I-e_a}·x_a` the product
rule gives `∂_ν x^I = (∂_ν x^{I-e_a})·x_a + x^{I-e_a}·ν_a`, so the derivatives ride along in the
pass that produces the values. This is why the scalar bases store no gradient matrix — the
recursion contracts against `ν` as it goes, where coefficient space would contract over the
*larger* space `m₊`.
"""
function (m::MonomialBasis)(x, ν::AbstractVector, nb::Integer = length(m))
    T = float(promote_type(eltype(x), eltype(ν)))
    v, dv = Vector{T}(undef, nb), Vector{T}(undef, nb)
    @inbounds v[1], dv[1] = one(T), zero(T)
    @inbounds for j in 2:nb
        p, a = m.parent[j], m.axis[j]
        v[j] = v[p] * x[a]
        dv[j] = dv[p] * x[a] + v[p] * ν[a]
    end
    return v, dv
end

"""
    degrees(m::MonomialBasis) -> d

Total degree of each monomial, `d[j] = |I_j|` — non-decreasing, and what every degree-graded
rescaling of a coefficient matrix is indexed by.
"""
degrees(m::MonomialBasis) = m.degrees

"""
    shifted_laplace_inverse_matrix(m::MonomialBasis, T = Float64) -> S₁

The inverse of `Δ + 1` in coefficient space over `m`: row `β` holds the coefficients of the
polynomial `u_β` with `(Δ + 1) u_β = m_β`.

Unlike `Δ`, the shifted operator is *invertible* on polynomials and its inverse raises no
degree — `S₁` is square and the solution lives in `m` itself. Both facts come from `Δ` being
nilpotent, which terminates the Neumann series `(Δ + 1)⁻¹ = Σ_{j=0}^{⌊n/2⌋} (-Δ)^j`, so
`S₁(Λ + I) = I` telescopes exactly and in integer arithmetic.

Nilpotency also makes the polynomial solution *unique*, hence forced to blow up like `k⁻²` as
`k → 0` — a property of the problem, not of this construction, and why
[`ShiftedLaplaceParticularBasis`](@ref) abandons this branch below a crossover.
"""
function shifted_laplace_inverse_matrix(m::MonomialBasis{N}, ::Type{T} = Float64) where {N, T}
    Λ = laplace_matrix(m, T)
    S₁, Λʲ = sparse(one(T) * I, length(m), length(m)), Λ
    for _ in 1:(degree(m) ÷ 2)
        S₁ -= Λʲ
        Λʲ = -(Λʲ * Λ)                       # carries the alternating sign along
    end
    return S₁
end

"""
    shifted_laplace_solution_matrix(S₁, degs, κ²) -> S

The inverse of `Δ + κ²` over `m`, from the unit solve `S₁` and the degrees `degs` of `m`.

No new solve is needed: `u_β^κ(x) = κ^{-(d_β+2)} u_β^1(κx)` solves the rescaled problem when
`m_β` is homogeneous of degree `d_β`, and `x ↦ κx` scales `m_γ`'s coefficient by `κ^{d_γ}`. The
whole shift dependence is therefore the entrywise, degree-graded factor below.

Parametrized by `κ²` rather than `κ` because the exponent `d_γ - d_β - 2` is *always even*
(`Λ` lowers the degree by exactly two). That is what lets a **negative** shift — Yukawa, where
`κ² = -(λr)²` — stay in real arithmetic instead of going through an imaginary `κ`.

The rescaling is genuinely two-sided diagonal, and can be written as one: every nonzero has
`d_γ ≡ d_β (mod 2)`, so `diag((κ²)^{-(⌊d_β/2⌋+1)}) S₁ diag((κ²)^{⌊d_γ/2⌋})` reproduces it in
integral powers of `κ²`, real shift of either sign included. It is kept entrywise only because
the split buys nothing: applying the two diagonals to the vector on every evaluation costs more
than forming this matrix once per element, and it would stop `solution_matrix` from returning
a plain sparse matrix, which the derived bases multiply on both sides.

Since the shift moves no nonzero, only its value, the result is `similar(S₁)` — the same
sparsity pattern, its values overwritten in one pass — rather than a `findnz`/`sparse`
round-trip through coordinate form.
"""
function shifted_laplace_solution_matrix(S₁::SparseMatrixCSC, degs, κ²)
    S = similar(S₁, typeof(zero(eltype(S₁)) * one(κ²)))
    rv, w, nz = rowvals(S), nonzeros(S), nonzeros(S₁)
    for γ in axes(S, 2), p in nzrange(S, γ)
        @inbounds w[p] = nz[p] * κ²^((degs[γ] - degs[rv[p]] - 2) ÷ 2)
    end
    return S
end

"""
    galerkin_parameter(op) -> a

The parameter `a` of the Galerkin representation `u = (Δ - a∇∇·)F` used by
[`galerkin_inverse_matrix`](@ref): `(λ+μ)/(λ+2μ)` for `Elastostatic`, and `1` for `Stokes`,
which is exactly its incompressible (`λ → ∞`) limit — and the value at which `∇·u` vanishes
identically, as Stokes requires.
"""
galerkin_parameter(op::Elastostatic) = (op.λ + op.μ) / (op.λ + 2op.μ)
galerkin_parameter(::Stokes) = 1

"""
    galerkin_inverse_matrix(op, m::MonomialBasis, T = Float64) -> (R, S)

A right inverse of the vector operator `ℒ₀` in coefficient space, together with the
isotropic part of the corresponding stress. `R` is `length(m) × length(m₊)` with
`SMatrix{N,N}` entries and satisfies `R·ℒ₀₊ = [I 0]`, the identity
[`laplace_inverse_matrix`](@ref) satisfies with `Λ₊`; `S` has `SVector{N}` entries and feeds
[`traction_matrix`](@ref). `R[β,j][d,n]` is the coefficient of `m_j` in component `d` of the
solution whose source is `m_β e_n`, so `Ψ = r²(R·m₊(x̃))` is one sparse matvec returning
`SMatrix`es.

Serves **both** vector operators, because one construction covers them:

- `Elastostatic` — `ℒ₀u = -(μΔu + (λ+μ)∇(∇·u))`, isotropic stress `λ(∇·u)`
- `Stokes`       — `ℒ₀u = -μΔu + ∇p` with `∇·u = 0`, isotropic stress `-p`

Nothing new is solved. The classical Galerkin vector reduces both to the *biharmonic*
equation: with `a = ` [`galerkin_parameter`](@ref)`(op)` the cross terms cancel exactly, so

    u = (Δ - a∇∇·)F    ⟹    μΔ²F = -ℒ₀u ,

and `Δ⁻²` followed by `(Δ - a∇∇·)` is a right inverse built entirely from the scalar
primitives — `laplace_matrix`, `gradient_matrix` and `divergence_matrix`, whose interleaved
layout is what lets `∇∇·` be their plain product. `F` sits four degrees above the source and
`u` two, so the solution lands in `m₊` as in the scalar case.

For Stokes the pressure is `p = -μΔ(∇·F)`, and `a = 1` makes `∇·u = 0` *identically* rather
than merely harmonic. That is the whole reason for this route: solving `Δp = ∇·f` and
`μΔu = ∇p - f` directly satisfies the momentum equation just as exactly, but leaves `∇·u` a
nonzero harmonic polynomial, the right inverse having no reason to pick the solenoidal
representative.
"""
function galerkin_inverse_matrix(
        op::Union{Elastostatic{N}, Stokes{N}}, m::MonomialBasis{N}, ::Type{T} = Float64,
    ) where {N, T}
    μ = op.μ
    n = degree(m)
    m₊, m₊₊ = MonomialBasis(Val(N), n + 2), MonomialBasis(Val(N), n + 4)
    nb, nb₊ = length(m), length(m₊)
    Iₙ = sparse(one(T) * I, N, N)
    # Δ⁻² out of `m`: the first inverse lands in `m₊`, the second in `m₊₊`
    B = laplace_inverse_matrix(m, T) * laplace_inverse_matrix(m₊, T)
    a = T(galerkin_parameter(op))
    ΔmGD = kron(laplace_matrix(m₊₊, T), Iₙ) -
        a * (gradient_matrix(m₊₊, T) * divergence_matrix(m₊₊, T))
    # `-1/μ` since `ℒ₀` carries the leading minus, as `ℒ = -Δ` does for Laplace
    Fflat = -(one(T) / μ) * kron(B, Iₙ)
    Rflat = Fflat * ΔmGD
    R = _blockify(Rflat[:, 1:(N * nb₊)], nb, nb₊, Val(N), T)
    return R, _isotropic_stress(op, Fflat, R, m₊, m₊₊, T)
end

# The scalar `S` with `σ = S·Id + μ(∇u + ∇uᵀ)`, as `SVector{N}` entries over `(m, m₊)`:
# `S[β,j][n]` is the coefficient of `m_j` for the source `m_β e_n`.
#
# Elastostatic reads it off `u` itself; Stokes cannot, because `∇·u ≡ 0` there and the
# pressure is independent information — it has to come from the Galerkin vector `F`. The two
# agree in the limit: `λ(∇·u) = λμ/(λ+2μ)·Δ(∇·F)`, whose `λ → ∞` limit is `μΔ(∇·F) = -p`.
function _isotropic_stress(
        op::Elastostatic{N}, _Fflat, R, m₊::MonomialBasis{N}, _m₊₊, ::Type{T},
    ) where {N, T}
    λ = op.λ
    D = ntuple(g -> Matrix(R * derivative_matrix(m₊, g, T)), Val(N))
    return [
        SVector(ntuple(n -> λ * sum(f -> D[f][β, j][f, n], 1:N), Val(N)))
            for β in axes(R, 1), j in 1:length(m₊)
    ]
end

function _isotropic_stress(
        op::Stokes{N}, Fflat, R, m₊::MonomialBasis{N}, m₊₊::MonomialBasis{N}, ::Type{T},
    ) where {N, T}
    nb, nb₊ = size(R)
    # `-p = μΔ(∇·F)`, over `m₊₊` and then restricted to the `m₊` prefix it lands in
    divF = sum(
        Fflat[:, (0:(length(m₊₊) - 1)) .* N .+ d] * derivative_matrix(m₊₊, d, T) for d in 1:N
    )
    mp = Matrix((op.μ * divF * laplace_matrix(m₊₊, T))[:, 1:nb₊])
    # row `N(β-1)+n` of the flat layout is the source `m_β e_n`
    return [SVector(ntuple(n -> mp[N * (β - 1) + n, j], Val(N))) for β in 1:nb, j in 1:nb₊]
end

"""
    traction_matrix(op, R, S, m₊, T = Float64) -> γ₁R

Coefficients of the *traction* of the particular solutions whose coefficients are `R`:
`N·nb × nb₊` with `SMatrix{N,N}` entries, row `N(β-1)+e` holding the `e`-th column of the
stress tensor of `Ψ_β`, so that

    γ₁Ψ_β(x, ν) = r · Σ_e ν_e (γ₁R · m₊(x̃))[N(β-1)+e] .

Storing this keeps the trace **one sparse matvec against the plain monomial values**. The whole
`∇u` dependence is contracted here, once per operator, out of `G_g = R·D_g` — the coefficients
of `∂_g Ψ`, from [`derivative_matrix`](@ref):

    σ_{de} = S_n δ_{de} + μ((G_e)_{d n} + (G_d)_{e n}) ,

with `S` the isotropic part from [`galerkin_inverse_matrix`](@ref) — `λ(∇·u)` for
`Elastostatic`, `-p` for `Stokes`, the only place the two operators differ.

The scalar bases need no such matrix: there `γ₁Ψ = ∇Ψ·ν` *is* a directional derivative, which
the monomial recursion contracts as it goes. The traction cannot be — `tr(∇u)` and `∇uᵀν` read
the gradient transversally to `ν`.
"""
function traction_matrix(
        op::Union{Elastostatic{N}, Stokes{N}}, R, S, m₊::MonomialBasis{N},
        ::Type{T} = Float64,
    ) where {N, T}
    μ = op.μ
    nb, nb₊ = size(R)
    SM = SMatrix{N, N, T, N * N}
    # `G[g][β,j][f,n]` — coefficient of `m_j` in `∂_g` of component `f` of the solution for
    # source `m_β e_n`. Dense here only because this runs once, on a small matrix.
    G = ntuple(g -> Matrix(R * derivative_matrix(m₊, g, T)), Val(N))
    rows, cols, vals = Int[], Int[], SM[]
    for β in 1:nb, j in 1:nb₊
        Gs = ntuple(g -> G[g][β, j], Val(N))
        Sβj = S[β, j]
        (all(iszero, Gs) && iszero(Sβj)) && continue
        for e in 1:N
            blk = SM(
                (d == e) * Sβj[n] + μ * (Gs[e][d, n] + Gs[d][e, n]) for d in 1:N, n in 1:N
            )
            iszero(blk) && continue
            push!(rows, N * (β - 1) + e)
            push!(cols, j)
            push!(vals, blk)
        end
    end
    return sparse(rows, cols, vals, N * nb, nb₊)
end

# Interleaved flat operator -> `SMatrix`-entried blocks. Row `N(β-1)+n` of `Rflat` is the
# source `m_β e_n` and column `N(j-1)+d` the coefficient of `m_j` in component `d`, so the
# block picks up `[d, n]` — solution direction down the columns, matching how a
# vector-valued kernel multiplies a density.
function _blockify(Rflat, nb, nb₊, ::Val{N}, ::Type{T}) where {N, T}
    SM = SMatrix{N, N, T, N * N}
    rows, cols, vals = Int[], Int[], SM[]
    for β in 1:nb, j in 1:nb₊
        blk = SM(Rflat[N * (β - 1) + n, N * (j - 1) + d] for d in 1:N, n in 1:N)
        iszero(blk) && continue
        push!(rows, β)
        push!(cols, j)
        push!(vals, blk)
    end
    return sparse(rows, cols, vals, nb, nb₊)
end

"""
    abstract type AbstractParticularBasis

Interpolation basis and matching particular solutions for one PDE — the **operator layer**,
built by `particular_basis(op, order)`:

    pb(c, r) -> (b, Ψ, γ₁Ψ)     with ℒΨ_β = b_β and γ₁Ψ_β the Neumann trace
    length(pb)                   # how many functions, i.e. the length of those vectors

This is what **adding a new operator** means, and in the general case a subtype supplies one
method — the solve — with the rest defaulted:

    solution_matrix(pb, r)     # Ψ̂, the coefficients of Ψ over `m₊` — the whole solve
    solution_space(pb)         # `m₊`      (default: `pb.m₊`)
    source_space(pb)           # `m`       (default: `pb.m`)
    source_closure(pb, c, r)   # `b`       (default: the monomials of `m`)
    trace_closure(pb, c, r)    # `γ₁Ψ`     (default: `∂_ν`)

Everything else — the values, the gradient, the derived bases — is built from `Ψ̂` below, so no
field is assumed of a subtype beyond what those accessors expose.

**Serving a new caller** is the other axis, and is not this: see
[`green_identity_terms`](@ref), which derives every volume operator from one Green identity
and needs nothing of a basis beyond the accessors above.
"""
abstract type AbstractParticularBasis end

"""
    solution_space(pb) -> m₊
    source_space(pb)   -> m

The monomial spaces the particular solutions and the interpolation basis live in. Defaults
read the `m₊`/`m` fields, which every basis so far has; a subtype storing them differently
overrides these two methods and nothing else.
"""
solution_space(pb::AbstractParticularBasis) = pb.m₊

@doc (@doc solution_space)
source_space(pb::AbstractParticularBasis) = pb.m

Base.length(pb::AbstractParticularBasis) = length(source_space(pb))
ambient_dimension(pb::AbstractParticularBasis) = ambient_dimension(solution_space(pb))

"""
    solution_matrix(pb, r) -> Ψ̂

Coefficients of the particular solutions over `m₊` in the scaled coordinate `x̃ = (x-c)/r`:

    Ψ(x) = r^s Ψ̂ m₊(x̃),   s = solution_order(pb)

The *only* place a particular solution is chosen, and the only element-dependent object in a
basis — `r` enters at all only because a shifted operator picks its branch on `κ² = c·r²`. Every
trace below is a matvec against the same `m₊(x̃)` with a matrix derived from this one.
"""
function solution_matrix end

"""
    solution_order(pb) -> s

The power of `r` carried by [`solution_matrix`](@ref): `2` for a particular solution, one
less per derivative taken of it (see [`source_derivative`](@ref)), since `Ψ` carries `r²`
and `∂ₓ = ∂x̃/r`.
"""
solution_order(::AbstractParticularBasis) = 2

"""
    source_closure(pb, c, r) -> b

The interpolation basis `b` on the element `(c, r)`, scalar for every operator (see
[`particular_basis`](@ref) for why). The monomials of `source_space(pb)` by default, evaluated
as a prefix of `m₊` so that nothing of higher degree is computed for them; only the tilted
branch of [`ShiftedLaplaceParticularBasis`](@ref) departs from them.
"""
source_closure(pb::AbstractParticularBasis, c, r) =
    x -> solution_space(pb)((x - c) / r, length(pb))

function (pb::AbstractParticularBasis)(c, r)
    Ψ̂ = solution_matrix(pb, r)
    return source_closure(pb, c, r), value_closure(pb, c, r, Ψ̂), trace_closure(pb, c, r, Ψ̂)
end

"""
    value_closure(pb, c, r, Ψ̂ = solution_matrix(pb, r)) -> Ψ
    trace_closure(pb, c, r, Ψ̂ = solution_matrix(pb, r)) -> γ₁Ψ

The particular solutions and their generalized Neumann trace on the element `(c, r)`. For every
scalar operator the trace is `∂_ν`, the default below, which the monomial recursion returns
alongside the values so that differentiation never enters coefficient space. The vector-valued
bases override it with the stored traction; see [`traction_matrix`](@ref).

Every closure derived from one element's solve — these two and [`gradient_solution`](@ref) —
takes `Ψ̂` as a trailing argument so a caller wanting several of them solves once. The default
keeps each usable on its own.
"""
function value_closure(
        pb::AbstractParticularBasis, c, r, Ψ̂ = solution_matrix(pb, r),
    )
    m₊, s = solution_space(pb), solution_order(pb)
    return x -> r^s * (Ψ̂ * m₊((x - c) / r))
end

@doc (@doc value_closure)
function trace_closure(
        pb::AbstractParticularBasis, c, r, Ψ̂ = solution_matrix(pb, r),
    )
    m₊, s = solution_space(pb), solution_order(pb)
    return (x, ν) -> r^(s - 1) * (Ψ̂ * last(m₊((x - c) / r, ν)))
end

"""
    gradient_solution(pb, c, r) -> ∇Ψ

`∇Ψ(x)`, as a vector of `SVector{N}` stacked over the basis index — the free term of the
`:gradient` identity, and the only trace that reads the solution transversally to a normal.
Coefficients only: `∂_d Ψ` is `Ψ̂ · derivative_matrix(m₊, d)`, so this costs `N` sparse products
at build time and one extra matvec per point. Takes a precomputed `Ψ̂` like the other closures
over one element's solve; see [`value_closure`](@ref).
"""
function gradient_solution(
        pb::AbstractParticularBasis, c, r, Ψ̂ = solution_matrix(pb, r),
    )
    m₊, s = solution_space(pb), solution_order(pb)
    # the `Val` is captured, not rebuilt from an `Int` inside the closure: a closure field is
    # only an `Int`, so `Val(N)` in the body would infer as a runtime type and the `ntuple`
    # below as an unknown-length tuple, making the returned vector's eltype abstract
    NV, nb = Val(ambient_dimension(m₊)), length(pb)
    Ĝ = ntuple(d -> Ψ̂ * derivative_matrix(m₊, d), NV)
    return function (x)
        v = m₊((x - c) / r)
        g = map(Ĝd -> Ĝd * v, Ĝ)
        return [r^(s - 1) * SVector(ntuple(d -> g[d][β], NV)) for β in 1:nb]
    end
end

"""
    source_derivative(pb, dirs::Integer...) -> pb′
    solution_derivative(pb, dirs::Integer...) -> pb′

Two different bases, both derivatives of one solve, and the distinction matters:

- `source_derivative` — the canonical solution of the *differentiated source*,
  `ℒΨ′_β = ∂_dirs b_β`. In coefficient space the source `∂_j m_α = Σ_γ (D_j)_{αγ} m_γ` is
  re-solved with the same right inverse, i.e. `D_j Ψ̂` — a **left** multiplication.
- `solution_derivative` — the *derivative of the solution*, `Ψ′_β = ∂_dirs Ψ_β`, which is
  `Ψ̂ D_j`, a **right** multiplication.

Both satisfy `ℒΨ′ = ∂ b` — `ℒ` has constant coefficients and so commutes with `∂` — so they
differ only by a homogeneous solution, but they are not interchangeable numerically. For `b_α`
constant, `∂_j b_α = 0`: `source_derivative` returns `Ψ′ = 0` exactly, while
`solution_derivative` returns `∂_jΨ_α`, a nonzero harmonic polynomial. Fed through a Green
identity the first contributes nothing and the second contributes the layer operators'
discretization error, which is why `W` and `X` want `source_derivative`; `X` separately wants
`solution_derivative`, for the `∂_cΦ′_β` of its `β`-dedup.

Where the source basis is not the monomials — the tilted branch, see
[`monomial_source`](@ref) — no coefficient matrix represents `∂_j b` and `source_derivative`
falls back to differentiating the solution. Still exact; it just loses the canonical solution's
extra property.

Either result is an `AbstractParticularBasis`, so the two nest freely and in either order, one
multiplying `Ψ̂` from the left and the other from the right. `source_closure` is inherited
unchanged: `W` and `X` interpolate the density in the *undifferentiated* `b`.
"""
source_derivative(pb::AbstractParticularBasis, dir::Integer) = DerivedBasis(pb, dir, true)

@doc (@doc source_derivative)
solution_derivative(pb::AbstractParticularBasis, dir::Integer) = DerivedBasis(pb, dir, false)

"""
    struct DerivedBasis <: AbstractParticularBasis

What [`source_derivative`](@ref) and [`solution_derivative`](@ref) return: a parent basis, one
direction, and which side of the solve to differentiate on. Stores no solve of its own, and
holds the *direction* rather than the matrix it becomes, because which side a source derivative
acts on depends on the element (see [`monomial_source`](@ref)). Repeated derivatives nest — a
`DerivedBasis` is a perfectly good parent — so no composition method is needed.
"""
struct DerivedBasis{P <: AbstractParticularBasis} <: AbstractParticularBasis
    parent::P
    dir::Int
    on_source::Bool
end

solution_space(pb::DerivedBasis) = solution_space(pb.parent)
source_space(pb::DerivedBasis) = source_space(pb.parent)
source_closure(pb::DerivedBasis, c, r) = source_closure(pb.parent, c, r)
# each derivative trades one power of `r`, since `Ψ` carries `r^s` and `∂ₓ = ∂x̃/r`
solution_order(pb::DerivedBasis) = solution_order(pb.parent) - 1

"""
    monomial_source(pb, r) -> Bool

Whether the interpolation basis of `pb` on an element of radius `r` *is* the monomials, so that
differentiating the source is `derivative_matrix(source_space(pb), j)`. True for every basis but
the tilted branch of [`ShiftedLaplaceParticularBasis`](@ref), where `b` is an `O(κ²)`
perturbation of them and no such matrix represents `∂_j b`.
"""
monomial_source(::AbstractParticularBasis, r) = true
# a derived basis inherits its parent's `b`, so it inherits the answer too
monomial_source(pb::DerivedBasis, r) = monomial_source(pb.parent, r)

function solution_matrix(pb::DerivedBasis, r)
    Ψ̂, m₊ = solution_matrix(pb.parent, r), solution_space(pb)
    (pb.on_source && monomial_source(pb.parent, r)) &&
        return derivative_matrix(source_space(pb), pb.dir) * Ψ̂
    # either the derivative is of the solution, or `b` is not the monomials (the tilted
    # branch) so the source has no coefficient matrix to differentiate; `∂_dΨ` solves the same
    # equation and is exact, forfeiting only the canonical solution's vanishing where `∂_d b`
    # does
    return Ψ̂ * derivative_matrix(m₊, pb.dir)
end

"""
    particular_basis(op, order, T = Float64) -> pb
    pb(c, r) -> (b, Ψ, γ₁Ψ)

Interpolation basis and matching particular solutions for `op`, of degree `≤ order`. Every
polynomial solve — all the element-*independent* work — happens once, here; calling the result
with an element's centre `c` and radius `r` gives three callables evaluating the *whole* basis
at once, each returning a vector stacked over the multiindices `β`:

- `b(x)`      — the basis values the density is interpolated in, always **scalar**;
- `Ψ(x)`      — particular solutions, `ℒ Ψ_β = b_β·Id`;
- `γ₁Ψ(x, n)` — the generalized Neumann traces of `Ψ_β` at `x` with unit normal `n`.

For a scalar operator `Id = 1` and this reads `ℒΨ_β = b_β`. For a vector-valued one `Ψ_β` and
`γ₁Ψ_β` are `SMatrix`es stacking the `N` solutions whose sources are `b_β e_n`, and `Id` is the
identity — so a column of `Ψ_β` is one particular solution and `b` stays scalar.

That `b` is scalar is not a convenience: every component of the density is interpolated in the
*same* space, so the interpolation is genuinely a scalar problem and the Vandermonde built from
`b` is a plain `Float64` matrix whatever the operator. A matrix-valued `b` would replace it by
`kron(V, I)` — the same factorization `N` times over inside one `N` times larger — and make the
solve ordering-ambiguous, a density multiplying a block from the right.

`x` is a physical point; the rescaling is baked into the callables. `ℒ Ψ_β = b_β` is exact and
is all a caller may assume — nothing constrains what `b_β` *is* beyond interpolating well over
the element. Each call returns a fresh vector, so a `pb` is safe to share between threads.

The concrete type returned depends on `op`, as with `factorize`, and callers never name it.
**Supporting a new operator is a new `<: AbstractParticularBasis` type and a method here.** Two
operators that solve nothing alike share no fields and no code, which is the point of splitting
the types rather than parametrizing one; where they *do* coincide they share free functions over
a [`MonomialBasis`](@ref), reached through the accessors above.

Everything is expanded in the isotropically scaled `x̃ = (x - c)/r`, which stays `O(1)` over the
element and the patch around it. Because `Δ_x = r⁻²Δ̃`, writing `Ψ = r²ψ(x̃)` makes
`ℒ_x Ψ = (ℒ̃ψ)(x̃)`: the tables need no element-dependent transformation, and `r` enters only as
the explicit powers the call method applies. A general affine map in place of `rI` was measured
to be no better and often much worse — it moves the element's shape from the Vandermonde into
the change of basis and gives up the normalization that keeps the monomials well scaled.
"""
function particular_basis end

"""
    struct LaplaceParticularBasis <: AbstractParticularBasis

Particular basis for `ℒ = -Δ`, built by [`particular_basis`](@ref). Stores the source basis `m`
of degree `n`, the space `m₊` of degree `n+2` its solutions live in, and the right inverse `R`
of `Δ` in coefficient space (`R·Λ₊ = [I 0]`), which is the solution outright up to the sign:
`Ψ = -R m₊`, with `b` the plain monomials.
"""
struct LaplaceParticularBasis{N, S} <: AbstractParticularBasis
    op::Laplace{N}
    m::MonomialBasis{N}
    m₊::MonomialBasis{N}
    R::S
end

"""
    struct ShiftedLaplaceParticularBasis <: AbstractParticularBasis

Particular basis for `ℒ = -(Δ + c)`, built by [`particular_basis`](@ref). Stores `m`, `m₊`, and
the inverses behind *both* branches of its solve, since which one an element takes is known
only from its `r`: the right inverse `R` of `Δ` for the tilted branch, and the inverse `S₁` of
`Δ + 1` for the `Λ`-series branch. `S₁` is stored padded with structural zeros to
`length(m) × length(m₊)`, the shape every [`solution_matrix`](@ref) has — the `Λ`-series
solution lives in `m`, but that padding is a property of the spaces, not of the element.

One type serves **both** operators of this form, since they differ in a *value* and not in
structure — the whole of the difference is [`pde_shift`](@ref): `c = k²` for `Helmholtz`, so
`κ² = (kr)² > 0`, and `c = -λ²` for `Yukawa` (Inti writes it `-Δu + λ²u`), so `κ² < 0`. The
negative shift stays in real arithmetic because everything downstream is a function of `κ²`
rather than `κ`; see [`shifted_laplace_solution_matrix`](@ref).

Nothing element-specialized is stored: `κ² = c·r²` is a property of an element, so the
rescaling happens in the call method.
"""
struct ShiftedLaplaceParticularBasis{Op, N, S} <: AbstractParticularBasis
    op::Op
    m::MonomialBasis{N}
    m₊::MonomialBasis{N}
    R::S
    S₁::S
end

"""
    pde_shift(op) -> c

The constant `c` in `ℒ = -(Δ + c)`, which is all that distinguishes the operators
[`ShiftedLaplaceParticularBasis`](@ref) serves: `k²` for `Helmholtz`, `-λ²` for `Yukawa`.
"""
pde_shift(op::Helmholtz) = op.k^2
pde_shift(op::Yukawa) = -op.λ^2

"""
    struct ElastostaticParticularBasis <: AbstractParticularBasis
    struct StokesParticularBasis <: AbstractParticularBasis

Particular bases for the two vector-valued operators, built by [`particular_basis`](@ref):

- `Elastostatic` — the Navier operator `ℒ = -(μΔ + (λ+μ)∇(∇·))`
- `Stokes`       — `ℒu = -μΔu + ∇p` with the constraint `∇·u = 0`

Each stores `m`, `m₊`, the right inverse `R` of `ℒ` (`R·ℒ₊ = [I 0]`) and the coefficients `γ₁R`
of the *traction* of those same solutions, both with `SMatrix{N,N}` entries.

`R` is the solution outright, as in [`LaplaceParticularBasis`](@ref) and for the same reason —
`ℒ` carries its own sign — so `b` is again the plain monomials and *scalar*: column `n` of `Ψ_β`
is the solution whose source is `b_β e_n`. See [`galerkin_inverse_matrix`](@ref), which builds
both from the *scalar* primitives via the Galerkin vector and differs between them only in
[`galerkin_parameter`](@ref).

`γ₁R` is the one field with no scalar counterpart, and why these bases do not use the generic
`∂_ν` trace: the traction is not a directional derivative, so it cannot ride the monomial
recursion; see [`traction_matrix`](@ref).

They are two types rather than one because they are two operators, but they share every line of
their construction and their call method; a future operator of this shape supplies a
`galerkin_parameter` and an `_isotropic_stress`.
"""
struct ElastostaticParticularBasis{N, T, S, S₁} <: AbstractParticularBasis
    op::Elastostatic{N, T}
    m::MonomialBasis{N}
    m₊::MonomialBasis{N}
    R::S
    γ₁R::S₁
end

@doc (@doc ElastostaticParticularBasis)
struct StokesParticularBasis{N, T, S, S₁} <: AbstractParticularBasis
    op::Stokes{N, T}
    m::MonomialBasis{N}
    m₊::MonomialBasis{N}
    R::S
    γ₁R::S₁
end

const VectorParticularBasis =
    Union{ElastostaticParticularBasis, StokesParticularBasis}

"""
    BASIS_CROSSOVER_KR

The value of `κ = |k|·r` at which [`ShiftedLaplaceParticularBasis`](@ref) switches branches: the
tilted basis below it, plain monomials and the terminating `Λ`-series above. Both branches
are exact, so this trades the accuracy of two *interpolation* bases against each other.
"""
const BASIS_CROSSOVER_KR = 0.3

function particular_basis(op::Laplace, order, ::Type{T} = Float64) where {T}
    m, m₊ = _monomial_spaces(op, order)
    return LaplaceParticularBasis(op, m, m₊, laplace_inverse_matrix(m, T))
end

function particular_basis(
        op::Union{Helmholtz, Yukawa}, order, ::Type{T} = float(typeof(pde_shift(op))),
    ) where {T}
    m, m₊ = _monomial_spaces(op, order)
    S₁ = shifted_laplace_inverse_matrix(m, T)
    S₁ = hcat(S₁, spzeros(T, length(m), length(m₊) - length(m)))   # padded once, see the struct
    return ShiftedLaplaceParticularBasis(op, m, m₊, laplace_inverse_matrix(m, T), S₁)
end

for (Op, Basis) in
    ((:Elastostatic, :ElastostaticParticularBasis), (:Stokes, :StokesParticularBasis))
    @eval function particular_basis(op::$Op, order, ::Type{T} = Float64) where {T}
        m, m₊ = _monomial_spaces(op, order)
        R, S = galerkin_inverse_matrix(op, m, T)
        return $Basis(op, m, m₊, R, traction_matrix(op, R, S, m₊, T))
    end
end

# the source space and the two-degrees-higher space every solution so far lands in
function _monomial_spaces(op, order)
    N = ambient_dimension(op)
    return MonomialBasis(Val(N), order), MonomialBasis(Val(N), order + 2)
end

# The per-operator halves of the contract documented on `particular_basis`: a
# `solution_matrix` each, a `source_closure` only where `b` is not the monomials, and a
# `trace_closure` only where the Neumann trace is not `∂_ν`. Everything else is generic.

# `ℒ = -Δ`, so `Ψ̂ = -R` outright and `b` is the plain monomials, the leading `nb` entries of
# the very vector `Ψ` is evaluated over.
solution_matrix(pb::LaplaceParticularBasis, r) = -pb.R

# `ℒ = -(Δ + c)`, and there is a real choice to make, on `κ² = c·r²`. Since `ℒΨ = b` is the
# *only* thing a caller assumes, nothing says `b` has to be the monomials, and that freedom is
# what makes the small-`κ` limit tractable:
#
#   `|κ| < BASIS_CROSSOVER_KR` — keep the Laplace solution and tilt the basis. With `Ψ = -R m₊`,
#     `ℒΨ = -(Δ̃ + κ²)Ψ = m + κ²R m₊`, so `ℒ`'s constant term moves into `b`, an `O(κ²)`
#     perturbation of the monomials. Exact, no solve beyond the shift-independent Laplace one,
#     and continuous into Laplace at `κ = 0`.
#   `|κ| ≥ BASIS_CROSSOVER_KR` — plain monomials and the exact polynomial solution from
#     `shifted_laplace_solution_matrix`, padded out to `m₊`.
#
# Each branch degrades where the other is used: the exact solution grows like `κ⁻²` as `κ → 0`
# (it is unique, so nothing can tame it), while the tilt takes `b` further from the monomials as
# `κ` grows.
_kappa2(pb::ShiftedLaplaceParticularBasis, r) = pde_shift(pb.op) * r^2
_tilted(pb::ShiftedLaplaceParticularBasis, r) = abs(_kappa2(pb, r)) < BASIS_CROSSOVER_KR^2
# only the tilted branch departs from the monomials
monomial_source(pb::ShiftedLaplaceParticularBasis, r) = !_tilted(pb, r)

function solution_matrix(pb::ShiftedLaplaceParticularBasis, r)
    _tilted(pb, r) && return -pb.R
    # `pb.S₁` is stored already padded to `length(m₊)` columns, the shape every
    # `solution_matrix` has, so the rescaling is all that happens per element
    return -shifted_laplace_solution_matrix(
        pb.S₁, degrees(solution_space(pb)), _kappa2(pb, r)
    )
end

function source_closure(pb::ShiftedLaplaceParticularBasis, c, r)
    m₊, nb = solution_space(pb), length(pb)
    _tilted(pb, r) || return x -> m₊((x - c) / r, nb)
    # the Laplace solution is kept as is, so the shift term of `ℒ` lands on `b` instead
    κ², R = _kappa2(pb, r), pb.R
    return function (x)
        v = m₊((x - c) / r)
        return v[1:nb] + κ² * (R * v)
    end
end

# `ℒ = -(μΔ + (λ+μ)∇∇·)` carries its own sign, so — as for Laplace — the right inverse is the
# solution outright and `b` is the plain monomials, here times the identity because the source of
# `Ψ_β`'s `n`-th column is `m_β e_n`. What does not carry over is the trace: `Ψ` and `γ₁Ψ` are
# the *same* matvec against the *same* monomial values, differing only in which stored matrix
# they apply, the traction having been contracted into `γ₁R` at construction.
solution_matrix(pb::VectorParticularBasis, r) = pb.R

# the traction was contracted into `γ₁R` at construction, so this trace ignores `Ψ̂`
function trace_closure(pb::VectorParticularBasis, c, r, Ψ̂ = nothing)
    m₊, γ₁R = solution_space(pb), pb.γ₁R
    N, nb, nb₊ = ambient_dimension(m₊), length(pb), length(m₊)
    # one power of `r` survives, as `Ψ` carries `r²` and `∂ₓ = ∂x̃/r`; `σ`'s columns are
    # contracted against `ν` here, the only place `ν` can enter
    return function (x, ν)
        σ = γ₁R * m₊((x - c) / r, nb₊)
        return [r * sum(e -> ν[e] * σ[N * (β - 1) + e], 1:N) for β in 1:nb]
    end
end

# Differentiating a *vector* solution would need the traction of the derivative, i.e. a
# `traction_matrix` rebuilt from the differentiated coefficients. `vdim`'s W and X variants
# are scalar-only, so that is left unwritten rather than guessed at.
for f in (:source_derivative, :solution_derivative)
    @eval function $f(::VectorParticularBasis, ::Integer)
        return error("$($f) is not implemented for vector-valued operators")
    end
end

# ---------------------------------------------------------------------------
# Green identities: the terms one volume operator needs
# ---------------------------------------------------------------------------

"""
    green_identity_terms(pb, variant, c, r) -> (b, γ₀, γ₁, σ)

The terms of the Green identity for the volume operator named by `variant`, on the element
`(c, r)`: for a density interpolated in `b`,

    S[γ₁_β](x) - D[γ₀_β](x) + μ(x)σ_β(x) + 𝒱[b_β](x) = 0                          (1)

with `(S, D)` the layer pair `variant` is written against — *boundary* integrals plus a
*pointwise* free term. This is [`_green_multiplier`](@ref)'s convention, the one `bdim` states
its identity in, extended by the volume term; solved for the potential at an interior target,
where `μ = -1`, it reads `𝒱[b_β] = σ_β + D[γ₀_β] - S[γ₁_β]`. For `:default`, `𝒱 = V` and this is
the base identity with `ℒΨ_β = b_β`, whose terms are the very triple `pb(c, r)` returns.

Nothing here solves anything, because every other variant is *the same identity on a
different member of the same family*: [`source_derivative`](@ref) and
[`solution_derivative`](@ref) return an `AbstractParticularBasis` again, and
[`gradient_solution`](@ref) is a third way of evaluating one. Two rules generate them from
(1):

- **R2** a target derivative of a *volume* term becomes a source derivative. From
  `∂ₓG = -∂_yG` and the divergence theorem, `∂_c V[h] = V[∂_c h] - S[h ν_c]`.
- **R3** a target derivative of a *boundary* term is irreducible: `∂_c S[φ]` is a new
  operator `∇ₓS`, and nothing moves it back.

| `variant`          | family (1) runs on             | `σ`  | by-parts terms                | layer pair         |
|:-------------------|:-------------------------------|:-----|:------------------------------|:-------------------|
| `:default`         | `pb`                           | `γ₀` | —                             | `S`, `D`           |
| `:gradient`        | `pb`                           | `∇Ψ` | —                             | `∇ₓS`, `∇ₓD`       |
| `:gradient_source` | `source_derivative(pb, j)`     | `γ₀` | `S[b ν_j]`                    | `S`, `D`           |
| `:hessian`         | `∂_c source_derivative(pb, j)` | `γ₀` | `S[(∂_j b)ν_c]`, `∇ₓS[b ν_j]` | `S`, `D` and `∇ₓS` |

`σ` is `γ₀` evaluated at the target rather than on the boundary in every row but `:gradient`
— the one row where the derivative went onto the kernels (R3) instead of the source (R2), so
the pointwise term is the only place left to differentiate by hand.

A by-parts term is R2's surface term, an extra source fed to a layer operator, and not a
choice of regularization. It needs no slot of its own: entering `S` added to `γ₁` is what
`:gradient_source`'s `+ b ν_j` below is — but that holds *one* such term, on the operator `γ₁`
already feeds. `:hessian` needs **two, on two different operators**, which is exactly one more
than (1) can carry; its row above is where it sits in this derivation, not a variant this
function returns, and [`_vdim_correction_X`](@ref) assembles it from the derived bases
directly.

The `j` of a source derivative is a *density* index, so `:gradient_source` stacks its terms as
`SVector`s over it while the layer operators stay the plain scalar pair. `b` is never
differentiated: it is what the density is interpolated in.
"""
green_identity_terms(pb::AbstractParticularBasis, variant::Symbol, c, r) =
    green_identity_terms(pb, Val(variant), c, r)

function green_identity_terms(pb::AbstractParticularBasis, ::Val{:default}, c, r)
    b, Ψ, γ₁Ψ = pb(c, r)
    return b, Ψ, γ₁Ψ, Ψ
end

function green_identity_terms(pb::AbstractParticularBasis, ::Val{:gradient}, c, r)
    Ψ̂ = solution_matrix(pb, r)
    return source_closure(pb, c, r), value_closure(pb, c, r, Ψ̂),
        trace_closure(pb, c, r, Ψ̂), gradient_solution(pb, c, r, Ψ̂)
end

function green_identity_terms(pb::AbstractParticularBasis, ::Val{:gradient_source}, c, r)
    # `NV` captured as a `Val`, for the reason given in `gradient_solution`
    NV, nb = Val(ambient_dimension(pb)), length(pb)
    b = source_closure(pb, c, r)
    pbj = ntuple(j -> source_derivative(pb, j)(c, r), NV)
    γ₀ = function (x)
        v = map(p -> p[2](x), pbj)
        return [SVector(ntuple(j -> v[j][β], NV)) for β in 1:nb]
    end
    γ₁ = function (x, ν)
        dv, bv = map(p -> p[3](x, ν), pbj), b(x)
        return [SVector(ntuple(j -> dv[j][β] + bv[β] * ν[j], NV)) for β in 1:nb]
    end
    return b, γ₀, γ₁, γ₀
end
