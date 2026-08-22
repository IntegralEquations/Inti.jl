"""
    _dim_correction(source, dict_near, bfun, Θ, Tw) -> δV

Sparse correction of a volume potential from the right-hand side `Θ` of the Green identity
(`num_target × num_basis`) and the batched interpolation basis `bfun`: on every source element
with near targets, interpolate `Θ` in `bfun` over the element's own quadrature nodes and store
the resulting weights, of type `Tw`.

Everything the method does once `Θ` is known. `lvdim` assembles `Θ` one element at a time and so
calls [`_dim_solve!`](@ref) directly, rather than through this loop.
"""
function _dim_correction(source, dict_near, bfun, Θ, ::Type{Tw}) where {Tw}
    Is, Js, Vs = Int[], Int[], Tw[]
    do_debug = debug_mode()
    vander_cond = vander_norm = -Inf
    for (E, qtags) in source.etype2qtags
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        # each element contributes exactly `nq * length(near)` entries, so the output is
        # sized up front and every element fills a disjoint slice
        counts = [nq * length(near) for near in near_list]
        offs = length(Is) .+ cumsum(counts) .- counts
        for v in (Is, Js, Vs)
            resize!(v, length(v) + sum(counts))
        end
        for n in 1:ne
            isempty(near_list[n]) && continue
            jglob = @view qtags[:, n]
            L = vandermonde(bfun, (coords(source[j]) for j in jglob))
            if do_debug
                vander_cond = max(vander_cond, cond(L))
                vander_norm = max(vander_norm, norm(L))
            end
            # basis index first, as `_dim_solve!` wants it; a permuted view rather than a
            # `transpose`, which would also transpose the `SMatrix` entries
            Θt = PermutedDimsArray(view(Θ, near_list[n], :), (2, 1))
            _dim_solve!(Is, Js, Vs, offs[n], L, Θt, jglob, near_list[n])
        end
    end
    @debug """Condition properties of the vdim correction:
    |-- max interp. matrix condition: $vander_cond
    |-- max interp. matrix norm:      $vander_norm
    """
    return sparse(Is, Js, Vs, size(Θ, 1), length(source))
end

"""
    _dim_interpolation_order(source) -> order

Default degree of the interpolation basis for a density integrated by `source`: the
[`interpolation_order`](@ref) of `source`'s own quadrature rules, the *weakest* of them on a
mixed mesh.

The density is interpolated on each element's quadrature nodes, so the basis must be one those
nodes can determine — which is what a rule's interpolation order reports, and it is well below
its quadrature order. A Vioreanu-Rokhlin triangle rule of quadrature order 7 has 15 nodes and
interpolation order 4: asking for the quadrature order would pose 36 basis functions on them.
"""
_dim_interpolation_order(source::Quadrature) =
    minimum(interpolation_order, values(source.etype2qrule))

"""
    _dim_solve!(Is, Js, Vs, off, L, Θ, jglob, near, work = _dim_work(L, Θ, length(near)))

Solve the local interpolation system of the density interpolation method for one element and
write that element's block of `δV` into the preallocated `(Is, Js, Vs)`, starting at `off`.

- `L`     — the interpolation basis' Vandermonde on the element's own quadrature nodes,
            `num_basis × nq`, and always **scalar** (see [`particular_basis`](@ref)).
            **Overwritten**: it is factored in place, and both callers rebuild it for the
            next element anyway;
- `Θ`     — the right-hand side of the Green identity, `num_basis × length(near)`, i.e.
            transposed relative to the correction it becomes;
- `jglob` — the element's global quadrature-node indices, `near` the targets it corrects.

The *same* for every variant: `vdim_correction`'s four kernel variants and `lvdim_correction`
differ only in how `Θ` is assembled. Whatever the entries of `Θ` are, a density multiplies them
componentwise, so the system decouples into [`_dim_ncomponents`](@ref) scalar right-hand sides
sharing one factorization of `L`. For a vector-valued PDE that is what replaces the
`num_basis·N × nq·N` system `kron(L, I)` would pose: the same solve, once instead of `N` times.
"""
function _dim_solve!(
        Is, Js, Vs, off, L, Θ, jglob, near, work = _dim_work(L, Θ, length(near)),
    )
    nb, nq = size(L)
    nc, nt = _dim_ncomponents(eltype(Θ)), length(near)
    # `svd!`: `L` is the caller's Vandermonde and is rebuilt for the next element anyway
    F = svd!(L)
    # every target is a right-hand side against the *same* factorization, so they are posed
    # as one `nb × nt·nc` block and solved once: `Θ` is already that block up to the
    # component flattening, and one `ldiv!` on it replaces `nt` of them on a single column
    bdata = view(work.bdata, :, 1:(nt * nc))
    wdata = view(work.wdata, :, 1:(nt * nc))
    for t in 1:nt
        blk = view(bdata, :, ((t - 1) * nc + 1):(t * nc))
        for m in 1:nb
            _dim_components!(blk, m, Θ[m, t])
        end
    end
    ldiv!(wdata, F, bdata)
    @inbounds for (t, i) in enumerate(near)
        blk = view(wdata, :, ((t - 1) * nc + 1):(t * nc))
        # column-major over the block: target slowest, the element's own nodes fastest
        for s in 1:nq
            q = off + (t - 1) * nq + s
            Is[q], Js[q] = i, jglob[s]
            Vs[q] = -_dim_weight(eltype(Vs), blk, s)
        end
    end
    return nothing
end

"""
    _dim_work(L, Θ, nt) -> (; bdata, wdata)

Right-hand-side and solution storage for [`_dim_solve!`](@ref), for at most `nt` targets.
Split out so that a caller looping over elements — `lvdim` — builds it once per task rather
than once per element; it is sized for the largest element that task will see and used
through a view of the leading columns.
"""
function _dim_work(L, Θ, nt)
    nb, nq = size(L)
    T, nc = _dim_scalar_type(eltype(Θ)), _dim_ncomponents(eltype(Θ))
    return (
        bdata = Matrix{T}(undef, nb, nt * nc), wdata = Matrix{T}(undef, nq, nt * nc),
    )
end

"""
    _dim_ncomponents(T) -> nc
    _dim_scalar_type(T) -> S

The number of scalar components of one entry of `Θ`, and the scalar type they live in —
the width and element type of the right-hand side [`_dim_solve!`](@ref) poses.
"""
_dim_ncomponents(::Type{<:Number}) = 1
_dim_ncomponents(::Type{T}) where {T <: StaticArray} = length(T) * _dim_ncomponents(eltype(T))

@doc (@doc _dim_ncomponents)
_dim_scalar_type(::Type{T}) where {T <: Number} = T
_dim_scalar_type(::Type{T}) where {T <: StaticArray} = _dim_scalar_type(eltype(T))

# `Θ[m, t]` spread over row `m` of the right-hand side, and the inverse map rebuilding one entry
# of the correction out of row `s` of the solution. The two need only be inverse to each other:
# the solve treats the columns independently, so any consistent flattening gives the same
# answer.
_dim_components!(bdata, m, val::Number) = (bdata[m, 1] = val)
# `Tuple`, not the value itself: broadcasting an `SMatrix` into a row would compare shapes
_dim_components!(bdata, m, val::StaticArray{<:Any, <:Number}) = (bdata[m, :] .= Tuple(val))
function _dim_components!(bdata, m, val::StaticArray{<:Any, <:StaticArray})
    nc = _dim_ncomponents(eltype(val))
    for (k, blk) in enumerate(val)
        bdata[m, ((k - 1) * nc + 1):(k * nc)] .= Tuple(blk)
    end
    return bdata
end

_dim_weight(::Type{T}, wdata, s) where {T <: Number} = wdata[s, 1]
_dim_weight(::Type{T}, wdata, s) where {T <: StaticArray{<:Any, <:Number}} =
    T(ntuple(c -> wdata[s, c], length(T)))
function _dim_weight(::Type{T}, wdata, s) where {T <: StaticArray{<:Any, <:StaticArray}}
    B = eltype(T)
    nc = _dim_ncomponents(B)
    return T(ntuple(k -> B(ntuple(c -> wdata[s, (k - 1) * nc + c], nc)), length(T)))
end
# `W`'s correction maps an `SVector` density to a scalar, so its entries are row vectors
_dim_weight(::Type{Transpose{S, V}}, wdata, s) where {S, V} =
    transpose(_dim_weight(V, wdata, s))

"""
    vdim_correction(op,X,Y,Y_boundary,S,D,V; green_multiplier, kwargs...)

Compute a correction to the volume potential `V : Y → X` such that `V + δV` is a
more accurate approximation of the underlying volume potential operator. The
correction is computed using the (volume) density interpolation method.

This function requires a `op::AbstractDifferentialOperator`, a target set `X`, a source
quadrature `Y`, a boundary quadrature `Y_boundary`, approximations `S :
Y_boundary -> X` and `D : Y_boundary -> X` to the single- and double-layer
potentials (correctly handling nearly-singular integrals), and a naive
approximation of the volume potential `V`. The `green_multiplier` is a vector of
the same length as `X` storing the value of `μ(x)` for `x ∈ X` in the Green
identity (see [`_green_multiplier`](@ref)).

See [anderson2024fast](@cite) for more details on the method.

## Optional `kwargs`:

- `interpolation_order`: the order of the polynomial interpolation. Defaults to
  [`_dim_interpolation_order`](@ref)`(Y)`, the interpolation order of `Y`'s own
  quadrature rules.
- `maxdist`: distance beyond which interactions are considered sufficiently far
  so that no correction is needed. This is used to determine a threshold for
  nearly-singular corrections.
- `kernel_variant`: which volume operator is being corrected — `:default` for `V`,
  `:gradient` for `∇V`, `:gradient_source` for `W[g] = -∫∇yG⋅g`, `:hessian` for
  `X[g] = ∇W[g]`. This selects the Green identity, and hence which operators must
  be passed: `:default`/`:gradient` want `S`, `D` and `V` all built with the same
  variant, while `:gradient_source` and `:hessian` want the *scalar* `S`/`D`
  alongside a `V` that is the gradient (resp. Hessian) volume operator. See
  [`particular_basis`](@ref) for the derivation.
- `grad_single_layer`: the gradient single-layer `∇ₓS`; required by `:hessian` and
  unused otherwise.
"""
function vdim_correction(
        op::AbstractDifferentialOperator{N},
        target,
        source::Quadrature{N},
        boundary::Quadrature,
        Sop,
        Dop,
        Vop;
        green_multiplier::Vector{<:Real},
        interpolation_order = nothing,
        kernel_variant::Symbol = :default,
        maxdist = Inf,
        grad_single_layer = nothing,
    ) where {N}
    # `X` is the one variant needing two by-parts boundary terms, on two different operators —
    # one more than `green_identity_terms` carries — so it keeps its own assembly.
    if kernel_variant === :hessian
        isnothing(grad_single_layer) &&
            error("kernel_variant = :hessian requires the `grad_single_layer` operator")
        return _vdim_correction_X(
            op, target, source, boundary, Sop, Dop, grad_single_layer, Vop;
            green_multiplier, interpolation_order, maxdist,
        )
    end
    Tout = eltype(Vop)
    Tbase = default_kernel_eltype(op)
    # determine type for dense matrices
    DenseOut = Tout <: SMatrix ? BlockArray : Array
    DenseBase = Tbase <: SMatrix ? BlockArray : Array
    num_target, num_source = length(target), length(source)
    # a reasonable interpolation_order if not provided
    isnothing(interpolation_order) &&
        (interpolation_order = _dim_interpolation_order(source))
    # One basis for the whole mesh, centered at the origin and unscaled: unlike
    # `lvdim_correction`, `Θ` here comes from *global* forward maps over whole columns, so a
    # single frame must serve every element.
    pb = particular_basis(op, interpolation_order)
    c, r = zero(SVector{N, Float64}), 1.0
    bfun, γ₀Ψ, γ₁Ψ, σfun = green_identity_terms(pb, kernel_variant, c, r)
    dict_near = etype_to_nearest_points(target, source; maxdist)
    num_basis = length(pb)
    # `W`'s density carries a direction index the layer operators never see, so its traces are
    # `SVector`s over `nc = N` components against *scalar* `Sop`/`Dop`. Every other variant has
    # `nc = 1` and lets the operator carry whatever structure the output has.
    nc = kernel_variant === :gradient_source ? N : 1
    if nc == 1
        @assert eltype(Dop) == eltype(Sop) == Tout "eltype of Sop, Dop, and Vop must match"
    else
        @assert eltype(Dop) == eltype(Sop) == Tbase "Sop and Dop must be the scalar layers"
    end
    Trace = nc == 1 ? DenseBase{Tbase} : Matrix{SVector{nc, Tbase}}
    b = DenseBase{Tbase}(undef, length(source), num_basis)
    γ₀B = Trace(undef, length(boundary), num_basis)
    γ₁B = Trace(undef, length(boundary), num_basis)
    # The basis is *batched* — one call per node yields all `num_basis` values — so these
    # fill row-wise
    for j in 1:length(source)
        v = bfun(coords(source[j]))
        for k in 1:num_basis
            b[j, k] = _vdim_basis_value(v[k], Tbase)
        end
    end
    for j in 1:length(boundary)
        x, ν = coords(boundary[j]), normal(boundary[j])
        v, dv = γ₀Ψ(x), γ₁Ψ(x, ν)
        for k in 1:num_basis
            γ₀B[j, k] = v[k]
            γ₁B[j, k] = dv[k]
        end
    end
    Θ = DenseOut{Tout}(undef, num_target, num_basis)
    fill!(Θ, zero(Tout))
    # Compute Θ <-- S * γ₁B - D * γ₀B + V * b + σ * B(x) using in-place matvec
    if nc > 1
        # One scalar application of `Sop`/`Dop` per density direction, its result landing in
        # that same component of `Θ`. The volume term needs no such split: by R2 the density
        # direction *is* the gradient direction, so `Vop * b` already delivers all `nc`.
        g₀ = Vector{Tbase}(undef, length(boundary))
        g₁ = Vector{Tbase}(undef, length(boundary))
        for n in 1:num_basis
            vol = Vop * view(b, :, n)
            for i in 1:num_target
                Θ[i, n] += vol[i]
            end
            for cc in 1:nc
                for j in 1:length(boundary)
                    g₀[j] = γ₀B[j, n][cc]
                    g₁[j] = γ₁B[j, n][cc]
                end
                sc, dc = Sop * g₁, Dop * g₀
                for i in 1:num_target
                    Θ[i, n] += Tout(
                        ntuple(d -> d == cc ? sc[i] - dc[i] : zero(Tbase), nc),
                    )
                end
            end
        end
    elseif DenseOut <: Array || (Sop isa BlockArray && Dop isa BlockArray && Vop isa BlockArray)
        for n in 1:num_basis
            @views mul!(Θ[:, n], Sop, γ₁B[:, n])
            @views mul!(Θ[:, n], Dop, γ₀B[:, n], -1, 1)
            @views mul!(Θ[:, n], Vop, b[:, n], 1, 1)
        end
    else
        # For vector-valued problems with FMM (LinearMap operators), we need to
        # perform multiplication column-by-column since FMM expects vector densities
        # (SVector) not matrix densities (SMatrix). See bdim.jl for similar handling.
        P, Q = size(Tout)
        S = eltype(Tout)
        Θ_data = parent(Θ)::Matrix
        γ₀B_data = parent(γ₀B)::Matrix
        γ₁B_data = parent(γ₁B)::Matrix
        b_data = parent(b)::Matrix
        for k in 1:size(Θ_data, 2)
            y = reinterpret(SVector{P, S}, @view Θ_data[:, k])
            x = reinterpret(SVector{Q, S}, @view γ₁B_data[:, k])
            mul!(y, Sop, x)
            x = reinterpret(SVector{Q, S}, @view γ₀B_data[:, k])
            mul!(y, Dop, x, -1, 1)
            x = reinterpret(SVector{Q, S}, @view b_data[:, k])
            mul!(y, Vop, x, 1, 1)
        end
    end
    # Add the free term `μ(x)σ(x)`
    for i in 1:num_target
        v = σfun(coords(target[i]))
        for n in 1:num_basis
            Θ[i, n] += green_multiplier[i] * v[n]
        end
    end
    # `W`'s correction contracts an `SVector` density to a scalar, hence the adjoint entry
    Tw = nc == 1 ? Tout : Transpose{Tbase, Tout}
    return _dim_correction(source, dict_near, bfun, Θ, Tw)
end

# `particular_basis` returns a *scalar* interpolation basis for every operator: for a
# vector-valued PDE the source of `Ψ_β`'s `n`-th column is `b_β e_n`, i.e. the identity is
# implicit. The volume term `Vop * b` is typed by `Tbase`, so spell it out there.
_vdim_basis_value(v, ::Type{T}) where {T <: Number} = T(v)
_vdim_basis_value(v, ::Type{SM}) where {SM <: SMatrix} = v * one(SM)


"""
    _vdim_correction_X(op, target, source, boundary, Sop, Dop, GSop, Vop; kwargs...)

VDIM correction for the operator `X[g] = ∇W[g] = S·g(x) - PV∫∇ₓ∇_yG(x,y)·g(y)dy`
(vector density `g`, vector output), using the regularization (3.12) of
[anderson2026general](@cite).

Here `Sop`, `Dop` are the (scalar) single-/double-layer, `GSop` is the gradient
single-layer (`∇ₓS`, scalar density -> `SVector` output), and `Vop` is the
Hessian volume operator (`SVector` density -> `SVector` output,
`SMatrix` entries) whose action equals the principal-value part `+∫∇ₓ∇ₓG·g`.

Per basis monomial `pₐ` and target `i`, `Θ[i,α] ∈ SMatrix{N,N}` collects, in its
`j`-th column, the (3.12) representation of `X[pₐeⱼ](xᵢ)` together with the naive
volume term; the local interpolation system is solved component-wise against the
scalar Vandermonde and the weights are stored as `SMatrix`es so the resulting
sparse correction maps an `SVector{N}` density to an `SVector{N}` output.

This is the one variant that builds its polynomial data here rather than from
[`green_identity_terms`](@ref). Taking `∂_c V[h] = V[∂_c h] - S[h ν_c]` twice
leaves the `Ψ` part of the identity on the *plain* `S`/`D` — hence `Sop`/`Dop`
above — and produces *two* by-parts terms rather than one: `S[(∂ⱼpₐ)ν_c]`, which
rides `γ₁` as usual, and `W`'s leftover `S[pₐνⱼ]`, which can only be
differentiated as `∇ₓS[pₐνⱼ]`, hence `GSop`. One by-parts term is all the Green
identity of [`green_identity_terms`](@ref) carries, which is why this variant is
assembled here.

(Implementation notes: This routine makes no use of Green's function isotropy
which would allow to exploit Φ'_{σ(β)} = σ(Φ'_β) for a coordinate permutation σ
and polynomial solution Φ. The volume term is batched across the coordinate
directions only for the FMM charge→Hessian operator — a single scalar→`SMatrix`
pass per monomial yields the full Mₐ = ∫∇ₓ∇ₓG pₐ, avoiding N dipole passes; this
only works if the fast algorithm for `Vop` supports a charge→Hessian operation.
The dense (`BlockArray`) operator and the boundary `S`/`D`/`∇ₓS` terms are
applied per coordinate direction.)
"""
function _vdim_correction_X(
        op::AbstractDifferentialOperator{N},
        target,
        source::Quadrature{N},
        boundary::Quadrature,
        Sop,
        Dop,
        GSop,
        Vop;
        green_multiplier::Vector{<:Real},
        interpolation_order = nothing,
        maxdist = Inf,
    ) where {N}
    T = default_kernel_eltype(op)          # scalar element type (Float64 / ComplexF64)
    SV = SVector{N, T}                      # density / output element type
    SM = SMatrix{N, N, T, N * N}            # Θ / correction entry type
    # `Vop` is the Hessian volume operator; its dense form has
    # `SMatrix` entries while the FMM form is a `LinearMap` with `SVector` output
    # (both map an `SVector` density to an `SVector` output).
    @assert eltype(Vop) in (SM, SV) "Vop must be the Hessian volume operator"
    @assert eltype(GSop) == SV "GSop must be the gradient single-layer operator (SVector output)"
    num_target, num_source = length(target), length(source)
    isnothing(interpolation_order) &&
        (interpolation_order = _dim_interpolation_order(source))
    # the operator layer, not a variant: `X` needs `Υₐⱼ = ∇Ψₐⱼ` and the conormal
    # `BνΥₐⱼ = (HessΨₐⱼ)·ν`, second and third derivatives of the one solve `ℒΨₐⱼ = ∂ⱼpₐ`
    pb = particular_basis(op, interpolation_order)
    c, r = zero(SVector{N, Float64}), 1.0
    bfun = first(pb(c, r))
    # `Υ[j](x)[α]` = ∇Ψₐⱼ(x), an `SVector{N}`
    Υ = ntuple(j -> gradient_solution(source_derivative(pb, j), c, r), N)
    num_basis = length(pb)
    dict_near = etype_to_nearest_points(target, source; maxdist)

    # source-node monomial values (scalar) for the volume term and the Vandermonde
    b = Matrix{T}(undef, num_source, num_basis)
    for j in 1:num_source
        v = bfun(coords(source[j]))
        for k in 1:num_basis
            b[j, k] = v[k]
        end
    end
    # multi-index `I` of each basis monomial. Must be `monomial_exponents`, the ordering the
    # basis itself uses — a raw `Iterators.product` has the same length but a different order,
    # which would silently mispair `indices[n]` with basis function `n` below.
    indices = monomial_exponents(N, interpolation_order)
    @assert length(indices) == num_basis
    # grad_single_trace (pₐν) on the boundary, for the ∇ₓS term
    nbnd = length(boundary)
    γs = Matrix{SV}(undef, nbnd, num_basis)
    for q in 1:nbnd
        x, ν = coords(boundary[q]), normal(boundary[q])
        v = bfun(x)
        for k in 1:num_basis
            γs[q, k] = v[k] * ν
        end
    end

    # --- Optimization: dedup the S/D boundary applications over β = I − eⱼ ---
    # `basis_from_monomial` is linear, so Ψₙⱼ = Iⱼ·Φ′_β with β = I − eⱼ and ℒΦ′_β = yᵝ;
    # hence Υₙⱼ = Iⱼ·∇Φ′_β and the two (3.12) layer terms factor through β alone:
    #   D[Υₙⱼ]_c             = Iⱼ · D[∂_cΦ′_β]                (γ₀ trace ∂_cΦ′_β)
    #   S[(∂ⱼpₐ)ν + BνΥₙⱼ]_c = Iⱼ · S[τ_{β,c}],  τ_{β,c} = p_β ν_c + ∂ν(∂_cΦ′_β)
    # Thus, the nominal per-(n,j,c) applications in the above description can be
    # reduced to per-(β,c), β over |β| ≤ order−1, each scaled by Iⱼ in the
    # assembly below.
    # `β` ranges over the *lower-degree* basis, which the graded ordering makes a prefix of
    # the full one — so `β2k` is just the position in `monomial_exponents` and the `β` data
    # is a prefix of the same solve. `Φ′_β` is the basis's own `Ψ`, `∂_cΦ′_β` its gradient,
    # and `∂ν(∂_cΦ′_β)` the `∂_ν` trace of `solution_derivative(pb, c)` — the dedup is now
    # exactly the sparsity of `derivative_matrix`, at no extra cost.
    βindices = monomial_exponents(N, interpolation_order - 1)
    β2k = Dict(B => k for (k, B) in enumerate(βindices))
    nβ = length(βindices)
    ∇Φ = gradient_solution(pb, c, r)
    ∂νΦc = ntuple(d -> last(solution_derivative(pb, d)(c, r)), N)
    g₀ = Matrix{T}(undef, nbnd, nβ * N)    # γ₀ traces ∂_cΦ′_β, column index (β,c) = (k-1)N+c
    g₁ = Matrix{T}(undef, nbnd, nβ * N)    # γ₁ traces τ_{β,c}
    for q in 1:nbnd
        xq, nu = coords(boundary[q]), normal(boundary[q])
        pv = bfun(xq)
        gv = ∇Φ(xq)
        hv = ntuple(d -> ∂νΦc[d](xq, nu), N)
        for k in 1:nβ, cc in 1:N
            g₀[q, (k - 1) * N + cc] = gv[k][cc]
            g₁[q, (k - 1) * N + cc] = pv[k] * nu[cc] + hv[cc][k]
        end
    end
    # Apply D (resp. S) once over all (β,c) columns; reused (scaled by Iⱼ) for every n.
    # `mul!` into a dense matrix forces eager evaluation: BLAS `gemm` when `Dop` is a
    # dense matrix, LinearMaps' column-wise apply when it is an (FMM) `LinearMap`.
    DG = Matrix{T}(undef, num_target, nβ * N)    # num_target × (nβ·N)
    SG = Matrix{T}(undef, num_target, nβ * N)
    nβ * N > 0 && (mul!(DG, Dop, g₀); mul!(SG, Sop, g₁))

    # Assemble Θ[i,n] ∈ SMatrix{N,N}. Following the internal sign convention used for
    # the scalar/W cases (Θ = naive_volume − analytic_representation, σ-term carried by
    # `green_multiplier`), the (3.12) boundary signs are flipped: +∇ₓS, +S, −D.
    Θ = Matrix{SM}(undef, num_target, num_basis)
    volbuf = Vector{SV}(undef, num_target)
    # Volume term: the j-th column of Mₐ(xᵢ) := ∫∇ₓ∇ₓG(xᵢ,y) pₐ(y) dy (an `SMatrix`)
    # is exactly `Vop·(pₐeⱼ)`, so all N directions share the single `SMatrix` Mₐ. The FMM
    # charge→Hessian operator (a `LinearMap{SMatrix}`, scalar density → `SMatrix`) forms
    # every Mₐ with a single FMM pass per monomial, avoiding the N per-direction dipole
    # passes (which re-traverse `Vop` N·num_basis times). All other operators — the dense
    # `BlockArray{SMatrix}` and the FMM dipole→gradient map (`SVector` output) — use the
    # per-direction application below.
    # `Υ[j]` evaluated at every target once, up front: `Υtargall[j][i][n] = ∇Ψₙⱼ(xᵢ)`.
    # Converted to `SV`, since the basis coefficients are real even when the kernel — and
    # hence everything accumulated alongside these below — is complex.
    Υtargall = ntuple(j -> [SV.(Υ[j](coords(target[i]))) for i in 1:num_target], N)
    opt_vol = eltype(Vop) == SM && !(Vop isa BlockArray)
    MVol = Matrix{SM}(undef, opt_vol ? num_target : 0, opt_vol ? num_basis : 0)
    if opt_vol
        mbuf = Vector{SM}(undef, num_target)
        for n in 1:num_basis
            mul!(mbuf, Vop, view(b, :, n))
            for i in 1:num_target
                MVol[i, n] = mbuf[i]
            end
        end
    end
    for n in 1:num_basis
        I = indices[n]
        # SMatrix Υₙ(xᵢ), column `j` equal to `∇Ψₙⱼ`
        Υtarg = [
            reduce(hcat, ntuple(j -> Υtargall[j][i][n], N)) for i in 1:num_target
        ]
        colvecs = ntuple(N) do j
            # σ-term: green_multiplier · Υₙⱼ(x)
            col = [green_multiplier[i] * Υtarg[i][:, j] for i in 1:num_target]
            # naive volume: +∫∇ₓ∇ₓG·(pₐeⱼ) = X PV part for direction j (column j of Mₐ)
            if opt_vol
                for i in 1:num_target
                    col[i] += MVol[i, n][:, j]
                end
            else
                ej = svector(d -> d == j ? one(T) : zero(T), N)
                dj = [b[k, n] * ej for k in 1:num_source]
                mul!(volbuf, Vop, dj)
                col .+= volbuf
            end
            # +∇ₓS[pₐνⱼ]
            φj = [γs[q, n][j] for q in 1:nbnd]
            col .+= GSop * φj
            # +S[(∂ⱼpₐ)ν + BνΥⱼ] − D[Υⱼ], via the β = I − eⱼ dedup (scaled by Iⱼ)
            if I[j] ≥ 1
                k = β2k[ntuple(d -> d == j ? I[d] - 1 : I[d], N)]
                Iⱼ = T(I[j])
                for c in 1:N
                    kc = (k - 1) * N + c
                    for i in 1:num_target
                        col[i] += SV(ntuple(d -> d == c ? Iⱼ * (SG[i, kc] - DG[i, kc]) : zero(T), N))
                    end
                end
            end
            col
        end
        for i in 1:num_target
            Θ[i, n] = reduce(hcat, ntuple(j -> colvecs[j][i], N))
        end
    end

    return _dim_correction(source, dict_near, bfun, Θ, SM)
end
