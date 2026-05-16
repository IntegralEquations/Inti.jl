"""
    struct DimParameters

Parameters associated with the density interpolation method used in
[`bdim_correction`](@ref).
"""
@kwdef struct DimParameters
    sources_oversample_factor::Float64 = 3
    sources_radius_multiplier::Float64 = 1.5
end

# ---- Internal helpers for bdim_correction ----
#
# The per-target linear solve is Mdata' * Wdata ≈ Θi_flat, where Mdata is a
# plain float matrix whose shape is determined by `Tbase` (the base kernel
# eltype). The RHS column count and pack/unpack logic depend on `Tout`:
#   Tout <: Number                     → 1 column
#   Tout <: SVector{P,<:Number}        → P columns
#   Tout <: SMatrix{N,N}               → N columns       (one block solve)
#   Tout <: SVector{K,<:SMatrix{N,N}}  → K*N columns     (K block solves batched)
#
# The `transpose` calls in the SMatrix variants reflect BlockArray's column-major
# block layout: writing an SMatrix through parent() exposes its transpose.

_bdim_num_rhs(::Type{T}) where {T <: Number} = 1
_bdim_num_rhs(::Type{SV}) where {P, SV <: SVector{P, <:Number}} = P
_bdim_num_rhs(::Type{SM}) where {N, SM <: SMatrix{N, N}} = N
_bdim_num_rhs(::Type{SV}) where {K, SM <: SMatrix, SV <: SVector{K, SM}} = K * size(SM, 1)

function _bdim_fill_Θi!(Θi_flat, Θ, j, ::Type{T}) where {T <: Number}
    @inbounds for m in axes(Θi_flat, 1)
        Θi_flat[m, 1] = Θ[j, m]
    end
end
function _bdim_fill_Θi!(Θi_flat, Θ, j, ::Type{SV}) where {P, SV <: SVector{P, <:Number}}
    @inbounds for m in axes(Θi_flat, 1)
        Θi_flat[m, :] .= Θ[j, m]
    end
end
function _bdim_fill_Θi!(Θi_flat, Θ, j, ::Type{SM}) where {N, SM <: SMatrix{N, N}}
    ns = size(Θi_flat, 1) ÷ N
    @inbounds for m in 1:ns
        Θi_flat[(m - 1) * N + 1:m * N, :] .= transpose(Θ[j, m])
    end
end
function _bdim_fill_Θi!(Θi_flat, Θ, j, ::Type{SV}) where {K, SM <: SMatrix, SV <: SVector{K, SM}}
    N = size(SM, 1)
    ns = size(Θi_flat, 1) ÷ N
    @inbounds for m in 1:ns, kk in 1:K
        Θi_flat[(m - 1) * N + 1:m * N, (kk - 1) * N + 1:kk * N] .= transpose(Θ[j, m][kk])
    end
end

function _bdim_push_weights!(Is, Js, Ss, Ds, Wdata, i, jglob, nq, ::Type{T}) where {T <: Number}
    @inbounds for k in 1:nq
        push!(Is, i); push!(Js, jglob[k])
        push!(Ss, -Wdata[nq + k, 1])
        push!(Ds,  Wdata[k, 1])
    end
end
function _bdim_push_weights!(Is, Js, Ss, Ds, Wdata, i, jglob, nq, ::Type{SV}) where {P, SV <: SVector{P, <:Number}}
    @inbounds for k in 1:nq
        push!(Is, i); push!(Js, jglob[k])
        push!(Ss, -SV(Wdata[nq + k, :]))
        push!(Ds,  SV(Wdata[k, :]))
    end
end
function _bdim_push_weights!(Is, Js, Ss, Ds, Wdata, i, jglob, nq, ::Type{SM}) where {N, SM <: SMatrix{N, N}}
    @inbounds for k in 1:nq
        push!(Is, i); push!(Js, jglob[k])
        push!(Ss, -transpose(SM(view(Wdata, N * (nq + k - 1) + 1:N * (nq + k), :))))
        push!(Ds,  transpose(SM(view(Wdata, N * (k - 1) + 1:k * N, :))))
    end
end
function _bdim_push_weights!(Is, Js, Ss, Ds, Wdata, i, jglob, nq, ::Type{SV}) where {K, SM <: SMatrix, SV <: SVector{K, SM}}
    N = size(SM, 1)
    @inbounds for k in 1:nq
        push!(Is, i); push!(Js, jglob[k])
        push!(Ss, SV(ntuple(kk -> -transpose(SM(view(Wdata, N * (nq + k - 1) + 1:N * (nq + k), (kk - 1) * N + 1:kk * N))), Val(K))))
        push!(Ds, SV(ntuple(kk ->  transpose(SM(view(Wdata, N * (k - 1) + 1:k * N,             (kk - 1) * N + 1:kk * N))), Val(K))))
    end
end

"""
    bdim_correction(op,X,Y,S,D; green_multiplier, kwargs...)

Given a `op` and a (possibly inaccurate) discretizations of its single and
double-layer operators `S` and `D` (taking a vector of values on `Y` and
returning a vector on of values on `X`), compute corrections `δS` and `δD` such
that `S + δS` and `D + δD` are more accurate approximations of the underlying
single- and double-layer integral operators.

See [faria2021general](@cite) for more details on the method.

# Arguments

## Required:

- `op` must be an [`AbstractDifferentialOperator`](@ref)
- `Y` must be a [`Quadrature`](@ref) object of a closed surface
- `X` is either inside, outside, or on `Y`
- `S` and `D` are approximations to the single- and double-layer operators for
  `op` taking densities in `Y` and returning densities in `X`.
- `green_multiplier` (keyword argument) is a vector with the same length as `X`
  storing the value of `μ(x)` for `x ∈ X` in the Green identity `S\\[γ₁u\\](x) -
  D\\[γ₀u\\](x) + μ*u(x) = 0`. See [`_green_multiplier`](@ref).

## Optional `kwargs`:

- `parameters::DimParameters`: parameters associated with the density
  interpolation method
- `kernel_variant`: `:default` for the standard single/double-layer, `:neumann` for
  the adjoint double-layer and hypersingular operators (Neumann trace), or `:gradient`
  for the gradient kernels. `S` and `D` must be consistent with the chosen variant.
- `maxdist`: distance beyond which interactions are considered sufficiently far
  so that no correction is needed. This is used to determine a threshold for
  nearly-singular corrections when `X` and `Y` are different surfaces. When `X
  === Y`, this is not needed.

"""
function bdim_correction(
        op,
        target,
        source::Quadrature,
        Sop,
        Dop;
        green_multiplier::Vector{<:Real},
        parameters = DimParameters(),
        kernel_variant::Symbol = :default,
        maxdist = Inf,
        filter_target_params = nothing,
    )
    imat_cond = imat_norm = res_norm = rhs_norm = theta_norm = -Inf
    Tout = eltype(Sop)
    Tbase = default_kernel_eltype(op)
    # determine type for dense matrices
    DenseBase = Tbase <: SMatrix ? BlockArray : Array
    N = ambient_dimension(source)
    @assert eltype(Dop) == Tout "eltype of S and D must match"
    m, n = length(target), length(source)
    # check if we are in debug mode to avoid expensive computations
    do_debug = debug_mode()
    if isnothing(filter_target_params)
        dict_near = etype_to_nearest_points(target, source; maxdist)
        num_trgs = m
        glob_loc_near_trgs = Dict(i => i for i in 1:m)
    else
        dict_near = filter_target_params.dict_near
        num_trgs = filter_target_params.num_trgs
        glob_loc_near_trgs = filter_target_params.glob_loc_near_trgs
    end
    # find first an appropriate set of source points to center the monopoles
    qmax = sum(size(mat, 1) for mat in values(source.etype2qtags)) # max number of qnodes per el
    ns = ceil(Int, parameters.sources_oversample_factor * qmax)
    # compute a bounding box for source points
    low_corner = reduce((p, q) -> min.(coords(p), coords(q)), source)
    high_corner = reduce((p, q) -> max.(coords(p), coords(q)), source)
    xc = (low_corner + high_corner) / 2
    R = parameters.sources_radius_multiplier * norm(high_corner - low_corner) / 2
    xs = if N === 2
        uniform_points_circle(ns, R, xc)
    elseif N === 3
        fibonnaci_points_sphere(ns, R, xc)
    else
        error("only 2D and 3D supported")
    end
    # compute traces of monopoles on the source mesh
    G = SingleLayerKernel(op, Tbase)
    γ₁G = AdjointDoubleLayerKernel(op, Tbase)

    if kernel_variant === :gradient
        G_target = GradientSingleLayerKernel(op, Tout)
        γ₁G_target = GradientDoubleLayerKernel(op, Tout)
    elseif kernel_variant === :neumann
        G_target = AdjointDoubleLayerKernel(op, Tout)
        γ₁G_target = HyperSingularKernel(op, Tout)
    else
        G_target = SingleLayerKernel(op, Tout)
        γ₁G_target = DoubleLayerKernel(op, Tout)
    end

    γ₀B = DenseBase{Tbase}(undef, length(source), ns)
    γ₁B = DenseBase{Tbase}(undef, length(source), ns)
    for k in 1:ns
        for j in 1:length(source)
            γ₀B[j, k] = G(source[j], xs[k])
            γ₁B[j, k] = γ₁G(source[j], xs[k])
        end
    end
    # integrate the monopoles/dipoles over Y with target on X. This is the
    # slowest step, and passing a custom S,D can accelerate this computation.
    DenseOut = Tout <: SMatrix ? BlockArray : Array
    Θ = DenseOut{Tout}(undef, m, ns)
    fill!(Θ, zero(Tout))
    # Compute Θ <-- S * γ₁B - D * γ₀B + μ * B(x) using in-place matvec
    for k in 1:ns
        for i in 1:length(target)
            μ = green_multiplier[i]
            v = G_target(target[i], xs[k])
            Θ[i, k] = μ * v
        end
    end
    if DenseOut <: Array || (Sop isa BlockArray && Dop isa BlockArray)
        mul!(Θ, Sop, γ₁B, 1, 1)
        mul!(Θ, Dop, γ₀B, -1, 1)
    else
        # for vector value problems, we only assume that Sop and Dop can be multiplied by
        # Vectors of SVectors, and so we need to perform multiplication column by column
        P, Q = size(Tbase)
        S = eltype(Tbase)
        Θ_data = parent(Θ)
        γ₀B_data = parent(γ₀B)
        γ₁B_data = parent(γ₁B)
        for k in 1:size(Θ_data, 2)
            y = reinterpret(SVector{P, S}, @view Θ_data[:, k])
            x = reinterpret(SVector{Q, S}, @view γ₁B_data[:, k])
            mul!(y, Sop, x, 1, 1)
            x = reinterpret(SVector{Q, S}, @view γ₀B_data[:, k])
            mul!(y, Dop, x, -1, 1)
        end
    end

    # finally compute the corrected weights as sparse matrices
    Is, Js, Ss, Ds = Int[], Int[], Tout[], Tout[]
    for (E, qtags) in source.etype2qtags
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        # M's block structure is fully determined by Tbase; its plain float
        # parent Mdata is what LAPACK actually factors. The RHS column count
        # `nc` is the only thing Tout contributes to the solve dimensions.
        M = DenseBase{Tbase}(undef, 2 * nq, ns)
        Mdata = parent(M)::Matrix
        S_scalar = eltype(Mdata)
        nc = _bdim_num_rhs(Tout)
        Θi_flat = Matrix{S_scalar}(undef, size(Mdata, 2), nc)
        Wdata   = Matrix{S_scalar}(undef, size(Mdata, 1), nc)
        # for each element, we will solve Mᵀ W = Θiᵀ, where W is a matrix of
        # size (flat_m × nc), and Θiᵀ has size (flat_ns × nc)
        for n in 1:ne
            # if there is nothing near, skip immediately to next element
            isempty(near_list[n]) && continue
            # copy the monopoles/dipoles for the current element
            jglob = @view qtags[:, n]
            M[1:nq, :] .= γ₀B[jglob, :]
            M[(nq + 1):2nq, :] .= γ₁B[jglob, :]
            # TODO: get rid of all this transposing mumble jumble by assembling
            # the matrix in the correct orientation in the first place
            F = qr!(transpose(Mdata))
            if do_debug
                imat_cond = max(cond(Mdata), imat_cond)
                imat_norm = max(norm(Mdata), imat_norm)
            end
            for i in near_list[n]
                j = glob_loc_near_trgs[i]
                _bdim_fill_Θi!(Θi_flat, Θ, j, Tout)
                ldiv!(Wdata, F, Θi_flat)
                _bdim_push_weights!(Is, Js, Ss, Ds, Wdata, i, jglob, nq, Tout)
            end
        end
    end
    @debug """Condition properties of bdim correction:
    |-- max interp. matrix cond.: $imat_cond
    |-- max interp. matrix norm : $imat_norm
    |-- max residual error:       $res_norm
    |-- max correction norm:      $theta_norm
    |-- max norm of source term:  $rhs_norm
    """
    δS = sparse(Is, Js, Ss, num_trgs, n)
    δD = sparse(Is, Js, Ds, num_trgs, n)
    return δS, δD
end
