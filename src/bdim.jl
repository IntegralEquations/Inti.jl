"""
    struct DimParameters

Parameters associated with the density interpolation method used in
[`bdim_correction`](@ref).
"""
@kwdef struct DimParameters
    sources_oversample_factor::Float64 = 3
    sources_radius_multiplier::Float64 = 1.5
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
- `derivative`: if true, compute the correction to the adjoint double-layer and
  hypersingular operators instead. In this case, `S` and `D` should be replaced
  by a (possibly innacurate) discretization of adjoint double-layer and
  hypersingular operators, respectively.
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
    derivative::Bool = false,
    maxdist = Inf,
    filter_target_params = nothing,
)
    imat_cond = imat_norm = res_norm = rhs_norm = theta_norm = -Inf
    T = eltype(Sop)
    # determine type for dense matrices
    Dense = T <: SMatrix ? BlockArray : Array
    N = ambient_dimension(source)
    @assert eltype(Dop) == T "eltype of S and D must match"
    m, n = length(target), length(source)
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
    ns   = ceil(Int, parameters.sources_oversample_factor * qmax)
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
    G = SingleLayerKernel(op, T)
    γ₁G = AdjointDoubleLayerKernel(op, T)
    γ₀B = Dense{T}(undef, length(source), ns)
    γ₁B = Dense{T}(undef, length(source), ns)
    for k in 1:ns
        for j in 1:length(source)
            γ₀B[j, k] = G(source[j], xs[k])
            γ₁B[j, k] = γ₁G(source[j], xs[k])
        end
    end
    # integrate the monopoles/dipoles over Y with target on X. This is the
    # slowest step, and passing a custom S,D can accelerate this computation.
    Θ = Dense{T}(undef, m, ns)
    fill!(Θ, zero(T))
    # Compute Θ <-- S * γ₁B - D * γ₀B + μ * B(x) usig in-place matvec
    for k in 1:ns
        for i in 1:length(target)
            μ = green_multiplier[i]
            v = derivative ? γ₁G(target[i], xs[k]) : G(target[i], xs[k])
            Θ[i, k] = μ * v
        end
    end
    @views mul!(Θ, Sop, γ₁B, 1, 1)
    @views mul!(Θ, Dop, γ₀B, -1, 1)

    # finally compute the corrected weights as sparse matrices
    Is, Js, Ss, Ds = Int[], Int[], T[], T[]
    for (E, qtags) in source.etype2qtags
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        # preallocate a local matrix to store interpolant values resulting
        # weights. To benefit from Lapack, we must convert everything to
        # matrices of scalars, so when `T` is an `SMatrix` we are careful to
        # convert between the `Matrix{<:SMatrix}` and `Matrix{<:Number}` formats
        # by viewing the elements of type `T` as `σ × σ` matrices of
        # `eltype(T)`.
        M = Dense{T}(undef, 2 * nq, ns)
        W = Dense{T}(undef, 2 * nq, 1)
        Θi = Dense{T}(undef, 1, ns)
        Mdata, Wdata, Θidata = parent(M)::Matrix, parent(W)::Matrix, parent(Θi)::Matrix
        # for each element, we will solve Mᵀ W = Θiᵀ, where W is a vector of
        # size 2nq, and Θi is a row vector of length(ns)
        for n in 1:ne
            # if there is nothing near, skip immediately to next element
            isempty(near_list[n]) && continue
            # copy the monopoles/dipoles for the current element
            jglob = @view qtags[:, n]
            M[1:nq, :] .= γ₀B[jglob, :]
            M[(nq+1):2nq, :] .= γ₁B[jglob, :]
            # TODO: get ride of all this transposing mumble jumble by assembling
            # the matrix in the correct orientation in the first place
            F = qr!(transpose(Mdata))
            @debug (imat_cond = max(cond(Mdata), imat_cond)) maxlog = 0
            @debug (imat_norm = max(norm(Mdata), imat_norm)) maxlog = 0
            for i in near_list[n]
                j = glob_loc_near_trgs[i]
                Θi .= Θ[j:j, :]
                @debug (rhs_norm = max(rhs_norm, norm(Θidata))) maxlog = 0
                ldiv!(Wdata, F, transpose(Θidata))
                @debug (
                    res_norm = max(norm(Matrix(F) * Wdata - transpose(Θidata)), res_norm)
                ) maxlog = 0
                @debug (theta_norm = max(theta_norm, norm(Wdata))) maxlog = 0
                for k in 1:nq
                    push!(Is, i)
                    push!(Js, jglob[k])
                    # Since we actually computed the tranpose of the weights, we
                    # need to transpose it again. This matters for e.g. elasticity
                    push!(Ss, -transpose(W[nq+k])) # single layer corresponds to α=0,β=-1
                    push!(Ds, transpose(W[k]))     # double layer corresponds to α=1,β=0
                end
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
