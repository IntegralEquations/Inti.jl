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
    bdim_correction(pde,X,Y,S,D; green_multiplier, kwargs...)

Given a `pde` and a (possibly innacurate) discretizations of its single and
double-layer operators `S` and `D` (taking a vector of values on `Y` and
returning a vector on of values on `X`), compute corrections `δS` and `δD` such
that `S + δS` and `D + δD` are more accurate approximations of the underlying
single- and double-layer integral operators.

See [faria2021general](@cite) for more details on the method.

# Arguments

## Required:

- `pde` must be an [`AbstractPDE`](@ref)
- `Y` must be a [`Quadrature`](@ref) object of a closed surface
- `X` is either inside, outside, or on `Y`
- `S` and `D` are approximations to the single- and double-layer operators for
  `pde` taking densities in `Y` and returning densities in `X`.
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
    pde,
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
    imat_cond = imat_norm = res_norm = rhs_norm = -Inf
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
    G   = SingleLayerKernel(pde, T)
    γ₁G = AdjointDoubleLayerKernel(pde, T)
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
            M[nq+1:2nq, :] .= γ₁B[jglob, :]
            F = qr!(transpose(Mdata))
            @debug (imat_cond = max(cond(Mdata), imat_cond)) maxlog = 0
            @debug (imat_norm = max(norm(Mdata), imat_norm)) maxlog = 0
            for i in near_list[n]
                j = glob_loc_near_trgs[i]
                Θi .= Θ[j:j, :]
                @debug (rhs_norm = max(rhs_norm, norm(Θidata))) maxlog = 0
                ldiv!(Wdata, F, transpose(Θidata))
                @debug (res_norm = max(norm(Matrix(F) * Wdata - transpose(Θi)), res_norm)) maxlog =
                    0
                for k in 1:nq
                    push!(Is, i)
                    push!(Js, jglob[k])
                    push!(Ss, -W[nq+k]) # single layer corresponds to α=0,β=-1
                    push!(Ds, W[k])       # double layer corresponds to α=1,β=0
                end
            end
        end
    end
    @debug """Condition properties of bdim correction:
    |-- max interp. matrix cond.: $imat_cond
    |-- max interp. matrix norm : $imat_norm
    |-- max residual error:       $res_norm
    |-- max norm of source term:  $rhs_norm
    """
    δS = sparse(Is, Js, Ss, num_trgs, n)
    δD = sparse(Is, Js, Ds, num_trgs, n)
    return δS, δD
end

function local_bdim_correction(
    pde,
    target,
    source::Quadrature;
    green_multiplier::Vector{<:Real},
    parameters = DimParameters(),
    derivative::Bool = false,
    maxdist = Inf,
)
    imat_cond = imat_norm = res_norm = rhs_norm = -Inf
    T = default_kernel_eltype(pde) # Float64
    N = ambient_dimension(source)
    m, n = length(target), length(source)
    msh = source.mesh
    qnodes = source.qnodes
    neighbors = topological_neighbors(msh)
    dict_near = etype_to_nearest_points(target, source; maxdist)
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
    # figure out if we are dealing with a scalar or vector PDE
    σ = if T <: Number
        1
    else
        @assert allequal(size(T))
        size(T, 1)
    end
    # compute traces of monopoles on the source mesh
    G    = SingleLayerKernel(pde, T)
    γ₁G  = DoubleLayerKernel(pde, T)
    γ₁ₓG = AdjointDoubleLayerKernel(pde, T)
    γ₀B  = Matrix{T}(undef, length(source), ns)
    γ₁B  = Matrix{T}(undef, length(source), ns)
    for k in 1:ns
        for j in 1:length(source)
            γ₀B[j, k] = G(source[j], xs[k])
            γ₁B[j, k] = γ₁ₓG(source[j], xs[k])
        end
    end
    Is, Js, Ss, Ds = Int[], Int[], T[], T[]
    for (E, qtags) in source.etype2qtags
        els = elements(msh, E)
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        # preallocate a local matrix to store interpolant values resulting
        # weights. To benefit from Lapack, we must convert everything to
        # matrices of scalars, so when `T` is an `SMatrix` we are careful to
        # convert between the `Matrix{<:SMatrix}` and `Matrix{<:Number}` formats
        # by viewing the elements of type `T` as `σ × σ` matrices of
        # `eltype(T)`.
        M_  = Matrix{eltype(T)}(undef, 2 * nq * σ, ns * σ)
        W_  = Matrix{eltype(T)}(undef, 2 * nq * σ, σ)
        W   = T <: Number ? W_ : Matrix{T}(undef, 2 * nq, 1)
        Θi_ = Matrix{eltype(T)}(undef, σ, ns * σ)
        Θi  = T <: Number ? Θi_ : Matrix{T}(undef, 1, ns)
        K   = derivative ? γ₁ₓG : G
        # for each element, we will solve Mᵀ W = Θiᵀ, where W is a vector of
        # size 2nq, and Θi is a row vector of length(ns)
        for n in 1:ne
            # if there is nothing near, skip immediately to next element
            isempty(near_list[n]) && continue
            el = els[n]
            # copy the monopoles/dipoles for the current element
            jglob = @view qtags[:, n]
            M0 = @view γ₀B[jglob, :]
            M1 = @view γ₁B[jglob, :]
            _copyto!(view(M_, 1:(nq*σ), :), M0)
            _copyto!(view(M_, (nq*σ+1):2*nq*σ, :), M1)
            F_ = qr!(transpose(M_))
            @debug (imat_cond = max(cond(M_), imat_cond)) maxlog = 0
            @debug (imat_norm = max(norm(M_), imat_norm)) maxlog = 0
            # quadrature for auxiliary surface. In global dim, this is the same
            # as the source quadrature, and independent of element. In local
            # dim, this is constructed for each element using its neighbors.
            function translate(q::QuadratureNode, x, s)
                return QuadratureNode(coords(q) + x, weight(q), s * normal(q))
            end
            nei = neighbors[(E, n)]
            qtags_nei = Int[]
            for (E, m) in nei
                append!(qtags_nei, source.etype2qtags[E][:, m])
            end
            qnodes_nei = source.qnodes[qtags_nei]
            jac        = jacobian(el, 0.5)
            ν          = -_normal(jac)
            h          = sum(qnodes[i].weight for i in jglob)
            qnodes_op  = map(q -> translate(q, h * ν, -1), qnodes_nei)
            bindx      = boundary1d(nei, msh)
            l, r       = nodes(msh)[-bindx[1]], nodes(msh)[bindx[2]]
            Q, W       = gauss(3nq, 0, h)
            qnodes_l   = [QuadratureNode(l.+q.*ν, w, SVector(-ν[2], ν[1])) for (q, w) in zip(Q, W)]
            qnodes_r   = [QuadratureNode(r.+q.*ν, w, SVector(ν[2], -ν[1])) for (q, w) in zip(Q, W)]
            qnodes_aux = append!(qnodes_nei, qnodes_op, qnodes_l, qnodes_r)
            # qnodes_aux = source.qnodes # this is the global dim
            for i in near_list[n]
                # integrate the monopoles/dipoles over the auxiliary surface
                # with target x: Θₖ <-- S[γ₁Bₖ](x) - D[γ₀Bₖ](x) + μ * Bₖ(x)
                x = target[i]
                μ = green_multiplier[i] # - 1/2
                for k in 1:ns
                    Θi[k] = μ * K(x, xs[k])
                end
                for q in qnodes_aux
                    SK = G(x, q)
                    DK = γ₁G(x, q)
                    for k in 1:ns
                        Θi[k] += (SK * γ₁ₓG(q, xs[k]) - DK * G(q, xs[k])) * weight(q)
                    end
                end
                Θi_ = _copyto!(Θi_, Θi)
                @debug (rhs_norm = max(rhs_norm, norm(Θi))) maxlog = 0
                W_ = ldiv!(W_, F_, transpose(Θi_))
                @debug (res_norm = max(norm(Matrix(F_) * W_ - transpose(Θi_)), res_norm)) maxlog =
                    0
                W = T <: Number ? W_ : _copyto!(W, W_)
                for k in 1:nq
                    push!(Is, i)
                    push!(Js, jglob[k])
                    push!(Ss, -W[nq+k]) # single layer corresponds to α=0,β=-1
                    push!(Ds, W[k])       # double layer corresponds to α=1,β=0
                end
            end
        end
    end
    @debug """Condition properties of bdim correction:
    |-- max interp. matrix cond.: $imat_cond
    |-- max interp. matrix norm : $imat_norm
    |-- max residual error:       $res_norm
    |-- max norm of source term:  $rhs_norm
    """
    δS = sparse(Is, Js, Ss, m, n)
    δD = sparse(Is, Js, Ds, m, n)
    return δS, δD
end
