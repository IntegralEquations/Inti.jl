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
    bdim_correction(pde,X,Y,S,D[;location,parameters,derivative,tol])

Given a `pde` and a (possibly innacurate) discretizations of its single and
double-layer operators `S` and `D` (taking a vector of values on `Y` and
returning a vector on of values on `X`), compute corrections `δS` and `δD` such
that `S + δS` and `D + δD` are more accurate approximations of the underlying
single- and double-layer integral operators.

The following optional keyword arguments are available:
- `parameters::DimParameters`: parameters associated with the density interpolation
  method
- `derivative`: if true, compute the correction to the adjoint double-layer and
  hypersingular operators instead. In this case, `S` and `D` should be replaced
  by a (possibly innacurate) discretization of adjoint double-layer and
  hypersingular operators, respectively.
- `tol`: distance beyond which interactions are considered sufficiently far so
  that no correction is needed. This is used to determine a threshold for
  nearly-singular corrections when `X` and `Y` are different surfaces.
"""
function bdim_correction(
    pde,
    target,
    source::Quadrature,
    Sop,
    Dop;
    target_location = :onsurface,
    parameters = DimParameters(),
    derivative = false,
    tol = Inf,
)
    max_cond = -Inf
    T = eltype(Sop)
    N = ambient_dimension(source)
    @assert eltype(Dop) == T "eltype of S and D must match"
    m, n = length(target), length(source)
    msg = "unrecognized value for kw `location`: received $target_location.
    Valid options are `:onsurface`, `:inside`, `:outside`, or the numeric value of the multiplier"
    μ::Float64 = if target_location === :onsurface
        -0.5
    elseif target_location === :inside
        -1.0
    elseif target_location === :outside
        0.0
    elseif target_location isa Number
        target_location
    else
        error(msg)
    end
    dict_near = etype_to_nearest_points(target, source; tol)
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
    γ₀B = Matrix{T}(undef, length(source), ns)
    γ₁B = Matrix{T}(undef, length(source), ns)
    for k in 1:ns
        for j in 1:length(source)
            γ₀B[j, k] = G(source[j], xs[k])
            γ₁B[j, k] = γ₁G(source[j], xs[k])
        end
    end
    # integrate the monopoles/dipoles over Y with target on X. This is the
    # slowest step, and passing a custom S,D can accelerate this computation.
    Θ = zeros(T, m, ns)
    # Compute Θ <-- S * γ₁B - D * γ₀B + σ * B(x) usig in-place matvec
    for k in 1:ns
        @views mul!(Θ[:, k], Sop, γ₁B[:, k])
        @views mul!(Θ[:, k], Dop, γ₀B[:, k], -1, 1)
        if derivative
            for i in 1:length(target)
                Θ[i, k] += μ * γ₁G(target[i], xs[k])
            end
        else
            for i in 1:length(target)
                Θ[i, k] += μ * G(target[i], xs[k])
            end
        end
    end
    @debug "Norm of correction: " norm(Θ)
    # finally compute the corrected weights as sparse matrices
    @debug "precomputation finished"
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
        σ   = if T <: Number
            1
        else
            @assert allequal(size(T))
            size(T, 1)
        end
        M_  = Matrix{eltype(T)}(undef, 2 * nq * σ, ns * σ)
        W_  = Matrix{eltype(T)}(undef, 2 * nq * σ, σ)
        W   = T <: Number ? W_ : Matrix{T}(undef, 2 * nq, 1)
        Θi_ = Matrix{eltype(T)}(undef, σ, ns * σ)
        # for each element, we will solve Mᵀ W = Θiᵀ, where W is a vector of
        # size 2nq, and Θi is a row vector of length(ns)
        for n in 1:ne
            # if there is nothing near, skip immediately to next element
            isempty(near_list[n]) && continue
            # copy the monopole/dipole weights for the current element
            jglob = @view qtags[:, n]
            M0 = @view γ₀B[jglob, :]
            M1 = @view γ₁B[jglob, :]
            _copyto!(view(M_, 1:(nq*σ), :), M0)
            _copyto!(view(M_, (nq*σ+1):2*nq*σ, :), M1)
            F_ = qr!(transpose(M_))
            @debug begin
                max_cond = max(cond(M_), max_cond)
            end
            for i in near_list[n]
                Θi  = @view Θ[i:i, :]
                Θi_ = _copyto!(Θi_, Θi)
                W_  = ldiv!(W_, F_, transpose(Θi_))
                W   = T <: Number ? W_ : _copyto!(W, W_)
                for k in 1:nq
                    push!(Is, i)
                    push!(Js, jglob[k])
                    push!(Ss, -W[nq+k]) # single layer corresponds to α=0,β=-1
                    push!(Ds, W[k])       # double layer corresponds to α=1,β=0
                end
            end
        end
    end
    @debug "Maximum condition number of linear system: " max_cond
    δS = sparse(Is, Js, Ss, m, n)
    δD = sparse(Is, Js, Ds, m, n)
    return δS, δD
end

"""
    etype_to_nearest_points(X,Y::Quadrature; tol)

For each element `el` in `Y.mesh`, return a list with the indices of all points
in `X` for which `el` is the nearest element. Ignore indices for which the
distance exceeds `tol`.
"""
function etype_to_nearest_points(X, Y::Quadrature; tol = Inf)
    if X === Y
        # when both surfaces are the same, the "near points" of an element are
        # simply its own quadrature points
        dict = Dict{DataType,Vector{Vector{Int}}}()
        for (E, idx_dofs) in Y.etype2qtags
            dict[E] = map(i -> collect(i), eachcol(idx_dofs))
        end
    else
        dict = _etype_to_nearest_points(collect(qcoords(X)), Y, tol)
    end
    return dict
end

function _etype_to_nearest_points(X, Y::Quadrature, tol = Inf)
    y = [coords(q) for q in Y]
    kdtree = KDTree(y)
    dict = Dict(j => Int[] for j in 1:length(y))
    for i in eachindex(X)
        qtag, d = nn(kdtree, X[i])
        d > tol || push!(dict[qtag], i)
    end
    # dict[j] now contains indices in X for which the j quadrature node in Y is
    # the closest. Next we reverse the map
    etype2nearlist = Dict{DataType,Vector{Vector{Int}}}()
    for (E, tags) in Y.etype2qtags
        nq, ne = size(tags)
        etype2nearlist[E] = nearlist = [Int[] for _ in 1:ne]
        for j in 1:ne # loop over each element of type E
            for q in 1:nq # loop over qnodes in the element
                qtag = tags[q, j]
                append!(nearlist[j], dict[qtag])
            end
        end
    end
    return etype2nearlist
end
