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
    S,
    D;
    location = :onsurface,
    parameters = DimParameters(),
    derivative = false,
    tol = Inf,
)
    max_cond = -Inf # maximum condition numbered encoutered in coeffs matrix
    T        = eltype(S)
    @assert eltype(D) == T "eltype of S and D must match"
    m, n = length(target), length(source)
    msg = "unrecognized value for kw `location`: received $location.
    Valid options are `:onsurface`, `:inside`, `:outside`, or the numeric value of the multiplier"
    σ = if location === :onsurface
        -0.5
    elseif location === :inside
        -1
    elseif location === :outside
        0
    elseif location isa Number
        location
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
    if ambient_dimension(source) == 2
        xs = uniform_points_circle(ns, R, xc)
    elseif ambient_dimension(source) == 3
        # xs = lebedev_points_sphere(ns, r, xc)
        xs = fibonnaci_points_sphere(ns, R, xc)
    else
        notimplemented()
    end
    # compute traces of monopoles on the source mesh
    G = SingleLayerKernel(pde, T)
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
    # slowest step, and passing a custom S,D can accelerate this computation
    Θ = S * γ₁B - D * γ₀B
    for k in 1:ns
        for i in 1:length(target)
            if derivative
                Θ[i, k] += σ * γ₁G(target[i], xs[k])
            else
                Θ[i, k] += σ * G(target[i], xs[k])
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
        # preallocate a local matrix to store interpolant values
        M = Matrix{T}(undef, 2nq, ns)
        for n in 1:ne # loop over elements of type E
            # if there is nothing near, skip immediately to next element
            isempty(near_list[n]) && continue
            # the weights for points in near_list[n] must be corrected when
            # integrating over the current element
            jglob = @view qtags[:, n]
            M0 = @view γ₀B[jglob, :]
            M1 = @view γ₁B[jglob, :]
            copy!(view(M, 1:nq, :), M0)
            copy!(view(M, (nq+1):(2nq), :), M1)
            F = qr!(M)
            max_cond = max(cond(F.R), max_cond)
            for i in near_list[n]
                Θi = @view Θ[i:i, :]
                W = (Θi / F.R) * adjoint(F.Q)
                for k in 1:nq
                    push!(Is, i)
                    push!(Js, jglob[k])
                    push!(Ss, -W[nq+k]) # single layer corresponds to α=0,β=-1
                    push!(Ds, W[k]) # double layer corresponds to α=1,β=0
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
