#=
This file contains the implementation of the local boundary density interpolation method.
=#

"""
    struct LocalDimParameters

Parameters for the local boundary density interpolation method.
"""
@kwdef struct LocalDimParameters
    sources_oversample_factor::Float64 = 3
    sources_radius_multiplier::Float64 = 5
end

function local_bdim_correction(
    pde,
    target,
    source::Quadrature;
    green_multiplier::Vector{<:Real},
    parameters = LocalDimParameters(),
    derivative::Bool = false,
    maxdist,
    kneighbor = 1,
)
    imat_cond = imat_norm = res_norm = rhs_norm = theta_norm = -Inf
    T = default_kernel_eltype(pde) # Float64
    # determine type for dense matrices
    Dense = T <: SMatrix ? BlockArray : Array
    N = ambient_dimension(source)
    m, n = length(target), length(source)
    msh = source.mesh
    qnodes = source.qnodes
    neighbors = topological_neighbors(msh, kneighbor)
    X = [coords(q) for q in qnodes]
    dict_near = local_bdim_element_to_target(X, msh; maxdist)
    # dict_near = etype_to_nearest_points(target, source; maxdist)
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
    γ₀B  = Dense{T}(undef, length(source), ns)
    γ₁B  = Dense{T}(undef, length(source), ns)
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
        M = Dense{T}(undef, 2 * nq, ns)
        W = Dense{T}(undef, 2 * nq, 1)
        Θi = Dense{T}(undef, 1, ns)
        Mdata, Wdata, Θidata = parent(M)::Matrix, parent(W)::Matrix, parent(Θi)::Matrix
        K = derivative ? γ₁ₓG : G
        # for each element, we will solve Mᵀ W = Θiᵀ, where W is a vector of
        # size 2nq, and Θi is a row vector of length(ns)
        for n in 1:ne
            # if there is nothing near, skip immediately to next element
            isempty(near_list[n]) && continue
            el = els[n]
            # copy the monopoles/dipoles for the current element
            jglob = @view qtags[:, n]
            M[1:nq, :] .= γ₀B[jglob, :]
            M[nq+1:2nq, :] .= γ₁B[jglob, :]
            F = qr!(transpose(Mdata))
            @debug (imat_cond = max(cond(Mdata), imat_cond)) maxlog = 0
            @debug (imat_norm = max(norm(Mdata), imat_norm)) maxlog = 0
            # quadrature for auxiliary surface. In global dim, this is the same
            # as the source quadrature, and independent of element. In local
            # dim, this is constructed for each element using its neighbors.
            nei = neighbors[(E, n)]
            qtags_nei = Int[]
            for (E′, m) in nei
                append!(qtags_nei, source.etype2qtags[E′][:, m])
            end
            qnodes_nei = source.qnodes[qtags_nei]
            jac = jacobian(el, 0.5)
            ν = _normal(jac)
            h = sum(qnodes[i].weight for i in jglob)
            for i in near_list[n]
                # integrate the monopoles/dipoles over the auxiliary surface
                # with target x: Θₖ <-- S[γ₁Bₖ](x) - D[γ₀Bₖ](x) + μ * Bₖ(x)
                x = target[i]
                aux_els, orientation = local_bdim_auxiliary_els(nei, msh, coords(x))
                qnodes_aux =
                    local_bdim_auxiliary_quadrature(aux_els, ceil(Int, 10 * abs(log(h))))
                μ = orientation == :exterior ? green_multiplier[i] : -green_multiplier[i]
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
                for q in qnodes_nei
                    SK = G(x, q)
                    DK = γ₁G(x, q)
                    for k in 1:ns
                        Θi[k] += (SK * γ₁ₓG(q, xs[k]) - DK * G(q, xs[k])) * weight(q)
                    end
                end
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
    δS = sparse(Is, Js, Ss, m, n)
    δD = sparse(Is, Js, Ds, m, n)
    return δS, δD
end

function local_bdim_element_to_target(
    X::AbstractVector{<:SVector},
    Y::AbstractMesh;
    maxdist,
)
    # inverse of the nearest_element_in_connected_components map
    t2e = nearest_element_in_connected_components(X, Y; maxdist = maxdist)
    dict = Dict(E => [Int[] for _ in 1:length(elements(Y, E))] for E in element_types(Y))
    for (i, els) in enumerate(t2e)
        for (E, j) in els
            push!(dict[E][j], i)
        end
    end
    return dict
end
