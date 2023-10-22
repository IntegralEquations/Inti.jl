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
    vdim_correction(pde,X,Y,Γ,S,D,V[;order])
"""
function vdim_correction(pde, X, Y, Γ, S, D, V; order)
    @assert X === Y
    N = ambient_dimension(pde)
    σ = -1
    @assert ambient_dimension(Y) == N "vdim only works for volume potentials"
    # basis = _basis_vdim(pde, order)
    p, P, γ₁P = polynomial_solutions_vdim(pde, order)
    B, R = _vdim_auxiliary_quantities(p, P, γ₁P, X, Y, Γ, σ, S, D, V)
    # compute sparse correction
    Is = Int[]
    Js = Int[]
    Vs = Float64[]
    nbasis = length(p)
    max_cond = -Inf
    for (E, qtags) in Y.etype2qtags
        nq, nel = size(qtags)
        @debug "nq,nbasis = $nq,$nbasis"
        for n in 1:nel
            # indices of nodes in element `n`
            j_idxs = qtags[:, n]
            i_idxs = j_idxs # FIXME: assumes that X === Y
            L = B[j_idxs, :] # vandermond matrix
            max_cond = max(max_cond, cond(L))
            for i in i_idxs
                wei = R[i:i, :] / L
                append!(Js, j_idxs)
                append!(Is, fill(i, length(j_idxs)))
                append!(Vs, wei)
            end
        end
    end
    @debug "maximum condition encountered: $max_cond"
    δV = sparse(Is, Js, Vs)
    return δV
end

function _vdim_auxiliary_quantities(
    p,
    P,
    γ₁P,
    X,
    Y::Quadrature,
    Γ::Quadrature,
    σ,
    Sop,
    Dop,
    Vop,
)
    num_basis = length(p)
    num_targets = length(X)
    b = [f(q) for q in Y, f in p]
    γ₀B = [f(q) for q in Γ, f in P]
    γ₁B = [f(q) for q in Γ, f in γ₁P]
    Θ = zeros(eltype(Vop), num_targets, num_basis)
    # Compute Θ <-- S * γ₁B - D * γ₀B - V * b + σ * B(x) usig in-place matvec
    for n in 1:num_basis
        @views mul!(Θ[:, n], Sop, γ₁B[:, n])
        @views mul!(Θ[:, n], Dop, γ₀B[:, n], -1, 1)
        @views mul!(Θ[:, n], Vop, b[:, n], -1, 1)
        for i in 1:num_targets
            Θ[i, n] += σ * P[n](X[i])
        end
    end
    return b, Θ
end

"""
    polynomial_solutions_vdim(pde, order)

For every monomial term `pₙ` of degree `order`, compute a polynomial `Pₙ` such
that `ℒ[Pₙ] = pₙ`, where `ℒ` is the differential operator associated with `pde`.
This function returns `{pₙ,Pₙ,γ₁Pₙ}`, where `γ₁Pₙ` is the generalized Neumann
trace of `Pₙ`.
"""
function polynomial_solutions_vdim(pde::AbstractPDE, order::Integer)
    N = ambient_dimension(pde)
    # create empty arrays to store the monomials, solutions, and traces. For the
    # neumann trace, we try to infer the concrete return type instead of simply
    # having a vector of `Function`.
    monomials = Vector{PolynomialSolutions.Polynomial{N,Float64}}()
    solutions = Vector{PolynomialSolutions.Polynomial{N,Float64}}()
    T = return_type(neumann_trace, typeof(pde), eltype(solutions))
    neumann_traces = Vector{T}()
    # iterate over N-tuples going from 0 to order
    for I in Iterators.product(ntuple(i -> 0:order, N)...)
        sum(I) > order && continue
        # define the monomial basis functions, and the corresponding solutions.
        # TODO: adapt this to vectorial case
        p   = PolynomialSolutions.Polynomial(I => 1.0)
        P   = polynomial_solution(pde, p)
        γ₁P = neumann_trace(pde, P)
        push!(monomials, p)
        push!(solutions, P)
        push!(neumann_traces, γ₁P)
    end
    return monomials, solutions, neumann_traces
end

# Laplace particular solutions
function polynomial_solution(::Laplace, p::PolynomialSolutions.Polynomial)
    P = PolynomialSolutions.solve_laplace(p)
    return PolynomialSolutions.convert_coefs(P, Float64)
end

function neumann_trace(::Laplace, P::PolynomialSolutions.Polynomial{N,T}) where {N,T}
    ∇P = PolynomialSolutions.gradient(P)
    return (q) -> dot(normal(q), ∇P(q))
end

function (∇P::NTuple{N,<:PolynomialSolutions.Polynomial})(x) where {N}
    return ntuple(n -> ∇P[n](x), N)
end

function _basis_vdim(::Laplace{3}, order)
    Q = Polynomial((1, 0) => 1.0)
    P = solve_laplace(Q)
    # define the monomial basis functions, and the corresponding solutions
    # P0
    r_000 = (dof) -> 1.0
    p_000 = (dof) -> begin
        x = coords(dof)
        1 / 6 * (x[1]^2 + x[2]^2 + x[3]^2)
    end
    dp_000 = (dof) -> begin
        x = coords(dof)
        n = normal(dof)
        1 / 3 * dot(x, n)
    end
    R, P, dP = (r_000,), (p_000,), (dp_000,)
    order == 0 && return R, P, dP
    # P1
    r_100 = (dof) -> coords(dof)[1]
    p_100 = (dof) -> begin
        x = coords(dof)
        1 / 6 * x[1]^3
    end
    dp_100 = (dof) -> begin
        x = coords(dof)
        n = normal(dof)
        1 / 2 * x[1]^2 * n[1]
    end
    r_010 = (dof) -> coords(dof)[2]
    p_010 = (dof) -> begin
        x = coords(dof)
        1 / 6 * x[2]^3
    end
    dp_010 = (dof) -> begin
        x = coords(dof)
        n = normal(dof)
        1 / 2 * x[2]^2 * n[2]
    end
    r_001 = (dof) -> coords(dof)[3]
    p_001 = (dof) -> begin
        x = coords(dof)
        1 / 6 * x[3]^3
    end
    dp_001 = (dof) -> begin
        x = coords(dof)
        n = normal(dof)
        1 / 2 * x[3]^2 * n[3]
    end
    R, P, dP = (R..., r_100, r_010, r_001),
    (P..., p_100, p_010, p_001),
    (dP..., dp_100, dp_010, dp_001)
    order == 1 && return R, P, dP
    return notimplemented()
    # P2
end

function (P::PolynomialSolutions.Polynomial)(q::QuadratureNode)
    x = q.coords
    return P(x)
end
