"""
    vdim_correction(pde,X,Y,Γ,S,D,V[;order, multiplier])
"""
function vdim_correction(
    pde,
    target,
    source::Quadrature,
    boundary::Quadrature,
    Sop,
    Dop,
    Vop;
    interpolation_order = nothing,
    derivative = false,
    green_multiplier = nothing,
    maxdist = Inf,
)
    max_cond = -Inf
    T = eltype(Vop)
    @assert eltype(Dop) == eltype(Sop) == T "eltype of Sop, Dop, and Vop must match"
    # figure out if we are dealing with a scalar or vector PDE
    σ = if T <: Number
        1
    else
        @assert allequal(size(T))
        size(T, 1)
    end
    N = ambient_dimension(pde)
    @assert ambient_dimension(source) == N "vdim only works for volume potentials"
    m, n = length(target), length(source)
    # maximum number of quadrature nodes per element
    qmax = sum(size(mat, 1) for mat in values(source.etype2qtags))
    # TODO: relate order to qmax?
    isnothing(interpolation_order) &&
        (interpolation_order = maximum(order, values(source.etype2qrule)))
    p, P, γ₁P = polynomial_solutions_vdim(pde, interpolation_order)
    μ = if isnothing(green_multiplier)
        μ_ = _green_multiplier(target[1], boundary)
        # snap to the nearest "generic" value of μ
        argmin(x -> norm(μ_ - x), (-1, -0.5, 0, 0.5, 1))
    elseif isscalar(green_multiplier)
        green_multiplier
    else
        error("green_multiplier must be a scalar")
    end
    dict_near = etype_to_nearest_points(target, source; maxdist)
    B, R, shifts, scales =
        _vdim_auxiliary_quantities(p, P, γ₁P, target, source, boundary, μ, Sop, Dop, Vop)
    # compute sparse correction
    Is = Int[]
    Js = Int[]
    Vs = eltype(Vop)[]
    nbasis = length(p)
    for (E, qtags) in source.etype2qtags
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        for n in 1:ne
            # indices of nodes in element `n`
            isempty(near_list[n]) && continue
            jglob = @view qtags[:, n]
            L = B[jglob, :] # vandermond matrix for current element
            @debug begin
                max_cond = max(max_cond, cond(L))
            end
            for i in near_list[n]
                wei = R[i:i, :] / L
                for k in 1:nq
                    push!(Is, i)
                    push!(Js, jglob[k])
                    push!(Vs, wei[k])
                end
            end
        end
    end
    @debug "maximum condition encountered: $max_cond"
    δV = sparse(Is, Js, Vs, m, n)
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
    num_els = Int(length(Y) / num_basis)
    N = ambient_dimension(Y)
    γ₀B = [f(q) for q in Γ, f in P]
    γ₁B = [f(q) for q in Γ, f in γ₁P]
    Θ = zeros(eltype(Vop), num_targets, num_basis)
    b = zeros(eltype(Vop), num_els * num_basis, num_basis)
    c = Array{eltype(Vop)}(undef, N, num_els)
    r = Array{eltype(Vop)}(undef, num_els)
    for E in element_types(Y.mesh)
        E <: LagrangeElement{ReferenceSimplex{N}} || error(
            "VDIM requires all source elements to be simplices in the appropriate dimension",
        )
        #for (idx, el) in pairs(IndexLinear(), elements(Y.mesh, E))
        for (idx, el) in enumerate(elements(Y.mesh, E))
            # Assume the vertices of the simplex are first three (low-order) nodes if `el` is a high-order element.
            vertices = el.vals[1:3]
            # TODO This will be wrong for a tetrahedron
            l1 = norm(vertices[1] - vertices[2])
            l2 = norm(vertices[2] - vertices[3])
            l3 = norm(vertices[3] - vertices[1])

            if ((l1^2 + l2^2 >= l3^2) && (l2^2 + l3^2 >= l1^2) && (l3^2 + l1^2 > l2^2))
                acuteright = true
            else
                acuteright = false
            end
            if acuteright
                # Compute the circumcenter and circumradius
                Bp = vertices[2] - vertices[1]
                Cp = vertices[3] - vertices[1]
                Dp = 2 * (Bp[1] * Cp[2] - Bp[2] * Cp[1])
                Upx = 1 / Dp * (Cp[2] * (Bp[1]^2 + Bp[1]^2) - Bp[2] * (Cp[1]^2 + Cp[2]^2))
                Upy = 1 / Dp * (Bp[2] * (Cp[1]^2 + Cp[1]^2) - Cp[2] * (Bp[1]^2 + Bp[2]^2))
                Up = SVector{2}(Upx, Upy)
                r[idx] = norm(Up)
                c[:, idx] = Up + vertices[1]
            else
                if (l1 >= l2) && (l1 >= l3)
                    c[:, idx] = (vertices[1] + vertices[2]) / 2
                    r[idx] = l1 / 2
                elseif (l2 >= l1) && (l2 >= l3)
                    c[:, idx] = (vertices[2] + vertices[3]) / 2
                    r[idx] = l2 / 2
                else
                    c[:, idx] = (vertices[1] + vertices[3]) / 2
                    r[idx] = l3 / 2
                end
            end
            interp_nodes_scaletranslate = [
                (q.coords - c[:, idx]) / r[idx] for q in Y[(1:num_basis).+(idx-1)*num_basis]
            ]
            b[(1:num_basis).+(idx-1)*num_basis, :] = [
                f(Tuple(eltype(Vop)(x) for x in q)) for
                q in interp_nodes_scaletranslate, f in p
            ]
        end
    end
    # Compute Θ <-- S * γ₁B - D * γ₀B - V * b + σ * B(x) using in-place matvec
    for n in 1:num_basis
        @views mul!(Θ[:, n], Sop, γ₁B[:, n])
        @views mul!(Θ[:, n], Dop, γ₀B[:, n], -1, 1)
        @views mul!(Θ[:, n], Vop, b[:, n], -1, 1)
        for i in 1:num_targets
            Θ[i, n] += σ * P[n](X[i])
        end
    end
    return b, Θ, c, r
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
    monomials = Vector{ElementaryPDESolutions.Polynomial{N,Float64}}()
    dirchlet_traces = Vector{ElementaryPDESolutions.Polynomial{N,Float64}}()
    T = return_type(neumann_trace, typeof(pde), eltype(dirchlet_traces))
    neumann_traces = Vector{T}()
    # iterate over N-tuples going from 0 to order
    for I in Iterators.product(ntuple(i -> 0:order, N)...)
        sum(I) > order && continue
        # define the monomial basis functions, and the corresponding solutions.
        # TODO: adapt this to vectorial case
        p   = ElementaryPDESolutions.Polynomial(I => 1.0)
        P   = polynomial_solution(pde, p)
        γ₁P = neumann_trace(pde, P)
        push!(monomials, p)
        push!(dirchlet_traces, P)
        push!(neumann_traces, γ₁P)
    end
    return monomials, dirchlet_traces, neumann_traces
end

# dispatch to the correct solver in ElementaryPDESolutions
function polynomial_solution(::Laplace, p::ElementaryPDESolutions.Polynomial)
    P = ElementaryPDESolutions.solve_laplace(p)
    return ElementaryPDESolutions.convert_coefs(P, Float64)
end

function polynomial_solution(pde::Helmholtz, p::ElementaryPDESolutions.Polynomial)
    k = pde.k
    P = ElementaryPDESolutions.solve_helmholtz(p; k)
    return ElementaryPDESolutions.convert_coefs(P, Float64)
end

function neumann_trace(
    ::Union{Laplace,Helmholtz},
    P::ElementaryPDESolutions.Polynomial{N,T},
) where {N,T}
    return _normal_derivative(P)
end

function _normal_derivative(P::ElementaryPDESolutions.Polynomial{N,T}) where {N,T}
    ∇P = ElementaryPDESolutions.gradient(P)
    return (q) -> dot(normal(q), ∇P(q))
end

function (∇P::NTuple{N,<:ElementaryPDESolutions.Polynomial})(x) where {N}
    return ntuple(n -> ∇P[n](x), N)
end

function (P::ElementaryPDESolutions.Polynomial)(q::QuadratureNode)
    x = q.coords.data
    return P(x)
end
