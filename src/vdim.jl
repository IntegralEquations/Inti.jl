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

- `interpolation_order`: the order of the polynomial interpolation. By default,
  the maximum order of the quadrature rules is used.
- `maxdist`: distance beyond which interactions are considered sufficiently far
  so that no correction is needed. This is used to determine a threshold for
  nearly-singular corrections.
- `center`: the center of the basis functions. By default, the basis functions
  are centered at the origin.
- `shift`: a boolean indicating whether the basis functions should be shifted
  and rescaled to each element.
"""

function vdim_correction(
        op,
        target,
        source::Quadrature,
        boundary::Quadrature,
        Sop,
        Dop,
        Vop;
        green_multiplier::Vector{<:Real},
        interpolation_order = nothing,
        maxdist = Inf,
        center = nothing,
        shift::Val{SHIFT} = Val(false),
    ) where {SHIFT}
    # variables for debugging the condition properties of the method
    vander_cond = vander_norm = rhs_norm = res_norm = shift_norm = -Inf
    T = eltype(Vop)
    @assert eltype(Dop) == eltype(Sop) == T "eltype of Sop, Dop, and Vop must match"
    # figure out if we are dealing with a scalar or vector PDE
    m, n = length(target), length(source)
    N = ambient_dimension(op)
    @assert ambient_dimension(source) == N "vdim only works for volume potentials"
    m, n = length(target), length(source)
    # a reasonable interpolation_order if not provided
    isnothing(interpolation_order) &&
        (interpolation_order = maximum(order, values(source.etype2qrule)))
    # by default basis centered at origin
    center = isnothing(center) ? zero(SVector{N, Float64}) : center
    p, P, γ₁P, multiindices = polynomial_solutions_vdim(op, interpolation_order, center)
    dict_near = etype_to_nearest_points(target, source; maxdist)
    R = _vdim_auxiliary_quantities(
        p,
        P,
        γ₁P,
        target,
        source,
        boundary,
        green_multiplier,
        Sop,
        Dop,
        Vop,
    )
    # compute sparse correction
    Is = Int[]
    Js = Int[]
    Vs = eltype(Vop)[]
    for (E, qtags) in source.etype2qtags
        els = elements(source.mesh, E)
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        for n in 1:ne
            # indices of nodes in element `n`
            isempty(near_list[n]) && continue
            jglob = @view qtags[:, n]
            # compute translation and scaling
            c, r = translation_and_scaling(els[n])
            if SHIFT
                iszero(center) || error("SHIFT is not implemented for non-zero center")
                L̃ = [f((q.coords - c) / r) for f in p, q in view(source, jglob)]
                S = change_of_basis(multiindices, p, c, r)
                F = svd(L̃)
                @debug (vander_cond = max(vander_cond, cond(L̃))) maxlog = 0
                @debug (shift_norm = max(shift_norm, norm(S))) maxlog = 0
                @debug (vander_norm = max(vander_norm, norm(L̃))) maxlog = 0
            else
                L = [f(q.coords) for f in p, q in view(source, jglob)]
                F = svd(L)
                @debug (vander_cond = max(vander_cond, cond(L))) maxlog = 0
                @debug (shift_norm = max(shift_norm, 1)) maxlog = 0
                @debug (vander_norm = max(vander_norm, norm(L))) maxlog = 0
            end
            # correct each target near the current element
            for i in near_list[n]
                b = @views R[i, :]
                wei = SHIFT ? F \ (S * b) : F \ b # weights for the current element and target i
                rhs_norm = max(rhs_norm, norm(b))
                res_norm = if SHIFT
                    max(res_norm, norm(L̃ * wei - S * b))
                else
                    max(res_norm, norm(L * wei - b))
                end
                for k in 1:nq
                    push!(Is, i)
                    push!(Js, jglob[k])
                    push!(Vs, -wei[k])
                end
            end
        end
    end
    @debug """Condition properties of vdim correction:
    |-- max interp. matrix condition: $vander_cond
    |-- max norm of source term:      $rhs_norm
    |-- max residual error:           $res_norm
    |-- max interp. matrix norm :     $vander_norm
    |-- max shift norm :              $shift_norm
    """
    δV = sparse(Is, Js, Vs, m, n)
    return δV
end

# function barrier for type stability purposes
function build_vander(vals_trg, pts, PFE_p, c, r)
    tmp = Vector{Float64}(undef, length(c))
    for i in 1:length(pts)
        tmp .= (pts[i].coords - c) / r
        ElementaryPDESolutions.fast_evaluate!(view(vals_trg, :, i), tmp, PFE_p)
    end
    return vals_trg
end

function _scaled_operator(op::AbstractDifferentialOperator{N}, scale) where {N}
    if op isa Helmholtz
        return Helmholtz(; k = scale * op.k, dim = N)
    elseif op isa Laplace
        return op
    else
        error("Unsupported operator for stabilized Local VDIM")
    end
end

function _lowfreq_operator(op::AbstractDifferentialOperator{N}) where {N}
    if op isa Helmholtz
        return Laplace(; dim = N)
    elseif op isa Laplace
        return op
    else
        error("Unsupported operator for stabilized Local VDIM")
    end
end

function local_vdim_correction(
        op,
        ::Type{Eltype},
        target,
        source::Quadrature,
        mesh::AbstractMesh,
        bdry_nodes;
        green_multiplier::Vector{<:Real},
        interpolation_order = nothing,
        quadrature_order = nothing,
        meshsize = 1.0,
        maxdist = Inf,
        center = nothing,
        shift::Val{SHIFT} = Val(false),
        form::Symbol = :contraction,
    ) where {SHIFT, Eltype}
    SHIFT || error("unsupported local VDIM without shifting")
    # `form` selects how the low-frequency local correction `δV = -(quad - exact)`
    # forms its `quad - exact` difference (both give the same sparse δV, columns
    # on the element's own nodes):
    #   :contraction — build the naive patch operator by contracting the literal
    #                  single-layer kernel entries against the basis values
    #                  (`Rq = Gmat*Bmat`) and subtract the exact operator `R`.
    #                  Uses the same kernel function (and `G(x,x)=0` convention)
    #                  as the dense/FMM forward map, so no analytic self-node
    #                  term is needed. This code path shows the math more
    #                  cleanly, but is much slower.
    #   :analytic    — obtain `quad - exact` directly from the rescaled-coordinate
    #                  Laplace expansion (eqs. (3.11)/(4.1)), reusing the scaled
    #                  single/double/volume operators already assembled by
    #                  `_local_vdim_auxiliary_quantities`. Mirrors the
    #                  manuscript, but must add an analytic self-node term to
    #                  account for the `G(x,x)=0` convention of the operator
    #                  being corrected, that depends on `H(z=0)`.
    form in (:contraction, :analytic) ||
        error("unknown local VDIM correction form: $form (expected :contraction or :analytic)")
    # variables for debugging the condition properties of the method
    vander_cond = vander_norm = rhs_norm = res_norm = shift_norm = -Inf
    # figure out if we are dealing with a scalar or vector PDE
    m, n = length(target), length(source)
    N = ambient_dimension(op)
    @assert ambient_dimension(source) == N "vdim only works for volume potentials"
    m, n = length(target), length(source)
    # a reasonable interpolation_order if not provided
    isnothing(interpolation_order) &&
        (interpolation_order = maximum(order, values(source.etype2qrule)))

    # Helmholtz PDE operator in x̂ coordinates where x = scale * x̂
    if op isa Helmholtz
        s = meshsize
    elseif op isa Laplace
        s = 1.0
    else
        error("not implemented")
    end
    op_hat = _scaled_operator(op, s)
    op_lowfreq = _lowfreq_operator(op)
    PFE_p_lowfreq, PFE_P_lowfreq, multiindices_lowfreq, monomials_indices_lowfreq =
        polynomial_solutions_local_vdim(op_lowfreq, interpolation_order + 2)
    PFE_p, PFE_P, multiindices, monomials_indices =
        polynomial_solutions_local_vdim(op_hat, interpolation_order)

    dict_near = etype_to_nearest_points(target, source; maxdist)
    bdry_kdtree = KDTree(bdry_nodes)
    # compute sparse correction
    Is = Int[]
    Js = Int[]
    Vs = Eltype[]
    for (E, qtags) in source.etype2qtags
        els = elements(source.mesh, E)
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        sizehint!(Is, ne * nq * nq)
        sizehint!(Js, ne * nq * nq)
        sizehint!(Vs, ne * nq * nq)
        num_basis = binomial(interpolation_order + N, N)
        L̃ = Matrix{Float64}(undef, nq, num_basis)
        vals_trg = Matrix{Float64}(undef, num_basis, nq)

        bdry_qorder = 2 * quadrature_order
        if N == 3
            bdry_qrule = _qrule_for_reference_shape(ReferenceSimplex{2}(), bdry_qorder)
            bdry_etype2qrule = OrderedDict(ReferenceSimplex{2} => bdry_qrule)
        else
            bdry_qrule = _qrule_for_reference_shape(ReferenceHyperCube{1}(), bdry_qorder)
            bdry_etype2qrule = OrderedDict(ReferenceHyperCube{1} => bdry_qrule)
        end
        vol_qrule = VioreanuRokhlin(; domain = domain(E), order = quadrature_order)
        vol_etype2qrule = OrderedDict(E => vol_qrule)

        topo_neighs = 1
        neighbors = topological_neighbors(mesh, topo_neighs)

        # preallocated local quadratures, reused (refilled in place) across the
        # element loop so the qnodes buffers are not reallocated every iteration
        Yvol = Quadrature{N, Float64}(
            nothing,
            vol_etype2qrule,
            QuadratureNode{N, Float64}[],
            OrderedDict{DataType, Matrix{Int}}(),
        )
        Ybdry = Quadrature{N, Float64}(
            nothing,
            bdry_etype2qrule,
            QuadratureNode{N, Float64}[],
            OrderedDict{DataType, Matrix{Int}}(),
        )
        # reusable scratch for the per-element auxiliary quantities
        ws = LocalVDIMWorkspace{Eltype, N}()

        for n in 1:ne
            # indices of nodes in element `n`
            isempty(near_list[n]) && continue
            c, r, diam = translation_and_scaling(els[n])
            # Low-frequency (kr small) elements use the Laplace-based expansion
            # of Section 3.1/4; otherwise the high-frequency rescaling of
            # Section 3.2 with s = meshsize (so that k*s = O(1) is fixed and
            # s/r is bounded from above and below) is stable.

            # Run Laplace through low-frequency path too, for stability.
            if op isa Laplace || (op isa Helmholtz && r * op.k < 10^(-3))
                lowfreq = true
                Yvol, Ybdry, need_layer_corr, els_idxs = _local_vdim_construct_local_quadratures(
                    N,
                    mesh,
                    neighbors,
                    n,
                    c,
                    r,
                    diam,
                    bdry_kdtree,
                    Yvol,
                    Ybdry,
                    bdry_qrule,
                    vol_qrule,
                )
                if form === :analytic
                    # R holds the physical (quad - exact) difference directly,
                    # from the rescaled expansion + analytic self-node term.
                    R = _lowfreq_vdim_cancellation_quantities(
                        op,
                        op_lowfreq,
                        c,
                        r,
                        num_basis,
                        PFE_p_lowfreq,
                        PFE_P_lowfreq,
                        multiindices,
                        multiindices_lowfreq,
                        monomials_indices,
                        monomials_indices_lowfreq,
                        target[near_list[n]],
                        green_multiplier[near_list[n]],
                        Yvol,
                        Ybdry,
                        diam,
                        need_layer_corr,
                        ws,
                    )
                else
                    # R holds the exact local operator; the naive quad side is
                    # built by contraction in the assembly below.
                    R = _lowfreq_vdim_auxiliary_quantities(
                        op,
                        op_lowfreq,
                        c,
                        r,
                        num_basis,
                        PFE_p_lowfreq,
                        PFE_P_lowfreq,
                        multiindices,
                        multiindices_lowfreq,
                        monomials_indices,
                        monomials_indices_lowfreq,
                        target[near_list[n]],
                        green_multiplier[near_list[n]],
                        Yvol,
                        Ybdry,
                        diam,
                        need_layer_corr
                    )
                end
            else
                lowfreq = false
                Yvol, Ybdry, need_layer_corr, els_idxs = _local_vdim_construct_local_quadratures(
                    N,
                    mesh,
                    neighbors,
                    n,
                    c,
                    s,
                    diam,
                    bdry_kdtree,
                    Yvol,
                    Ybdry,
                    bdry_qrule,
                    vol_qrule,
                )
                R, b = _local_vdim_auxiliary_quantities(
                    op_hat,
                    c,
                    s,
                    PFE_p,
                    PFE_P,
                    target[near_list[n]],
                    green_multiplier[near_list[n]],
                    Yvol,
                    Ybdry,
                    diam,
                    need_layer_corr,
                    ws,
                )
            end
            jglob = @view qtags[:, n]
            L̃ .= transpose(build_vander(vals_trg, view(source, jglob), PFE_p, c, r))
            Linv = pinv(L̃)
            if lowfreq
                # eq. (2.10): δV = -(quad - exact) applied through the
                # interpolant of element `n`, with columns on the element's own
                # nodes only (and including the V_N[f - fₙ] remainder). How the
                # `quad - exact` difference `Δ` is formed depends on `form`.
                if form === :analytic
                    # R already holds the physical (quad - exact) directly.
                    Δ = R
                else
                    # `R` holds the exact local operator V_exact,N[p_β]; the quad
                    # side is obtained by contracting the naive operator entries
                    # against the basis values, using the same kernel function as
                    # the dense assembly (and consistent with the FMM), including
                    # its zero convention at coincident points — so it matches the
                    # operator being corrected by construction.
                    #
                    # The naive-quadrature contraction is the matrix product
                    # Rq = Gmat * Bmat, with
                    #   Gmat[ii, j] = G(xᵢ, yⱼ)   (near target × patch node)
                    #   Bmat[j,  β] = wⱼ pβ(yⱼ)   (patch node × monomial)
                    Gker = SingleLayerKernel(op)
                    ntarg = length(near_list[n])
                    nsrc = nq * length(els_idxs)
                    Gmat = Matrix{eltype(R)}(undef, ntarg, nsrc)
                    Bmat = Matrix{Float64}(undef, nsrc, num_basis)
                    pvals = Vector{Float64}(undef, num_basis)
                    ytmp = Vector{Float64}(undef, N)
                    jcol = 0
                    for el_idx in els_idxs
                        for j in @view qtags[:, el_idx]
                            jcol += 1
                            yq = source[j]
                            ytmp .= (coords(yq) - c) / r
                            ElementaryPDESolutions.fast_evaluate!(pvals, ytmp, PFE_p)
                            w = yq.weight
                            @inbounds for β in 1:num_basis
                                Bmat[jcol, β] = w * pvals[β]
                            end
                            @inbounds for (ii, i) in enumerate(near_list[n])
                                Gmat[ii, jcol] = Gker(target[i], yq)
                            end
                        end
                    end
                    Δ = Gmat * Bmat - R
                end
                wei = transpose(Linv) * transpose(Δ)
                append!(Is, repeat(near_list[n]; inner = nq))
                append!(Js, repeat(jglob; outer = length(near_list[n])))
                append!(Vs, -wei)
            else
                S = ws.Sdiagvec
                resize!(S, length(multiindices))
                S .= s^2 * (s / r) .^ (abs.(multiindices))
                R .*= transpose(S)
                wei = transpose(Linv) * transpose(R)
                # δV = -(quad - exact)
                append!(Is, repeat(near_list[n]; inner = nq))
                append!(Js, repeat(jglob; outer = length(near_list[n])))
                append!(Vs, -wei)
            end
        end
    end
    @debug """Condition properties of vdim correction:
    |-- max interp. matrix condition: $vander_cond
    |-- max norm of source term:      $rhs_norm
    |-- max residual error:           $res_norm
    |-- max interp. matrix norm :     $vander_norm
    |-- max shift norm :              $shift_norm
    """
    δV = sparse(Is, Js, Vs, m, n)
    return δV
end

function change_of_basis(multiindices, p, c, r)
    nbasis = length(multiindices)
    P = zeros(nbasis, nbasis)
    for i in 1:nbasis
        α = multiindices[i]
        for j in 1:nbasis
            β = multiindices[j]
            β ≤ α || continue
            # P[i, j] = prod((-c) .^ ((α - β).indices)) / r^abs(α) / factorial(α
            # - β)
            γ = α - β
            p_γ = p[findfirst(x -> x == γ, multiindices)] # p_{\alpha - \beta}
            P[i, j] = p_γ(-c) / r^abs(α)
        end
    end
    return P
end

function translation_and_scaling(el::LagrangeTriangle)
    vertices = el.vals[1:3]
    l1 = norm(vertices[1] - vertices[2])
    l2 = norm(vertices[2] - vertices[3])
    l3 = norm(vertices[3] - vertices[1])
    diam = max(l1, l2, l3)
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
        Upx = 1 / Dp * (Cp[2] * (Bp[1]^2 + Bp[2]^2) - Bp[2] * (Cp[1]^2 + Cp[2]^2))
        Upy = 1 / Dp * (Bp[1] * (Cp[1]^2 + Cp[2]^2) - Cp[1] * (Bp[1]^2 + Bp[2]^2))
        Up = SVector{2}(Upx, Upy)
        r = norm(Up)
        c = Up + vertices[1]
    else
        if (l1 >= l2) && (l1 >= l3)
            c = (vertices[1] + vertices[2]) / 2
            r = l1 / 2
        elseif (l2 >= l1) && (l2 >= l3)
            c = (vertices[2] + vertices[3]) / 2
            r = l2 / 2
        else
            c = (vertices[1] + vertices[3]) / 2
            r = l3 / 2
        end
    end
    return c, r, diam
end

function translation_and_scaling(el::ParametricElement{ReferenceSimplex{3}})
    straight_nodes =
        [el([0.0, 0.0, 0.0]), el([1.0, 0.0, 0.0]), el([0.0, 1.0, 0.0]), el([0.0, 0.0, 1.0])]
    return translation_and_scaling(
        LagrangeElement{ReferenceSimplex{3}, 4, SVector{3, Float64}}(straight_nodes),
    )
end

function translation_and_scaling(el::ParametricElement{ReferenceSimplex{2}})
    straight_nodes = [el([1.0e-18, 1.0e-18]), el([1.0, 0.0]), el([0.0, 1.0])]
    return translation_and_scaling(
        LagrangeElement{ReferenceSimplex{2}, 3, SVector{2, Float64}}(straight_nodes),
    )
end

function translation_and_scaling(el::LagrangeTetrahedron)
    vertices = el.vals[1:4]
    # Compute the circumcenter in barycentric coordinates
    # formulas here are due to: https://math.stackexchange.com/questions/2863613/tetrahedron-centers
    a = norm(vertices[4] - vertices[1])
    b = norm(vertices[2] - vertices[4])
    c = norm(vertices[3] - vertices[4])
    d = norm(vertices[3] - vertices[2])
    e = norm(vertices[3] - vertices[1])
    f = norm(vertices[2] - vertices[1])
    diam = max(a, b, c, d, e, f)
    f² = f^2
    a² = a^2
    b² = b^2
    c² = c^2
    d² = d^2
    e² = e^2

    ρ =
        a² * d² * (-d² + e² + f²) + b² * e² * (d² - e² + f²) + c² * f² * (d² + e² - f²) -
        2 * d² * e² * f²
    α =
        a² * d² * (b² + c² - d²) + e² * b² * (-b² + c² + d²) + f² * c² * (b² - c² + d²) -
        2 * b² * c² * d²
    β =
        b² * e² * (a² + c² - e²) + d² * a² * (-a² + c² + e²) + f² * c² * (a² - c² + e²) -
        2 * a² * c² * e²
    γ =
        c² * f² * (a² + b² - f²) + d² * a² * (-a² + b² + f²) + e² * b² * (a² - b² + f²) -
        2 * a² * b² * f²
    if (ρ >= 0 && α >= 0 && β >= 0 + γ >= 0)
        # circumcenter lays inside `el`
        center =
            (α * vertices[1] + β * vertices[2] + γ * vertices[3] + ρ * vertices[4]) /
            (ρ + α + β + γ)
        # ref: https://math.stackexchange.com/questions/1087011/calculating-the-radius-of-the-circumscribed-sphere-of-an-arbitrary-tetrahedron
        R = sqrt(1 / 2 * (β * f² + γ * e² + ρ * a²) / (ρ + α + β + γ))
    else
        if (a >= b && a >= c && a >= d && a >= e && a >= f)
            center = (vertices[1] + vertices[4]) / 2
            R = a / 2
        elseif (b >= a && b >= c && b >= d && b >= e && b >= f)
            center = (vertices[2] + vertices[4]) / 2
            R = b / 2
        elseif (c >= a && c >= b && c >= d && c >= e && c >= f)
            center = (vertices[3] + vertices[4]) / 2
            R = c / 2
        elseif (d >= a && d >= b && d >= c && d >= e && d >= f)
            center = (vertices[3] + vertices[2]) / 2
            R = d / 2
        elseif (e >= a && e >= b && e >= c && e >= d && e >= f)
            center = (vertices[3] + vertices[1]) / 2
            R = e / 2
        else
            center = (vertices[2] + vertices[1]) / 2
            R = f / 2
        end
    end
    return center, R, diam
end

# function barrier for type stability purposes
_newbord_line(vtxs) = LagrangeLine(SVector{2}(vtxs))

# function barrier for type stability purposes
_newbord_tri(vtxs) = LagrangeElement{ReferenceSimplex{2}}(SVector{3}(vtxs))

function _local_vdim_construct_local_quadratures(
        N,
        mesh,
        neighbors,
        el,
        center,
        scale,
        diam,
        bdry_kdtree,
        Yvol,
        Ybdry,
        bdry_qrule,
        vol_qrule
    )
    # construct the local region
    Etype = first(element_types(mesh))
    el_neighs = neighbors[(Etype, el)]

    T = first(el_neighs)[1]
    els_idxs = [i[2] for i in collect(el_neighs)]
    els_list = mesh.etype2els[Etype][els_idxs]

    loc_bdry = boundarynd(T, els_idxs, mesh)
    # TODO handle curved boundary of Γ??
    if N == 2
        bords = LagrangeElement{ReferenceHyperCube{N - 1}, 2, SVector{N, Float64}}[]
    else
        bords = LagrangeElement{ReferenceSimplex{N - 1}, 3, SVector{N, Float64}}[]
    end

    for idxs in loc_bdry
        vtxs = nodes(mesh)[idxs]
        if N == 2
            bord = _newbord_line(vtxs)
        else
            bord = _newbord_tri(vtxs)
        end
        push!(bords, bord)
    end

    # Check if we need to do near-singular layer potential evaluation
    vertices = mesh.etype2els[Etype][el].vals[vertices_idxs(Etype)]
    need_layer_corr = sum(inrangecount(bdry_kdtree, vertices, diam / 2)) > 0

    # Now begin working in x̂ coordinates where x = scale * x̂

    # build O(h) volume neighbors, reusing the preallocated quadrature buffers
    build_local_quadrature!(Yvol, els_list, vol_qrule; center, scale)
    build_local_quadrature!(Ybdry, bords, bdry_qrule; center, scale)

    return Yvol, Ybdry, need_layer_corr, els_idxs
end

# Reusable scratch buffers for LVDIM.  Each field is a flat Vector whose
# capacity is retained across calls (growing via `resize!` only at new
# high-water marks), so after the first few elements no per-element allocation
# of these buffers occurs. `T` is the kernel/matrix eltype.
struct LocalVDIMWorkspace{T, N}
    Xshift::Vector{SVector{N, Float64}}
    Smat::Vector{T}
    Dmat::Vector{T}
    Vmat::Vector{T}
    Θ::Vector{T}
    b::Vector{Float64}
    γ₀B::Vector{Float64}
    γ₁B::Vector{Float64}
    P::Vector{Float64}
    grad::Vector{Float64}
    gm::Vector{Float64}
    Sdiagvec::Vector{Float64}
end

function LocalVDIMWorkspace{T, N}() where {T, N}
    return LocalVDIMWorkspace{T, N}(
        SVector{N, Float64}[],
        T[], T[], T[], T[],
        Float64[], Float64[], Float64[], Float64[], Float64[],
        Float64[],Float64[],
    )
end

# Return a `dims`-shaped, BLAS-strided view backed by `buf`, growing `buf` only
# when a larger block is needed. `reshape(view(buf, 1:len), dims)` is a packed
# `StridedArray`, so the result stays on the BLAS path in `mul!`.
@inline function _ws_reshape(buf::Vector, dims::Dims)
    len = prod(dims)
    length(buf) < len && resize!(buf, len)
    return reshape(view(buf, 1:len), dims)
end
_ws_reshape(buf::Vector, dims::Integer...) = _ws_reshape(buf, dims)

function _local_vdim_auxiliary_quantities(
        op::AbstractDifferentialOperator{N},
        center,
        scale,
        PFE_p,
        PFE_P,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr,
        ws::LocalVDIMWorkspace{T, N},
    ) where {T, N}
    # TODO handle derivative case
    G = SingleLayerKernel(op)
    dG = DoubleLayerKernel(op)
    num_basis = length(PFE_P)
    num_targets = length(X)
    nb = length(Ybdry)
    nv = length(Yvol)

    Xshift = ws.Xshift
    resize!(Xshift, num_targets)
    @inbounds for i in 1:num_targets
        Xshift[i] = (coords(X[i]) - center) / scale
    end
    Sop = IntegralOperator(G, Xshift, Ybdry)
    Dop = IntegralOperator(dG, Xshift, Ybdry)
    Vop = IntegralOperator(G, Xshift, Yvol)
    Smat = _ws_reshape(ws.Smat, num_targets, nb)
    Dmat = _ws_reshape(ws.Dmat, num_targets, nb)
    Vmat = _ws_reshape(ws.Vmat, num_targets, nv)
    assemble_matrix!(Smat, Sop; threads = false)
    assemble_matrix!(Dmat, Dop; threads = false)
    assemble_matrix!(Vmat, Vop; threads = false)
    Smap = LinearMap(Smat)
    Dmap = LinearMap(Dmat)
    if need_layer_corr
        green_multiplier = resize!(ws.gm, num_targets)
        copyto!(green_multiplier, μ)
        δS, δD = bdim_correction(
            op,
            Xshift,
            Ybdry,
            Smat,
            Dmat;
            green_multiplier,
            maxdist = diam / scale,
            derivative = false,
        )

        #Smat += δS
        Smap += LinearMap(δS)
        #Dmat += δD
        Dmap += LinearMap(δD)
    end

    b = _ws_reshape(ws.b, nv, num_basis)
    γ₁B = _ws_reshape(ws.γ₁B, nb, num_basis)
    γ₀B = _ws_reshape(ws.γ₀B, nb, num_basis)
    P = _ws_reshape(ws.P, num_targets, num_basis)
    grad = _ws_reshape(ws.grad, num_basis, N, nb)

    for i in 1:nv
        ElementaryPDESolutions.fast_evaluate!(view(b, i, :), Yvol[i].coords, PFE_p)
    end
    for i in 1:num_targets
        ElementaryPDESolutions.fast_evaluate!(view(P, i, :), Xshift[i], PFE_P)
    end
    for i in 1:nb
        ElementaryPDESolutions.fast_evaluate_with_jacobian!(
            view(γ₀B, i, :),
            view(grad, :, :, i),
            Ybdry[i].coords,
            PFE_P,
        )
    end
    for i in 1:nb
        for j in 1:num_basis
            γ₁B[i, j] = 0
            for k in 1:N
                γ₁B[i, j] += grad[j, k, i] * Ybdry[i].normal[k] #nrml_bdry_vec[i][k]
            end
        end
    end

    Θ = _ws_reshape(ws.Θ, num_targets, num_basis)
    fill!(Θ, zero(T))
    # Compute Θ <-- S * γ₁B - D * γ₀B + V * b + σ * B(x) using in-place matvec
    for n in 1:num_basis
        @views mul!(Θ[:, n], Smap, γ₁B[:, n])
        @views mul!(Θ[:, n], Dmap, γ₀B[:, n], -1, 1)
        @views mul!(Θ[:, n], Vmat, b[:, n], 1, 1)
        for i in 1:num_targets
            Θ[i, n] += μ[i] * P[i, n]
        end
    end
    return Θ, b
end

"""
    _local_vdim_exact_quantities(op, center, scale, PFE_p, PFE_P, X, μ, Yvol, Ybdry, diam, need_layer_corr)

Exact local volume potentials on the rescaled patch: return `(E, b)` where
`E[i, n] = Ṽ_exact[pₙ](x̃ᵢ)` is the exact volume potential of the monomial `pₙ`
over the patch, evaluated via Green's identity (no naive patch quadrature is
involved), and `b` holds the monomial values at the patch quadrature nodes.
"""
function _local_vdim_exact_quantities(
        op::AbstractDifferentialOperator{N},
        center,
        scale,
        PFE_p,
        PFE_P,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr
    ) where {N}
    G = SingleLayerKernel(op)
    dG = DoubleLayerKernel(op)
    Xshift = [(coords(q) - center) / scale for q in X]
    Sop = IntegralOperator(G, Xshift, Ybdry)
    Dop = IntegralOperator(dG, Xshift, Ybdry)
    Smat = assemble_matrix(Sop)
    Dmat = assemble_matrix(Dop)
    if need_layer_corr
        green_multiplier = collect(Float64, μ)
        δS, δD = bdim_correction(
            op,
            Xshift,
            Ybdry,
            Smat,
            Dmat;
            green_multiplier,
            maxdist = diam / scale,
            derivative = false,
        )
        Smat += δS
        Dmat += δD
    end

    num_basis = length(PFE_P)
    num_targets = length(X)
    b = Matrix{Float64}(undef, length(Yvol), num_basis)
    γ₁B = Matrix{Float64}(undef, length(Ybdry), num_basis)
    γ₀B = Matrix{Float64}(undef, length(Ybdry), num_basis)
    P = Matrix{Float64}(undef, length(X), num_basis)
    grad = Array{Float64}(undef, num_basis, N, length(Ybdry))

    for i in 1:length(Yvol)
        ElementaryPDESolutions.fast_evaluate!(view(b, i, :), Yvol[i].coords, PFE_p)
    end
    for i in 1:length(X)
        ElementaryPDESolutions.fast_evaluate!(view(P, i, :), Xshift[i], PFE_P)
    end
    for i in 1:length(Ybdry)
        ElementaryPDESolutions.fast_evaluate_with_jacobian!(
            view(γ₀B, i, :),
            view(grad, :, :, i),
            Ybdry[i].coords,
            PFE_P,
        )
    end
    for i in 1:length(Ybdry)
        for j in 1:num_basis
            γ₁B[i, j] = 0
            for k in 1:N
                γ₁B[i, j] += grad[j, k, i] * Ybdry[i].normal[k]
            end
        end
    end

    # Green's identity, valid for any target location relative to the patch:
    # Ṽ_exact[pₙ](x̃) = -S[γ₁Pₙ](x̃) + D[γ₀Pₙ](x̃) - μ(x̃) Pₙ(x̃)
    # (μ = -1 inside the patch, 0 outside, -1/2 on its smooth boundary)
    E = zeros(eltype(Sop), num_targets, num_basis)
    for n in 1:num_basis
        @views mul!(E[:, n], Smat, γ₁B[:, n], -1, 0)
        @views mul!(E[:, n], Dmat, γ₀B[:, n], 1, 1)
        for i in 1:num_targets
            E[i, n] -= μ[i] * P[i, n]
        end
    end
    return E, b
end

function _lowfreq_vdim_auxiliary_quantities(
        op::Laplace{2},
        op_lowfreq::Laplace{2},
        center,
        scale,
        num_basis,
        PFE_p_lowfreq,
        PFE_P_lowfreq,
        multiindices,
        multiindices_lowfreq,
        monomials_indices,
        monomials_indices_lowfreq,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr
    )
    E, b = _local_vdim_exact_quantities(
        op_lowfreq,
        center,
        scale,
        PFE_p_lowfreq,
        PFE_P_lowfreq,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr
    )
    num_targets = length(X)

    # Exact local operator via eq. (4.1):
    # V_N[p_β](x) = scale² (Ṽ_exact[p̃_β](x̃) - log(scale)/(2π) ∫_Ñ p̃_β dỹ),
    # with ∫_Ñ p̃_β evaluated by the local quadrature rule.
    R = Matrix{eltype(E)}(undef, num_targets, num_basis)
    for n in 1:num_basis
        β = multiindices[n]
        col = monomials_indices_lowfreq[β]
        intp = sum(Yvol[j].weight * b[j, col] for j in 1:length(Yvol))
        for i in 1:num_targets
            R[i, n] = scale^2 * (E[i, col] - 1 / (2π) * log(scale) * intp)
        end
    end
    return R
end

function _lowfreq_vdim_auxiliary_quantities(
        op::Helmholtz{2},
        op_lowfreq::Laplace{2},
        center,
        scale,
        num_basis,
        PFE_p_lowfreq,
        PFE_P_lowfreq,
        multiindices,
        multiindices_lowfreq,
        monomials_indices,
        monomials_indices_lowfreq,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr
    )
    Xshift = [(coords(q) - center) / scale for q in X]
    # Exact Laplace potentials on the rescaled patch
    E, b = _local_vdim_exact_quantities(
        op_lowfreq,
        center,
        scale,
        PFE_p_lowfreq,
        PFE_P_lowfreq,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr,
    )

    num_targets = length(X)
    R = zeros(ComplexF64, num_targets, num_basis)
    kr2 = (op.k * scale)^2
    γ = 0.5772156649015328606

    # Smooth part H of the kernel split G_kr = -1/(2π) log|x̃-ỹ| J₀(z) + H(z),
    # z² = (k*scale)²|x̃-ỹ|² (cf. eqs. (3.4)-(3.6)), with both series truncated
    # at O(z⁶); the entries carry the local quadrature weights so that Hmat*b
    # approximates ∫_Ñ H(x̃,ỹ) p̃_β(ỹ) dỹ.
    Hmat = Matrix{ComplexF64}(undef, num_targets, length(Yvol))
    for i in 1:num_targets
        for j in 1:length(Yvol)
            z2 =
                kr2 * (
                (Xshift[i][1] - Yvol[j].coords[1])^2 +
                    (Xshift[i][2] - Yvol[j].coords[2])^2
            )
            bessj0 = 0.0
            for m in 0:3
                bessj0 += (-1)^m * (z2 / 4)^m / factorial(m)^2
            end
            ytail =
                z2 / 4 - (1 + 1 / 2) * (z2 / 4)^2 / 4 +
                (1 + 1 / 2 + 1 / 3) * (z2 / 4)^3 / 36
            Hmat[i, j] =
                ((im / 4 - 1 / (2π) * (γ + 1 / 2 * log(kr2 / 4))) * bessj0 - 1 / (2π) * ytail) *
                Yvol[j].weight
        end
    end

    # Exact local operator via eqs. (3.11)-(3.12): the log-kernel part is the
    # P_J⁽¹⁾ combination of exact Laplace potentials of nearby monomials, the
    # smooth part is ∫_Ñ H p̃_β computed with the local quadrature.
    for n in 1:num_basis
        beta = multiindices[n]
        beta10 = beta + MultiIndex((1, 0))
        beta01 = beta + MultiIndex((0, 1))
        beta20 = beta + MultiIndex((2, 0))
        beta02 = beta + MultiIndex((0, 2))
        for j in 1:num_targets
            x1t = Xshift[j][1]
            x2t = Xshift[j][2]
            R[j, n] =
                scale^2 * (
                (1 - 1 / 4 * kr2 * (x1t^2 + x2t^2)) * E[j, monomials_indices_lowfreq[beta]] +
                    1 / 2 * kr2 * x1t * factorial(beta10) / factorial(beta) * E[j, monomials_indices_lowfreq[beta10]] +
                    1 / 2 * kr2 * x2t * factorial(beta01) / factorial(beta) * E[j, monomials_indices_lowfreq[beta01]] -
                    1 / 4 * kr2 * factorial(beta20) / factorial(beta) * E[j, monomials_indices_lowfreq[beta20]] -
                    1 / 4 * kr2 * factorial(beta02) / factorial(beta) * E[j, monomials_indices_lowfreq[beta02]]
            )
        end
        @views R[:, n] .+= scale^2 .* (Hmat * b[:, monomials_indices_lowfreq[beta]])
    end
    return R
end

"""
    _lowfreq_vdim_cancellation_quantities(op, op_lowfreq, center, scale, num_basis, ...)

Return the physical *(naive quad - exact)* local volume operator `R[i, β] =
V_quad,N[p_β](xᵢ) - V_exact,N[p_β](xᵢ)` for the low-frequency path, formed
analytically from the rescaled-coordinate Laplace expansion rather than by an
explicit kernel contraction (see [`local_vdim_correction`](@ref) `form` kwarg).

The key reuse is that `_local_vdim_auxiliary_quantities` already returns
`Θ = V_scaled·b - E_scaled`, i.e. the scaled (naive quad - exact) Laplace
potentials of the padded monomial basis. The log(scale)/(2π)∫p̃ term of eq.
(4.1) (Laplace) and the smooth ∫H p̃ term of eq. (3.12) (Helmholtz) cancel
between quad and exact, leaving only an analytic self-node term that accounts
for the `G(x,x)=0` convention of the operator being corrected.
"""
function _lowfreq_vdim_cancellation_quantities(
        op::Laplace{2},
        op_lowfreq::Laplace{2},
        center,
        scale,
        num_basis,
        PFE_p_lowfreq,
        PFE_P_lowfreq,
        multiindices,
        multiindices_lowfreq,
        monomials_indices,
        monomials_indices_lowfreq,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr,
        ws::LocalVDIMWorkspace,
    )
    Θ, b = _local_vdim_auxiliary_quantities(
        op_lowfreq,
        center,
        scale,
        PFE_p_lowfreq,
        PFE_P_lowfreq,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr,
        ws,
    )
    num_targets = length(X)
    # quad - exact = scale² Θ; the log(scale)/(2π)∫p̃ term cancels.
    R = Matrix{eltype(Θ)}(undef, num_targets, num_basis)
    for n in 1:num_basis
        col = monomials_indices_lowfreq[multiindices[n]]
        for i in 1:num_targets
            R[i, n] = scale^2 * Θ[i, col]
        end
    end
    # Self-node term: the physical naive operator uses G(x,x) = 0, omitting the
    # -log(scale)/(2π) shift the rescaling applies at the coincident node.
    Xshift = [(coords(q) - center) / scale for q in X]
    for i in 1:num_targets
        for j in 1:length(Yvol)
            if norm(Xshift[i] - Yvol[j].coords) ≤ SAME_POINT_TOLERANCE
                coef = 1 / (2π) * log(scale) * Yvol[j].weight * scale^2
                for n in 1:num_basis
                    R[i, n] += coef * b[j, monomials_indices_lowfreq[multiindices[n]]]
                end
            end
        end
    end
    return R
end

function _lowfreq_vdim_cancellation_quantities(
        op::Helmholtz{2},
        op_lowfreq::Laplace{2},
        center,
        scale,
        num_basis,
        PFE_p_lowfreq,
        PFE_P_lowfreq,
        multiindices,
        multiindices_lowfreq,
        monomials_indices,
        monomials_indices_lowfreq,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr,
        ws::LocalVDIMWorkspace,
    )
    Θ, b = _local_vdim_auxiliary_quantities(
        op_lowfreq,
        center,
        scale,
        PFE_p_lowfreq,
        PFE_P_lowfreq,
        X,
        μ,
        Yvol,
        Ybdry,
        diam,
        need_layer_corr,
        ws,
    )
    Xshift = [(coords(q) - center) / scale for q in X]
    num_targets = length(X)
    R = zeros(ComplexF64, num_targets, num_basis)
    kr2 = (op.k * scale)^2
    γ = 0.5772156649015328606

    # quad - exact: the P_J⁽¹⁾ combination (eq. (3.11)) of the Laplace
    # (quad - exact) Θ of nearby monomials. The smooth ∫H p̃ part cancels
    # between quad and exact except at the self node, handled below.
    for n in 1:num_basis
        beta = multiindices[n]
        beta10 = beta + MultiIndex((1, 0))
        beta01 = beta + MultiIndex((0, 1))
        beta20 = beta + MultiIndex((2, 0))
        beta02 = beta + MultiIndex((0, 2))
        for j in 1:num_targets
            x1t = Xshift[j][1]
            x2t = Xshift[j][2]
            R[j, n] =
                scale^2 * (
                (1 - 1 / 4 * kr2 * (x1t^2 + x2t^2)) * Θ[j, monomials_indices_lowfreq[beta]] +
                    1 / 2 * kr2 * x1t * factorial(beta10) / factorial(beta) * Θ[j, monomials_indices_lowfreq[beta10]] +
                    1 / 2 * kr2 * x2t * factorial(beta01) / factorial(beta) * Θ[j, monomials_indices_lowfreq[beta01]] -
                    1 / 4 * kr2 * factorial(beta20) / factorial(beta) * Θ[j, monomials_indices_lowfreq[beta20]] -
                    1 / 4 * kr2 * factorial(beta02) / factorial(beta) * Θ[j, monomials_indices_lowfreq[beta02]]
            )
        end
    end
    # Self-node term: physical naive drops H(0) = i/4 - (γ + log(kr/2))/(2π)
    # at the coincident node (G_k(x,x) = 0).
    H0 = im / 4 - 1 / (2π) * (γ + 1 / 2 * log(kr2 / 4))
    for i in 1:num_targets
        for j in 1:length(Yvol)
            if norm(Xshift[i] - Yvol[j].coords) ≤ SAME_POINT_TOLERANCE
                w = Yvol[j].weight * scale^2
                for n in 1:num_basis
                    R[i, n] -= H0 * w * b[j, monomials_indices_lowfreq[multiindices[n]]]
                end
            end
        end
    end
    return R
end

function _vdim_auxiliary_quantities(
        p,
        P,
        γ₁P,
        X,
        Y::Quadrature,
        Γ::Quadrature,
        μ,
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
    # Compute Θ <-- S * γ₁B - D * γ₀B + V * b + σ * B(x) using in-place matvec
    for n in 1:num_basis
        @views mul!(Θ[:, n], Sop, γ₁B[:, n])
        @views mul!(Θ[:, n], Dop, γ₀B[:, n], -1, 1)
        @views mul!(Θ[:, n], Vop, b[:, n], 1, 1)
        for i in 1:num_targets
            Θ[i, n] += μ[i] * P[n](X[i])
        end
    end
    return Θ
end

"""
    vdim_mesh_center(msh)

Point `x` which minimizes ∑ (x-xⱼ)²/r²ⱼ, where xⱼ and rⱼ are the circumcenter
and circumradius of the elements of `msh`, respectively.
"""
function vdim_mesh_center(msh::AbstractMesh)
    N = ambient_dimension(msh)
    M = 0.0
    xc = zero(SVector{N, Float64})
    for E in element_types(msh)
        for el in elements(msh, E)
            c, r = translation_and_scaling(el)
            # w = 1/r^2
            w = 1
            M += w
            xc += c * w
        end
    end
    return xc / M
end
"""
    polynomial_solutions_local_vdim(op, order)

For every monomial term `pₙ` of degree `order`, compute a polynomial `Pₙ` such
that `ℒ[Pₙ] = pₙ`, where `ℒ` is the differential operator `op`.
This function returns `{pₙ,Pₙ,γ₁Pₙ}`, where `γ₁Pₙ` is the generalized Neumann
trace of `Pₙ`.
"""
function polynomial_solutions_local_vdim(op::AbstractDifferentialOperator, order::Integer)
    N = ambient_dimension(op)
    # create empty arrays to store the monomials, solutions, and traces. For the
    # neumann trace, we try to infer the concrete return type instead of simply
    # having a vector of `Function`.
    monomials = Vector{ElementaryPDESolutions.Polynomial{N, Float64}}()
    poly_solutions = Vector{ElementaryPDESolutions.Polynomial{N, Float64}}()
    multiindices = Vector{MultiIndex{N}}()
    # iterate over N-tuples going from 0 to order
    for I in Iterators.product(ntuple(i -> 0:order, N)...)
        sum(I) > order && continue
        # define the monomial basis functions, and the corresponding solutions.
        # TODO: adapt this to vectorial case
        p = ElementaryPDESolutions.Polynomial(I => 1 / factorial(MultiIndex(I)))
        P = polynomial_solution(op, p)
        push!(multiindices, MultiIndex(I))
        push!(monomials, p)
        push!(poly_solutions, P)
    end
    monomials_indices = Dict(multiindices .=> 1:length(multiindices))

    PFE_monomials = ElementaryPDESolutions.assemble_fastevaluator(monomials, Float64)
    PFE_polysolutions =
        ElementaryPDESolutions.assemble_fastevaluator(poly_solutions, Float64)

    return PFE_monomials, PFE_polysolutions, multiindices, monomials_indices
end

"""
    polynomial_solutions_vdim(op, order[, center])

For every monomial term `pₙ` of degree `order`, compute a polynomial `Pₙ` such
that `ℒ[Pₙ] = pₙ`, where `ℒ` is the differential operator associated with `op`.
This function returns `{pₙ,Pₙ,γ₁Pₙ}`, where `γ₁Pₙ` is the generalized Neumann
trace of `Pₙ`.

Passing a point `center` will shift the monomials and solutions accordingly.
"""
function polynomial_solutions_vdim(
        op::AbstractDifferentialOperator,
        order::Integer,
        center = nothing,
    )
    N = ambient_dimension(op)
    center = isnothing(center) ? zero(SVector{N, Float64}) : center
    # create empty arrays to store the monomials, solutions, and traces. For the
    # neumann trace, we try to infer the concrete return type instead of simply
    # having a vector of `Function`.
    monomials = Vector{ElementaryPDESolutions.Polynomial{N, Float64}}()
    dirchlet_traces = Vector{ElementaryPDESolutions.Polynomial{N, Float64}}()
    T = return_type(neumann_trace, typeof(op), eltype(dirchlet_traces))
    neumann_traces = Vector{T}()
    multiindices = Vector{MultiIndex{N}}()
    # iterate over N-tuples going from 0 to order
    for I in Iterators.product(ntuple(i -> 0:order, N)...)
        sum(I) > order && continue
        # define the monomial basis functions, and the corresponding solutions.
        # TODO: adapt this to vectorial case
        p = ElementaryPDESolutions.Polynomial(I => 1 / factorial(MultiIndex(I)))
        P = polynomial_solution(op, p)
        γ₁P = neumann_trace(op, P)
        push!(multiindices, MultiIndex(I))
        push!(monomials, p)
        push!(dirchlet_traces, P)
        push!(neumann_traces, γ₁P)
    end
    monomial_shift = map(monomials) do f
        return (q) -> f(coords(q) - center)
    end
    dirchlet_shift = map(dirchlet_traces) do f
        return (q) -> f(coords(q) - center)
    end
    neumann_shift = map(neumann_traces) do f
        # return (q) -> f((coords = q.coords - center, normal = q.normal))
        return (q) -> f(coords(q) - center, normal(q))
    end
    return monomial_shift, dirchlet_shift, neumann_shift, multiindices
    # return monomials, dirchlet_traces, neumann_traces, multiindices
end

# TODO/FIXME: This is fixed in `main` but we have not yet merged that into this branch
# Dispatch to the correct solver in ElementaryPDESolutions.
# CONVENTION: `polynomial_solution(op, p)` returns `P` with `L[P] = -p`. The
# Green-identity assemblies in (global and local) VDIM rely on this sign:
# V[p](x) = -(S[γ₁P](x) - D[γ₀P](x) + μ(x)P(x)). All operators must follow the
# same convention; note `solve_laplace(Q)` and `solve_helmholtz(Q; k)` both
# return solutions with L[P] = +Q, hence the `-p` below.
function polynomial_solution(::Laplace, p::ElementaryPDESolutions.Polynomial)
    P = ElementaryPDESolutions.solve_laplace(-p)
    return ElementaryPDESolutions.convert_coefs(P, Float64)
end

function polynomial_solution(op::Helmholtz, p::ElementaryPDESolutions.Polynomial)
    k = op.k
    P = ElementaryPDESolutions.solve_helmholtz(-p; k)
    return ElementaryPDESolutions.convert_coefs(P, Float64)
end

function polynomial_solution(op::Yukawa, p::ElementaryPDESolutions.Polynomial)
    k = im * op.λ
    P = ElementaryPDESolutions.solve_helmholtz(-p; k)
    return ElementaryPDESolutions.convert_coefs(P, Float64)
end

function neumann_trace(
        ::Union{Laplace, Helmholtz, Yukawa},
        P::ElementaryPDESolutions.Polynomial{N, T},
    ) where {N, T}
    return _normal_derivative(P)
end

function _normal_derivative(P::ElementaryPDESolutions.Polynomial{N, T}) where {N, T}
    ∇P = ElementaryPDESolutions.gradient(P)
    # return (q) -> dot(normal(q), ∇P(coords(q)))
    return (x, n) -> dot(n, ∇P(x))
end

function (∇P::NTuple{N, <:ElementaryPDESolutions.Polynomial})(x) where {N}
    return ntuple(n -> ∇P[n](x), N)
end

function (P::ElementaryPDESolutions.Polynomial)(q::QuadratureNode)
    x = q.coords.data
    return P(x)
end
