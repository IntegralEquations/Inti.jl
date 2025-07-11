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
    center = isnothing(center) ? zero(SVector{N,Float64}) : center
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
                    push!(Vs, wei[k])
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

function _scaled_operator(op::AbstractDifferentialOperator, scale)
    if op isa Helmholtz
        return Helmholtz(; k = scale * op.k, dim = ambient_dimension(op))
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
        return Laplace(; dim = N)
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
) where {SHIFT,Eltype}
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
    s = meshsize
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
            bdry_etype2qrule = Dict(ReferenceSimplex{2} => bdry_qrule)
        else
            bdry_qrule = _qrule_for_reference_shape(ReferenceHyperCube{1}(), bdry_qorder)
            bdry_etype2qrule = Dict(ReferenceHyperCube{1} => bdry_qrule)
        end
        vol_qrule = VioreanuRokhlin(; domain = domain(E), order = quadrature_order)
        vol_etype2qrule = Dict(E => vol_qrule)

        topo_neighs = 1
        neighbors = topological_neighbors(mesh, topo_neighs)

        for n in 1:ne
            # indices of nodes in element `n`
            isempty(near_list[n]) && continue
            c, r, diam = translation_and_scaling(els[n])
            s=1.0
            if false
                #if r * op.k < 10^(-3)
                lowfreq = true
                Yvol, Ybdry, need_layer_corr = _local_vdim_construct_local_quadratures(
                    N,
                    mesh,
                    neighbors,
                    n,
                    c,
                    r,
                    diam,
                    bdry_kdtree,
                    bdry_etype2qrule,
                    vol_etype2qrule,
                    bdry_qrule,
                    vol_qrule,
                )
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
                    green_multiplier,
                    Yvol,
                    Ybdry,
                    diam,
                    need_layer_corr;
                )
                Yvol_s1, Ybdry_s1, need_layer_corr_s1 = _local_vdim_construct_local_quadratures(
                    N,
                    mesh,
                    neighbors,
                    n,
                    c,
                    1.0,
                    diam,
                    bdry_kdtree,
                    bdry_etype2qrule,
                    vol_etype2qrule,
                    bdry_qrule,
                    vol_qrule,
                )
                R_s1, b_s1 = _local_vdim_auxiliary_quantities(
                    op_hat,
                    c,
                    1.0,
                    PFE_p,
                    PFE_P,
                    target[near_list[n]],
                    green_multiplier,
                    Yvol_s1,
                    Ybdry_s1,
                    diam,
                    need_layer_corr_s1;
                )
            else
                lowfreq = false
                Yvol, Ybdry, need_layer_corr = _local_vdim_construct_local_quadratures(
                    N,
                    mesh,
                    neighbors,
                    n,
                    c,
                    s,
                    diam,
                    bdry_kdtree,
                    bdry_etype2qrule,
                    vol_etype2qrule,
                    bdry_qrule,
                    vol_qrule,
                )
        #if isdefined(Main, :Infiltrator)
        #    Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
        #  end
                R, b = _local_vdim_auxiliary_quantities(
                    op_hat,
                    c,
                    s,
                    PFE_p,
                    PFE_P,
                    target[near_list[n]],
                    green_multiplier,
                    Yvol,
                    Ybdry,
                    diam,
                    need_layer_corr;
                )
                Yvol_lowfreq, Ybdry_lowfreq, need_layer_corr_lowfreq = _local_vdim_construct_local_quadratures(
                    N,
                    mesh,
                    neighbors,
                    n,
                    c,
                    r,
                    diam,
                    bdry_kdtree,
                    bdry_etype2qrule,
                    vol_etype2qrule,
                    bdry_qrule,
                    vol_qrule,
                )
                R_lowfreq = _lowfreq_vdim_auxiliary_quantities(
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
                    green_multiplier,
                    Yvol_lowfreq,
                    Ybdry_lowfreq,
                    diam,
                    need_layer_corr_lowfreq;
                )
                #Yvol_s1, Ybdry_s1, need_layer_corr_s1 = _local_vdim_construct_local_quadratures(
                #    N,
                #    mesh,
                #    neighbors,
                #    n,
                #    c,
                #    1.0,
                #    diam,
                #    bdry_kdtree,
                #    bdry_etype2qrule,
                #    vol_etype2qrule,
                #    bdry_qrule,
                #    vol_qrule,
                #)
                #R_s1, b_s1 = _local_vdim_auxiliary_quantities(
                #    op_hat,
                #    c,
                #    1.0,
                #    PFE_p,
                #    PFE_P,
                #    target[near_list[n]],
                #    green_multiplier,
                #    Yvol_s1,
                #    Ybdry_s1,
                #    diam,
                #    need_layer_corr_s1;
                #)
        #if isdefined(Main, :Infiltrator)
        #    Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
        #  end
            end
            jglob = @view qtags[:, n]
            # compute translation and scaling
            if SHIFT
                # TODO copy this from (part of) the output of _local_vdim_auxiliary_quantities ?
                L̃ .= transpose(build_vander(vals_trg, view(source, jglob), PFE_p, c, r))
                Linv = pinv(L̃)
                if !lowfreq
                    S = s^2 * Diagonal((s / r) .^ (abs.(multiindices)))
                    #S_s1 = Diagonal((1 / r) .^ (abs.(multiindices)))
                    wei = transpose(Linv) * S * transpose(R)
                    #wei_s1 = transpose(Linv) * S_s1 * transpose(R_s1)

                    #area = 0.0
                    #for i in 1:length(Yvol)
                    #    area += Yvol[i].weight
                    #end
                    #area_s1 = 0.0
                    #for i in 1:length(Yvol_s1)
                    #    area_s1 += Yvol_s1[i].weight
                    #end
                    #area_contour = 0.0
                    #for i in 1:length(Ybdry)
                    #    x = Ybdry[i].coords[1]
                    #    y = Ybdry[i].coords[2]
                    #    nx = Ybdry[i].normal[1]
                    #    ny = Ybdry[i].normal[2]
                    #    area_contour += (x/2 * nx + y/2*ny) * Ybdry[i].weight
                    #end
                    #area_contour_s1 = 0.0
                    #for i in 1:length(Ybdry_s1)
                    #    x = Ybdry_s1[i].coords[1]
                    #    y = Ybdry_s1[i].coords[2]
                    #    nx = Ybdry_s1[i].normal[1]
                    #    ny = Ybdry_s1[i].normal[2]
                    #    area_contour_s1 += (x/2 * nx + y/2*ny) * Ybdry_s1[i].weight
                    #end

                    #Vint = 0.0
                    #for i in 1:length(Yvol)
                    #    x = Yvol[i].coords[1]
                    #    y = Yvol[i].coords[2]
                    #    Vint += (3*x^2 + 3*y^2) * Yvol[i].weight
                    #end
                    #Vint_s1 = 0.0
                    #for i in 1:length(Yvol)
                    #    x = Yvol_s1[i].coords[1]
                    #    y = Yvol_s1[i].coords[2]
                    #    Vint_s1 += (3*x^2 + 3*y^2) * Yvol_s1[i].weight
                    #end
                    #Vcontour = 0.0
                    #for i in 1:length(Ybdry)
                    #    x = Ybdry[i].coords[1]
                    #    y = Ybdry[i].coords[2]
                    #    nx = Ybdry[i].normal[1]
                    #    ny = Ybdry[i].normal[2]
                    #    Vcontour += (x^3 * nx + y^3*ny) * Ybdry[i].weight
                    #end
                    #Vcontour_s1 = 0.0
                    #for i in 1:length(Ybdry_s1)
                    #    x = Ybdry_s1[i].coords[1]
                    #    y = Ybdry_s1[i].coords[2]
                    #    nx = Ybdry_s1[i].normal[1]
                    #    ny = Ybdry_s1[i].normal[2]
                    #    Vcontour_s1 += (x^3 * nx + y^3*ny) * Ybdry_s1[i].weight
                    #end
                    if isdefined(Main, :Infiltrator)
                        Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
                      end
                else
                    D = Vector{Float64}(undef, num_basis)
                    D .= r^2
                    D = Diagonal(D)
                    wei = transpose(Linv) * D * transpose(R)

                    S_s1 = Diagonal((1 / r) .^ (abs.(multiindices)))
                    wei_s1 = transpose(Linv) * S_s1 * transpose(R_s1)
                    if isdefined(Main, :Infiltrator)
                        Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
                    end
                end
            else
                error("unsupported local VDIM without shifting")
            end
            # correct each target near the current element
            append!(Is, repeat(near_list[n]; inner = nq))
            append!(Js, repeat(jglob; outer = length(near_list[n])))
            append!(Vs, wei)
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
_newbord_line(vtxs) = LagrangeLine(SVector{3}(vtxs))

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
    bdry_etype2qrule,
    vol_etype2qrule,
    bdry_qrule,
    vol_qrule;
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
        bords = LagrangeElement{ReferenceHyperCube{N - 1},3,SVector{N,Float64}}[]
    else
        bords = LagrangeElement{ReferenceSimplex{N - 1},3,SVector{N,Float64}}[]
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

    # build O(h) volume neighbors
    Yvol = Quadrature(Float64, els_list, vol_etype2qrule, vol_qrule; center, scale)
    Ybdry = Quadrature(Float64, bords, bdry_etype2qrule, bdry_qrule; center, scale)

    return Yvol, Ybdry, need_layer_corr
end

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
    need_layer_corr;
) where {N}
    # TODO handle derivative case
    G = SingleLayerKernel(op)
    dG = DoubleLayerKernel(op)
    Xshift = [(q.coords - center) / scale for q in X]
    Sop = IntegralOperator(G, Xshift, Ybdry)
    Dop = IntegralOperator(dG, Xshift, Ybdry)
    Vop = IntegralOperator(G, Xshift, Yvol)
    Smat = assemble_matrix(Sop)
    Dmat = assemble_matrix(Dop)
    Vmat = assemble_matrix(Vop)
    if need_layer_corr
        μloc = _green_multiplier(:inside)
        green_multiplier = fill(μloc, length(X))
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
                γ₁B[i, j] += grad[j, k, i] * Ybdry[i].normal[k]#nrml_bdry_vec[i][k]
            end
        end
    end

    Θ = zeros(eltype(Vop), num_targets, num_basis)
    # Compute Θ <-- S * γ₁B - D * γ₀B - V * b + σ * B(x) using in-place matvec
    for n in 1:num_basis
        @views mul!(Θ[:, n], Smat, γ₁B[:, n])
        @views mul!(Θ[:, n], Dmat, γ₀B[:, n], -1, 1)
        @views mul!(Θ[:, n], Vmat, b[:, n], -1, 1)
        for i in 1:num_targets
            Θ[i, n] += μ[i] * P[i, n]
        end
    end
    return Θ, b
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
    need_layer_corr;
)
    θ, b = _local_vdim_auxiliary_quantities(
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
        need_layer_corr;
    )
    Xshift = [(q.coords - center) / scale for q in X]
    num_targets = length(Xshift)

    Hmat = Matrix{eltype(θ)}(undef, num_targets, length(Yvol))
    for n in 1:num_targets
        for j in 1:length(Yvol)
            Hmat[n, j] = Yvol[j].weight
        end
    end

    R = Matrix{eltype(θ)}(undef, num_targets, num_basis)
    for n in 1:num_basis
        β = multiindices[n]
        R[:, n] = θ[:, monomials_indices_lowfreq[β]]
        R[:, n] += log(scale) * Hmat * b[:, monomials_indices_lowfreq[β]]
    end
    if isdefined(Main, :Infiltrator)
        Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
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
    need_layer_corr;
)
    Xshift = [(q.coords - center) / scale for q in X]
    # Laplace
    θ, b = _local_vdim_auxiliary_quantities(
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

    # Set up integral operators just to find the element type of the correction matrix R
    G = SingleLayerKernel(op)
    Vop = IntegralOperator(G, Xshift, Yvol)
    R = zeros(eltype(Vop), num_targets, num_basis)
    kr2 = (op.k * scale)^2
    γ = 0.57721566490153286060

    Hmat = Matrix{ComplexF64}(undef, length(X), length(Yvol))
    #Hmat .= 0
    #@show scale * op.k
    for i in 1:num_targets
        for j in 1:length(Yvol)
            z2 =
                kr2 * (
                    (Xshift[i][1] - Yvol[j].coords[1])^2 +
                    (Xshift[i][2] - Yvol[j].coords[2])^2
                )
            bessj0 = 0
            for k in 0:3
                bessj0 += (-1)^k * (1 / 4 * z2)^k / (factorial(k)^2)
            end
            Hmat[i, j] =
                (
                    (1 + 2 * im / pi * (γ + 1 / 2 * log(kr2 / 4))) * bessj0 +
                    2 * im / pi *
                    (1 / 4 * z2 - 3 / 2 * (1 / 4 * z2)^2 / 4 + 11 / 3456 * z2^3)
                ) *
                Yvol[j].weight *
                scale^2
        end
    end

    for n in 1:num_basis
        beta = multiindices[n]
        beta10 = beta + Inti.MultiIndex((1, 0))
        beta01 = beta + Inti.MultiIndex((0, 1))
        beta20 = beta + Inti.MultiIndex((2, 0))
        beta02 = beta + Inti.MultiIndex((0, 2))
        for j in 1:num_targets
            x1t = Xshift[j][1]
            x2t = Xshift[j][2]
            R[j, n] =
                (1 - 1 / 4 * kr2 * (x1t^2 + x2t^2)) * θ[j, monomials_indices_lowfreq[beta]]
            +1 / 2 * kr2 * x1t * θ[j, monomials_indices_lowfreq[beta10]]
            +1 / 2 * kr2 * x2t * θ[j, monomials_indices_lowfreq[beta01]]
            -1 / 4 * kr2 * θ[j, monomials_indices_lowfreq[beta20]]
            -1 / 4 * kr2 * θ[j, monomials_indices_lowfreq[beta02]]
        end
        R[:, n] .*= 2 * im / pi * scale^2
        # Commenting this yields more accuracy
        R[:, n] += Hmat * b[:, monomials_indices_lowfreq[beta]]
    end
    return R, b
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
    # Compute Θ <-- S * γ₁B - D * γ₀B - V * b + σ * B(x) using in-place matvec
    for n in 1:num_basis
        @views mul!(Θ[:, n], Sop, γ₁B[:, n])
        @views mul!(Θ[:, n], Dop, γ₀B[:, n], -1, 1)
        @views mul!(Θ[:, n], Vop, b[:, n], -1, 1)
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
    xc = zero(SVector{N,Float64})
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
    monomials = Vector{ElementaryPDESolutions.Polynomial{N,Float64}}()
    poly_solutions = Vector{ElementaryPDESolutions.Polynomial{N,Float64}}()
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
    center = isnothing(center) ? zero(SVector{N,Float64}) : center
    # create empty arrays to store the monomials, solutions, and traces. For the
    # neumann trace, we try to infer the concrete return type instead of simply
    # having a vector of `Function`.
    monomials = Vector{ElementaryPDESolutions.Polynomial{N,Float64}}()
    dirchlet_traces = Vector{ElementaryPDESolutions.Polynomial{N,Float64}}()
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

# dispatch to the correct solver in ElementaryPDESolutions
function polynomial_solution(::Laplace, p::ElementaryPDESolutions.Polynomial)
    P = ElementaryPDESolutions.solve_laplace(p)
    return ElementaryPDESolutions.convert_coefs(P, Float64)
end

function polynomial_solution(op::Helmholtz, p::ElementaryPDESolutions.Polynomial)
    k = op.k
    P = ElementaryPDESolutions.solve_helmholtz(p; k)
    return ElementaryPDESolutions.convert_coefs(P, Float64)
end

function polynomial_solution(op::Yukawa, p::ElementaryPDESolutions.Polynomial)
    k = im * op.λ
    P = ElementaryPDESolutions.solve_helmholtz(p; k)
    return ElementaryPDESolutions.convert_coefs(P, Float64)
end

function neumann_trace(
    ::Union{Laplace,Helmholtz,Yukawa},
    P::ElementaryPDESolutions.Polynomial{N,T},
) where {N,T}
    return _normal_derivative(P)
end

function _normal_derivative(P::ElementaryPDESolutions.Polynomial{N,T}) where {N,T}
    ∇P = ElementaryPDESolutions.gradient(P)
    # return (q) -> dot(normal(q), ∇P(coords(q)))
    return (x, n) -> dot(n, ∇P(x))
end

function (∇P::NTuple{N,<:ElementaryPDESolutions.Polynomial})(x) where {N}
    return ntuple(n -> ∇P[n](x), N)
end

function (P::ElementaryPDESolutions.Polynomial)(q::QuadratureNode)
    x = q.coords.data
    return P(x)
end
