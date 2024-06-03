#=
Routines for correcting the singular or nearly singular entries of an integral
operator based on:
    1. Identifying (a priori) the entries of the integral operator which need to
    be corrected. This is done based on distance between points/elements and
    analytical knowldge of the convergence rate of the quadrature and the
    underlying kernels
    2. Performing a change-of-variables to alleviate the singularity of the
    integrand
    3. Doing an adaptive quadrature on the new integrand using HCubature.jl

The specialized integration is precomputed on the reference element for each
quadrature node of the standard quadrature.
=#

"""
    adaptive_correction(iop::IntegralOperator; tol, maxdist = farfield_distance(iop; atol), maxsplit = 1000])

Given an integral operator `iop`, this function provides a sparse correction to
`iop` for the entries `i,j` such that the distance between the `i`-th target and
the `j`-th source is less than `maxdist`.

Choosing `maxdist` is a trade-off between accuracy and efficiency. The smaller
the value, the fewer corrections are needed, but this may compromise the
accuracy. For a *fixed* quadrature, the size of `maxdist` has to grow as the
tolerance `tol` decreases. The default `[farfield_distance(iop; tol)](@ref)
provides a heuristic to determine a suitable `maxdist`.

The correction is computed by using the [`adaptive_integration`](@ref) routine,
with a tolerance `atol` and a maximum number of subdivisions `maxsplit`; see
[`adaptive_integration`](@ref) for more details.
"""
function adaptive_correction(iop::IntegralOperator; tol, maxdist = nothing, maxsplit = 1000)
    maxdist = isnothing(maxdist) ? farfield_distance(iop; tol) : maxdist
    # unpack type-unstable fields in iop, allocate output, and dispatch
    X, Y, K = target(iop), source(iop), kernel(iop)
    # normalize maxdist
    maxdist = if isa(maxdist, Number)
        Dict(E => maxdist for E in element_types(mesh(Y)))
    elseif isnothing(maxdist)
        farfield_distance(Y, K; tol = tol)
    else
        maxdist
    end
    dict_near = near_interaction_list(X, Y; tol = maxdist)
    T = eltype(iop)
    msh = mesh(Y)
    correction = (I = Int[], J = Int[], V = T[])
    for E in element_types(msh)
        # dispatch on element type
        nearlist = dict_near[E]
        qrule    = quadrature_rule(Y, E)
        L        = lagrange_basis(qrule)
        iter     = elements(msh, E)
        _adaptive_correction_etype!(
            correction,
            iter,
            qrule,
            L,
            nearlist,
            X,
            Y,
            K,
            maxdist[E],
            tol,
            maxsplit,
        )
    end
    m, n = size(iop)
    return sparse(correction.I, correction.J, correction.V, m, n)
end

@noinline function _adaptive_correction_etype!(
    correction,
    iter,
    qreg,
    L,
    nearlist,
    X,
    Y,
    K,
    maxdist,
    atol,
    maxsplit,
)
    E = eltype(iter)
    Xqnodes = collect(X)
    Yqnodes = collect(Y)
    τ̂ = domain(E)
    N = geometric_dimension(τ̂)
    a, b = svector(i -> 0.0, N), svector(i -> 1.0, N)
    x̂, ŵ = collect.(qreg())
    el2qtags = etype2qtags(Y, E)
    # buffers = [hcubature_buffer(x -> one(eltype(correction.V)) * L(a) * first(ŵ), a, b) for _ in 1:Threads.nthreads()]
    buffer = allocate_buffer(x -> one(eltype(correction.V)) * L(a) * first(ŵ), τ̂)
    max_abser = 0.0
    max_reler = 0.0
    for n in 1:length(iter)
        el = iter[n]
        jglob = view(el2qtags, :, n)
        inear = nearlist[n]
        for i in inear
            xnode = Xqnodes[i]
            # closest quadrature node
            dmin, j = findmin(
                n -> norm(coords(xnode) - coords(Yqnodes[jglob[n]])),
                1:length(jglob),
            )
            x̂nearest = x̂[j]
            dmin > maxdist && continue
            issingular = iszero(dmin)
            integrand = (ŷ) -> begin
                y   = el(ŷ)
                jac = jacobian(el, ŷ)
                ν   = _normal(jac)
                τ′  = _integration_measure(jac)
                return K(xnode, (coords = y, normal = ν)) * L(ŷ) * τ′
            end
            # use hcubature for singular integration of lagrange basis.
            if issingular
                W, er = adaptive_integration_singular(
                    integrand,
                    τ̂,
                    x̂nearest;
                    buffer,
                    atol,
                    maxsplit,
                )
            else
                W, er = adaptive_integration(integrand, τ̂; buffer, atol, maxsplit)
            end
            max_abser = max(max_abser, er)
            max_reler = max(max_reler, er / norm(W))
            # @lock lck for (k, j) in enumerate(jglob)
            for (k, j) in enumerate(jglob)
                qx, qy = Xqnodes[i], Yqnodes[j]
                push!(correction.I, i)
                push!(correction.J, j)
                push!(correction.V, W[k] - K(qx, qy) * weight(qy))
            end
        end
    end
    if max_abser > atol
        @warn """failed to meet tolerance of $atol with $maxsplit evaluations: maximum absolute error of
        $max_abser. Consider increasing maxsplit."""
    end
    return correction
end

"""
    adaptive_integration_singular(f, τ̂, x̂ₛ; kwargs...)

Similar to [`adaptive_integration`](@ref), but indicates that `f` has an
isolated (integrable) singularity at `x̂ₛ ∈ x̂ₛ`.

The integration is performed by splitting `τ̂` so that `x̂ₛ` is a fixed vertex,
guaranteeing that `f` is never evaluated at `x̂ₛ`. Aditionally, a suitable
change of variables may be applied to alleviate the singularity and improve the
rate of convergence.
"""
function adaptive_integration_singular(f, τ̂::ReferenceLine, x̂ₛ; kwargs...)
    τ₁, τ₂ = decompose(τ̂, x̂ₛ) # split τ̂ at x̂ₛ so that xₛ = τ₁(0) = τ₂(0)
    l₁, l₂ = integration_measure(τ₁), integration_measure(τ₂)
    return adaptive_integration(τ̂; kwargs...) do x
        return (f(τ₁(x)) * l₁ + f(τ₂(x)) * l₂)
    end
end

function near_interaction_list(QX::Quadrature, QY::Quadrature; tol)
    X = [coords(q) for q in QX]
    msh = mesh(QY)
    return near_interaction_list(X, msh; tol)
end
function near_interaction_list(X, QY::Quadrature; tol)
    msh = mesh(QY)
    return near_interaction_list(X, msh; tol)
end

"""
    decompose(s::ReferenceShape,x)

Decompose an [`ReferenceShape`](@ref) into [`LagrangeElement`](@ref)s so
that `x` is a fixed vertex of the children elements.

The decomposed elements may be oriented differently than the parent, and thus
care has to be taken regarding e.g. normal vectors.
"""
function decompose(ln::ReferenceLine, x::SVector{1,<:Real} = SVector(0.5))
    # @assert x ∈ ln
    a, b = vertices(ln)
    # two lines with x on (0,) reference vertex
    return LagrangeLine(x, a), LagrangeLine(x, b)
end
