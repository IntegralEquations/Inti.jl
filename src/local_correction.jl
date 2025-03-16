"""
    local_correction(iop::IntegralOperator; [maxdist, tol, threads = true, kwargs...])

This function computes a sparse correction for the integral operator `iop`, addressing its singular or nearly singular entries.

The parameter `maxdist` specifies the maximum distance between target points  and source
elements to be considered for correction (only interactions within this distance are
corrected).

The parameter `tol` defines the tolerance for the adaptive quadrature used to compute the
corrections for singular or nearly singular entries.

Selecting `maxdist` and `tol` involves balancing accuracy and computational cost. A smaller
`maxdist` reduces the number of corrections but may impact accuracy. Conversely, a smaller
`tol` improves correction accuracy but increases computational expense. The ideal values for
`maxdist` and `tol` depend on the kernel and the mesh/quadrature rule applied.

By default, `maxdist` and `tol` are estimated using the
[`local_correction_dist_and_tol`](@ref), but it is often possible to improve performance by
manually tunning these parameters.

Additional keyword arguments are passed to [`adaptive_quadrature`](@ref); see its
documentation for more information.
"""
function local_correction(
    iop::IntegralOperator;
    maxdist = nothing,
    tol = nothing,
    threads = true,
    kwargs...,
)
    if isnothing(maxdist) || isnothing(tol)
        maxdist_, tol_ = local_correction_dist_and_tol(iop)
    end
    maxdist    = isnothing(maxdist) ? maxdist_ : maxdist
    tol        = isnothing(tol) ? 10 * tol_ : tol
    msh        = mesh(target(iop))
    quads_dict = Dict()
    for E in element_types(msh)
        ref_domain = reference_domain(E)
        quads = (
            nearfield_quad = adaptive_quadrature(ref_domain; atol = tol, kwargs...),
            radial_quad    = adaptive_quadrature(ReferenceLine(); atol = tol, kwargs...),
            angular_quad   = adaptive_quadrature(ReferenceLine(); atol = tol, kwargs...),
        )
        quads_dict[E] = quads
    end
    return local_correction(iop, maxdist, quads_dict, threads)
end

"""
    local_correction(iop::IntegralOperator, maxdist, quads_dict::Dict, threads = true)

This method extends `local_correction(iop::IntegralOperator; maxdist, tol)` by allowing the
user to provide a custom dictionary `quads_dict` containing quadrature rules for each
reference element type present in the mesh of `source(iop)`.

The dictionary `quads_dict` must adhere to the following structure:
- `quads_dict[E].nearfield_quad`: A function that integrates over the nearfield of the
  reference element type `E`. Used in the nearly-singular correction.
- `quads_dict[E].radial_quad`: A function that integrates over the radial direction of the
  reference element type `E`. Used in the singular correction.
- `quads_dict[E].angular_quad`: A function that integrates over the angular direction of the
  reference element type `E`. Used in the singular correction.

This flexibility enables to fine-tune the quadrature rules for specific element types,
improving accuracy or performance based on the problem's requirements.

!!! note "Finite part integrals"
    This function handles strongly singular integrals by implementing the method of
    Guiggiani [guiggiani1992general(@cite), which consists of a polar change of variables
    followed by a Laurent series expansion of the integrand.
"""
function local_correction(iop, maxdist, quads_dict::Dict, threads = true)
    # unpack type-unstable fields in iop, allocate output, and dispatch
    X, Y, K = target(iop), source(iop), kernel(iop)
    @assert X === Y "source and target of integraloperator must coincide"
    dict_near = near_interaction_list([coords(x) for x in X], mesh(Y); tol = maxdist)
    T = eltype(iop)
    msh = mesh(Y)
    # use the singularity order of the kernel and the geometric dimension to compute the
    # singularity order of the kernel in polar/spherical coordinates
    geo_dim    = geometric_dimension(msh)
    p          = singularity_order(K) # K(x,y) ~ |x-y|^{-p} as y -> 0
    sing_order = if isnothing(p)
        @warn "missing method `singularity_order` for kernel. Assuming finite part integral."
        2
    else
        p - (geo_dim - 1) # in polar coordinates you muliply by r^{geo_dim-1}
    end
    # allocate output in a sparse matrix style
    correction = (I = Int[], J = Int[], V = T[])
    # loop over element types in the source mesh, unpack, and dispatch to type-stable
    # function
    for E in element_types(msh)
        nearlist = dict_near[E]
        els = elements(msh, E)
        # append the regular quadrature rule to the list of quads for the element type E
        # radial singularity order
        quads = merge(quads_dict[E], (regular_quad = quadrature_rule(Y, E),))
        L = lagrange_basis(quads.regular_quad)
        _local_correction_etype!(
            correction,
            els,
            quads,
            L,
            nearlist,
            X,
            Y,
            K,
            Val(sing_order),
            maxdist,
            threads,
        )
    end
    m, n = size(iop)
    return sparse(correction.I, correction.J, correction.V, m, n)
end

@noinline function _local_correction_etype!(
    correction,
    el_iter,
    quads,
    L,
    nearlist,
    X,
    Y,
    K,
    sorder, # singularity order in polar coordinates
    nearfield_distance,
    threads,
)
    E = eltype(el_iter)
    Xqnodes = collect(X)
    Yqnodes = collect(Y)
    # reference quadrature nodes and weights
    x̂ = qcoords(quads.regular_quad) |> collect
    el2qtags = etype2qtags(Y, E)
    nel = length(el_iter)
    lck = Threads.SpinLock()
    # lck = ReentrantLock()
    @maybe_threads threads for n in 1:nel
        el = el_iter[n]
        jglob = view(el2qtags, :, n)
        # inear = union(nearlist[n], jglob) # make sure to include nearfield nodes AND the element nodes
        inear = nearlist[n]
        for i in inear
            xnode = Xqnodes[i]
            # closest quadrature node
            dmin, j = findmin(
                n -> norm(coords(xnode) - coords(Yqnodes[jglob[n]])),
                1:length(jglob),
            )
            x̂nearest = x̂[j]
            dmin > nearfield_distance && continue
            # If singular, use Guiggiani's method. Otherwise use an ovesampled quadrature
            if iszero(dmin)
                W = guiggiani_singular_integral(
                    K,
                    L,
                    x̂nearest,
                    el,
                    quads.radial_quad,
                    quads.angular_quad,
                    sorder,
                )
            else
                integrand = (ŷ) -> begin
                    y = el(ŷ)
                    jac = jacobian(el, ŷ)
                    ν = _normal(jac)
                    τ′ = _integration_measure(jac)
                    M = K(xnode, (coords = y, normal = ν))
                    v = L(ŷ)
                    map(v -> M * v, v) * τ′
                end
                W = quads.nearfield_quad(integrand)
            end
            @lock lck for (k, j) in enumerate(jglob)
                qx, qy = Xqnodes[i], Yqnodes[j]
                push!(correction.I, i)
                push!(correction.J, j)
                push!(correction.V, W[k] - K(qx, qy) * weight(qy))
            end
        end
    end
    return correction
end

"""
    polar_decomposition(shape::ReferenceSquare, x̂::SVector{2,Float64})

Decompose the square `[0,1] × [0,1]` into four triangles, and return four tuples of the form
`θₛ, θₑ, ρ` where `θₛ` and `θₑ` are the initial and final angles of the triangle, and `ρ` is
the function that gives the distance from `x̂` to the border of the square in the direction
`θ`.
"""
function polar_decomposition(::ReferenceSquare, x::SVector{2,<:Number})
    theta1 = atan(1 - x[2], 1 - x[1])
    theta2 = atan(x[1], 1 - x[2]) + π / 2
    theta3 = atan(x[2], x[1]) + π
    theta4 = atan(1 - x[1], x[2]) + 3π / 2
    rho1 = θ -> (1 - x[2]) / sin(θ)
    rho2 = θ -> x[1] / (-cos(θ))
    rho3 = θ -> x[2] / (-sin(θ))
    rho4 = θ -> (1 - x[1]) / cos(θ)
    return (theta1, theta2, rho1),
    (theta2, theta3, rho2),
    (theta3, theta4, rho3),
    (theta4, theta1 + 2π, rho4)
end

"""
	polar_decomposition(shape::ReferenceTriangle, x̂::SVector{2,Float64})

Decompose the triangle `{x,y ≥ 0, x + y ≤ 1}` into three triangles, and return three tuples
of the form `θₛ, θₑ, ρ` where `θₛ` and `θₑ` are the initial and final angles of the
triangle, and `ρ` is the function that gives the distance from `x̂` to the border of the
triangle in the direction `θ`.
"""
function polar_decomposition(::ReferenceTriangle, x::SVector{2,<:Number})
    theta1 = atan(x[1], 1 - x[2]) + π / 2
    theta2 = atan(x[2], x[1]) + π
    theta3 = atan(1 - x[1], x[2]) + 3π / 2
    rho1 = θ -> x[1] / (-cos(θ))
    rho2 = θ -> x[2] / (-sin(θ))
    rho3 = θ -> (1 - x[1] - x[2]) / (sqrt(2) * cos(θ - π / 4))
    return (theta1, theta2, rho1), (theta2, theta3, rho2), (theta3, theta1 + 2π, rho3)
end

function guiggiani_singular_integral(
    K,
    û,
    x̂,
    el::ReferenceInterpolant{<:Union{ReferenceTriangle,ReferenceSquare}},
    quad_rho,
    quad_theta,
    sorder::Val{P} = Val(2),
) where {P}
    ref_shape = reference_domain(el)
    x         = el(x̂)
    nx        = normal(el, x̂)
    qx        = (coords = x, normal = nx)
    # function to integrate in polar coordinates
    F = (ρ, θ) -> begin
        s, c = sincos(θ)
        ŷ = x̂ + ρ * SVector(c, s)
        y = el(ŷ)
        jac = jacobian(el, ŷ)
        ny = _normal(jac)
        μ = _integration_measure(jac)
        qy = (coords = y, normal = ny)
        M = K(qx, qy)
        v = û(ŷ)
        return ρ * map(v -> M * v, v) * μ
    end
    acc = zero(return_type(F, Float64, Float64))
    # integrate
    for (theta_min, theta_max, rho_func) in polar_decomposition(ref_shape, x̂)
        delta_theta = theta_max - theta_min
        I_theta = quad_theta() do (theta_ref,)
            theta = theta_min + theta_ref * delta_theta
            rho_max = rho_func(theta)::Float64
            F₋₂, F₋₁, F₀ = laurent_coefficients(
                rho -> F(rho, theta),
                rho_max / 2,
                sorder;
                atol = 1e-10,
                rtol = 1e-8,
                contract = 1 / 2,
            )
            I_rho = quad_rho() do (rho_ref,)
                rho = rho_ref * rho_max
                rho < 1e-4 && (return F₀)
                # compute F(rho, theta) - F₋₁ / rho - F₋₂ / rho^2, but ignore terms that are
                # known to be zero
                if P == 2
                    return F(rho, theta) - F₋₁ / rho - F₋₂ / rho^2
                elseif P == 1
                    return F(rho, theta) - F₋₁ / rho
                else
                    return F(rho, theta)
                end
            end
            # compute I_rho * rho_max + F₋₁ * log(rho_max) - F₋₂ / rho_max but manually
            # ignore terms that are known to be zero
            if P == 2
                return I_rho * rho_max + F₋₁ * log(rho_max) - F₋₂ / rho_max
            elseif P == 1
                return I_rho * rho_max + F₋₁ * log(rho_max)
            else
                return I_rho * rho_max
            end
        end
        I_theta *= delta_theta
        acc += I_theta
    end
    return acc
end

function guiggiani_singular_integral(
    K,
    û,
    x̂,
    el::ReferenceInterpolant{ReferenceLine},
    quad_rho,
    quad_theta, # unused, but kept for consistency with the 2D case
    sorder::Val{P} = Val(2),
) where {P}
    x  = el(x̂)
    nx = normal(el, x̂)
    qx = (coords = x, normal = nx)
    # function to integrate in 1D "polar" coordinates. We use `s ∈ {-1,1}` to denote the
    # angles `π` and `0`.
    F = (ρ, s) -> begin
        ŷ = x̂ + SVector(ρ * s)
        y = el(ŷ)
        jac = jacobian(el, ŷ)
        ny = _normal(jac)
        μ = _integration_measure(jac)
        qy = (coords = y, normal = ny)
        M = K(qx, qy)
        v = û(ŷ)
        map(v -> M * v, v) * μ
    end
    acc = zero(return_type(F, Float64, Int))
    # integrate
    for (s, rho_max) in ((-1, x̂[1]), (1, 1 - x̂[1]))
        F₋₂, F₋₁, F₀ =
            F₋₂, F₋₁, F₀ = laurent_coefficients(
                rho -> F(rho, s),
                rho_max / 2,
                sorder;
                atol = 1e-10,
                rtol = 1e-8,
                contract = 1 / 2,
            )
        I_rho = quad_rho() do (rho_ref,)
            rho = rho_ref * rho_max
            rho < 1e-4 && (return F₀)
            if P == 2
                return F(rho, s) - F₋₂ / rho^2 - F₋₁ / rho
            elseif P == 1
                return F(rho, s) - F₋₁ / rho
            else
                return F(rho, s)
            end
        end
        if P == 2
            acc += (F₋₁ * log(rho_max) - F₋₂ / rho_max) + I_rho * rho_max
        elseif P == 1
            acc += F₋₁ * log(rho_max) + I_rho * rho_max
        else
            acc += I_rho * rho_max
        end
    end
    return acc
end

"""
    laurent_coefficients(f, h, order::Val{N}) where {N}

Given a one-dimensional function `f`, return `f₋₂, f₋₁, f₀` such that `f(x) = f₋₂ / x^2 +
f₋₁ / x + f₀ + 𝒪(x)` as `x -> 0`, where we assume that `f₋ₙ = 0` for `n > N`.
"""
function laurent_coefficients(f, h, order::Val{2}; kwargs...)
    g = x -> x^2 * f(x)
    f₋₂, e₋₂ = extrapolate(h; x0 = 0, kwargs...) do x
        return g(x)
    end
    f₋₁, e₋₁ = extrapolate(h; x0 = 0, kwargs...) do x
        return x * f(x) - f₋₂ / x
    end
    f₀, e₀ = extrapolate(h; x0 = 0, kwargs...) do x
        return f(x) - f₋₂ / x^2 - f₋₁ / x
    end
    return f₋₂, f₋₁, f₀
end
function laurent_coefficients(f, h, ::Val{1}; kwargs...)
    f₋₁, e₋₁ = extrapolate(h; x0 = 0, kwargs...) do x
        return x * f(x)
    end
    f₀, e₀ = extrapolate(h; x0 = 0, kwargs...) do x
        return f(x) - f₋₁ / x
    end
    return 0, f₋₁, f₀
end
function laurent_coefficients(f, h, ::Val{0}; kwargs...)
    f₀, e₀ = extrapolate(h; x0 = 0, kwargs...) do x
        return f(x)
    end
    return 0, 0, f₀
end
function laurent_coefficients(f, h, ::Val{N}; kwargs...) where {N}
    return error("laurent_coefficients: order $N not implemented")
end

"""
    local_correction_dist_and_tol(iop::IntegralOperator, kmax)

Try to estimate resonable `maxdist` and `tol` parameters for the [`local_correction`](@ref)
function, where `maxdist` is at most `kmax` times the radius of the largest element in the
source mesh of `iop`.

!!! note
    This is a heuristic and may not be accurate/efficient in all cases. It is recommended to
    test different values of `maxdist` and `tol` to find the optimal values for your
    problem.

# Extended help

The heuristic works as follows, where we let `K = kernel(iop)` and `msh =
mesh(source(iop))`:

1. Pick the largest element in `msh`
2. Let `h` be the radius of `el`
3. For `k` between `1` and `kmax`, estimate the quadrature
   error when integrating `y -> K(x,y)` for `x` at a distance `k * h` from
   the center of the element using a regular quadrature rule
4. Find a `k` such that ratio between errors at distances `k * h` and `(k + 1) * h` is below
   a certain threshold. This indicates stagnation in the error, and suggests that little is
   gained by increasing the distance.
5. Return `maxdist = k * h` and `tol` as the error at distance `k * h`.
"""
function local_correction_dist_and_tol(iop::IntegralOperator, kmax = 10)
    K       = kernel(iop)
    Q       = source(iop)
    msh     = mesh(Q)
    maxdist = 0.0
    tol     = 0.0
    for E in element_types(msh)
        ref_domain     = reference_domain(E)
        els            = elements(msh, E)
        regular_quad   = quadrature_rule(Q, E)
        reference_quad = adaptive_quadrature(ref_domain; atol = 1e-8, maxsubdiv = 10_000)
        # pick the biggest element as a reference
        qtags = etype2qtags(Q, E)
        a, i = @views findmax(j -> sum(weight, Q[qtags[:, j]]), 1:size(qtags, 2))
        dist, er =
            _regular_integration_errors(els[i], K, regular_quad, reference_quad, kmax)
        # find first index such that er[i+1] > er[i] / ratio
        ratio = 8
        i = findfirst(i -> er[i+1] > er[i] / ratio, 1:(kmax-1))
        isnothing(i) && (i = kmax)
        maxdist = max(maxdist, dist[i])
        tol     = max(tol, er[i])
    end
    return maxdist, tol
end

function _regular_integration_errors(el, K, qreg, qref, maxiter)
    x₀ = center(el) # center
    h = radius(el)  # reasonable scale
    f = (x, ŷ) -> begin
        y   = el(ŷ)
        jac = jacobian(el, ŷ)
        ν   = _normal(jac)
        νₓ  = (x - x₀) |> normalize
        τ′  = _integration_measure(jac)
        return K((coords = x, normal = νₓ), (coords = y, normal = ν)) * τ′
    end
    N = length(x₀)
    er = 0.0
    cc = 0
    ers = Float64[]
    dists = Float64[]
    while cc < maxiter
        cc += 1
        # explore a few directions and pick the worst error
        er = 0.0
        for dir in -N:N
            iszero(dir) && continue
            k  = abs(dir)
            x  = setindex(x₀, x₀[k] + sign(N) * cc * h, k)
            I  = qref(ŷ -> f(x, ŷ))
            Ia = qreg(ŷ -> f(x, ŷ))
            er = max(er, norm(Ia - I))
        end
        push!(ers, er)
        push!(dists, cc * h)
    end
    return dists, ers
end
