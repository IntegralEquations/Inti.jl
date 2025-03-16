#=
Implementation of the singular integration method of Guiggiani et al. (1992)
=#

@kwdef struct GuiggianiParameters
    radial_quadrature  = GaussLegendre(; order = 5)
    angular_quadrature = GaussLegendre(; order = 40)
end

function GuiggianiParameters(tol)
    segment = HAdaptiveIntegration.segment(0.0, 1.0)
    quad_func = (f) -> begin
        I, E = HAdaptiveIntegration.integrate(f, segment; atol = tol)
        if E > tol
            error("adaptive integration failed to converge: E = $E")
        end
        return I
    end
    quad_gauss = GaussLegendre(; order = 5)
    return GuiggianiParameters(quad_func, quad_func)
    # return GuiggianiParameters(quad_gauss, quad_func)
end

"""
    guiggiani_correction(iop::IntegralOperator; nearfield_distance, nearfield_qorder)

Routine for correcting the hypersingular operator for Laplace in 3D. Works by identifying (a
priori) the entries of the integral operator which need to be corrected. This is done based
on distance between target points and source elements, and the parameter
`nearfield_distance` controls the threshold for this. For strongly singular entries
corresponding to target points on the source element, we use Guiggiani's method to compute
the correction. For nearly singular entries, we use an oversampled quadrature rule of order
`nearfield_qorder`.
"""
function guiggiani_correction(
    iop::IntegralOperator;
    nearfield_distance,
    nearfield_qorder,
    p::GuiggianiParameters = GuiggianiParameters(1e-5),
)
    # unpack type-unstable fields in iop, allocate output, and dispatch
    X, Y, K = target(iop), source(iop), kernel(iop)
    @assert X === Y "source and target of integraloperator must coincide"
    # @assert K isa HyperSingularKernel{<:Any,Laplace{3}} "integral operator must be associated to Laplace's hypersingular kernel in 3D"
    dict_near = near_interaction_list(X, Y; tol = nearfield_distance)
    T = eltype(iop)
    msh = mesh(Y)
    correction = (I = Int[], J = Int[], V = T[])
    # create quadratures (since they are their own type, we need to create them here for a
    # type-stable inner loop)

    for E in element_types(msh)
        # dispatch on element type
        τ̂ = domain(E)
        nearlist = dict_near[E]
        iter = elements(msh, E)
        quads = (
            regular_quad   = quadrature_rule(Y, E),
            nearfield_quad = _qrule_for_reference_shape(τ̂, nearfield_qorder),
            radial_quad    = p.radial_quadrature,
            angular_quad   = p.angular_quadrature,
        )
        L = lagrange_basis(quads.regular_quad)
        _guiggiani_correction_etype!(
            correction,
            iter,
            quads,
            L,
            nearlist,
            X,
            Y,
            K,
            nearfield_distance,
            p,
        )
    end
    m, n = size(iop)
    return sparse(correction.I, correction.J, correction.V, m, n)
end

"""
    polar_decomposition(shape::ReferenceSquare, x̂::SVector{2,Float64})

Decompose the square `[0,1] × [0,1]` into four triangles, and return four tuples of the form
`θₛ, θₑ, ρ` where `θₛ` and `θₑ` are the initial and final angles of the triangle, and `ρ` is
the function that gives the distance from `x̂` to the border of the square in the direction
`θ`.
"""
function polar_decomposition(::ReferenceSquare, x::SVector{2,Float64})
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
    laurent_coefficients(f, h, order::Val{N}) where {N}

Compute the Laurent coefficients of a function `f` at the origin. The function `f` is
assumed to diverge as `1/ρ^N` at the origin, and this function returns the negative Laurent
coefficients associated to the diverging terms.
"""
function laurent_coefficients(f, h = 1, ::Val{N} = Val(2); kwargs...) where {N}
    N == 2 || throw(ArgumentError("only N=2 is supported"))
    g = x -> x^N * f(x)
    f₋₂, e₋₂ = extrapolate(h; x0 = 0, kwargs...) do x
        return g(x)
    end
    # f₋₁, e₋₁ = extrapolate(h; x0 = 0, kwargs...) do x
    #     return ForwardDiff.derivative(g, x)
    # end
    # f₀, e₀ = extrapolate(h; x0 = 0, kwargs...) do x
    #     return ForwardDiff.derivative(x -> ForwardDiff.derivative(g, x), x) / 2
    # end
    f₋₁, e₋₁ = extrapolate(h; x0 = 0, kwargs...) do x
        return x * f(x) - f₋₂ / x
    end
    f₀, e₀ = extrapolate(h; x0 = 0, kwargs...) do x
        return f(x) - f₋₂ / x^2 - f₋₁ / x
    end
    # @show e₋₂, e₋₁, e₀
    return f₋₂, f₋₁, f₀
end

function guiggiani_singular_integral(
    K,
    û,
    x̂,
    el::ReferenceInterpolant{<:Union{ReferenceTriangle,ReferenceSquare}},
    quad_rho,
    quad_theta,
)
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
                Val(2);
                atol = 1e-12,
                contract = 1 / 2,
            )
            I_rho = quad_rho() do (rho_ref,)
                rho = rho_ref * rho_max
                if rho < 1e-4
                    return F₀
                end
                return F(rho, theta) - F₋₂ / rho^2 - F₋₁ / rho
            end
            return (F₋₁ * log(rho_max) - F₋₂ / rho_max) + I_rho * rho_max
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
)
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
        F₋₂, F₋₁, F₀ = laurent_coefficients(rho -> F(rho, s), Val(2), 1e-2)
        I_rho = quad_rho() do (rho_ref,)
            rho = rho_ref * rho_max
            return F(rho, s) - F₋₂ / rho^2 - F₋₁ / rho
        end
        acc += (F₋₁ * log(rho_max) - F₋₂ / rho_max) + I_rho * rho_max
    end
    return acc
end

@noinline function _guiggiani_correction_etype!(
    correction,
    el_iter,
    quads,
    L,
    nearlist,
    X,
    Y,
    K,
    nearfield_distance,
    p,
)
    qreg = quads.regular_quad
    nearfield_quad = quads.nearfield_quad
    angular_quad = quads.angular_quad
    radial_quad = quads.radial_quad

    E = eltype(el_iter)
    Xqnodes = collect(X)
    Yqnodes = collect(Y)
    τ̂ = domain(E)

    # reference quadrature nodes and weights
    x̂ = qcoords(qreg) |> collect
    x̂_near = qcoords(nearfield_quad) |> collect
    ŵ_near = qweights(nearfield_quad) |> collect

    N = geometric_dimension(τ̂)
    el2qtags = etype2qtags(Y, E)
    for n in eachindex(el_iter)
        el = el_iter[n]
        jglob = view(el2qtags, :, n)
        inear = union(nearlist[n], jglob) # make sure to include nearfield nodes AND the element nodes
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
                    radial_quad,
                    angular_quad,
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
                W = integrate(integrand, x̂_near, ŵ_near)
            end
            for (k, j) in enumerate(jglob)
                qx, qy = Xqnodes[i], Yqnodes[j]
                push!(correction.I, i)
                push!(correction.J, j)
                push!(correction.V, W[k] - K(qx, qy) * weight(qy))
            end
        end
    end
    return correction
end
