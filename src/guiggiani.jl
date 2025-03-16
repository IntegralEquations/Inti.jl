#=
A generic routines for correcting the singular or nearly singular entries of an integral
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
    local_correction(iop::IntegralOperator; maxdist, tol)

Given an integral operator `iop`, this function provides a sparse correction to `iop` for
the entries `i,j` such that the distance between the `i`-th target and the `j`-th source is
less than `maxdist`. Each entry of the correction matrix is computed by using an adaptive
integration routine with an absolute tolerance `tol`.

Choosing `maxdist` and `tol` is a trade-off between accuracy and efficiency. The smaller the
value of `maxdist`, the fewer corrections are needed, but this may compromise the accuracy.
Similarly, the smaller the value of `tol`, the more accurate the correction, but the more
expensive the computation. The optimal values of `maxdist` and `tol` depend on the kernel
and the quadrature rule used.
"""
function local_correction(iop::IntegralOperator; maxdist, tol)
    msh = mesh(target(iop))
    quads_dict = Dict()
    for E in element_types(msh)
        ref_domain = reference_domain(E)
        quads = (
            nearfield_quad = adaptive_quadrature(ref_domain; atol = tol),
            radial_quad    = adaptive_quadrature(ReferenceLine(); atol = tol),
            angular_quad   = adaptive_quadrature(ReferenceLine(); atol = tol),
        )
        quads_dict[E] = quads
    end
    return local_correction(iop, maxdist, quads_dict)
end

"""
    local_correction(iop::IntegralOperator, maxdist, quads_dict::Dict)

Similar to `local_correction(iop::IntegralOperator; maxdist, tol)`, but allows the user to
pass a dictionary `quads_dict` with precomputed quadrature rules for each element type in
the mesh of `target(iop)`.

The dictionary `quads_dict` should have the following structure:
- `quads_dict[E].nearfield_quad` should be a function that integrates a function over the
  nearfield of the reference element `E`.
- `quads_dict[E].radial_quad` should be a function that integrates a function over the
  radial direction of the reference element `E`.
- `quads_dict[E].angular_quad` should be a function that integrates a function over the
  angular direction of the reference element `E`.
"""
function local_correction(iop, maxdist, quads_dict::Dict)
    # unpack type-unstable fields in iop, allocate output, and dispatch
    X, Y, K = target(iop), source(iop), kernel(iop)
    @assert X === Y "source and target of integraloperator must coincide"
    dict_near = near_interaction_list(X, Y; tol = maxdist)
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
    Threads.@threads for n in 1:nel
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
triangle in the direction `θ`
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
    return f₋₁, f₀, 0
end
function laurent_coefficients(f, h, ::Val{0}; kwargs...)
    f₀, e₀ = extrapolate(h; x0 = 0, kwargs...) do x
        return f(x)
    end
    return f₀, 0, 0
end
