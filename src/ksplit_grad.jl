"""
    _laplace_double_layer_gradient_kernel(component)

Return the `component`-th Cartesian component (`1` for `x`, `2` for `y`) of the
2D Laplace double-layer gradient kernel.
"""
function _laplace_double_layer_gradient_kernel(component::Integer)
    component ∈ (1, 2) || throw(ArgumentError("component must be 1 or 2"))

    return (target, source) -> begin
        x = Inti.coords(target)
        y = Inti.coords(source)
        ny = Inti.normal(source)
        r = x - y
        d2 = dot(r, r)
        d2 ≤ Inti.SAME_POINT_TOLERANCE^2 && return 0.0
        rdny = dot(r, ny)
        return 1 / (2π) * (ny[component] / d2 - 2 * r[component] * rdny / d2^2)
    end
end

function _yukawa_double_layer_gradient_kernel(component::Integer, λ)
    component ∈ (1, 2) || throw(ArgumentError("component must be 1 or 2"))

    return (target, source) -> begin
        x = Inti.coords(target)
        y = Inti.coords(source)
        ny = Inti.normal(source)
        r = x - y
        d = norm(r)
        d ≤ Inti.SAME_POINT_TOLERANCE && return 0.0
        rdny = dot(r, ny)
        k1 = Bessels.besselk(1, λ * d)
        k0 = Bessels.besselk(0, λ * d)
        pref = λ / (2π) * k1 / d
        deriv_pref = -λ^2 / (2π) * k0 / d - λ / π * k1 / d^2
        return pref * ny[component] + (r[component] / d) * deriv_pref * rdny
    end
end

function _get_ksplit_gradient_kernels(op, component, T)
    component ∈ (1, 2) || throw(ArgumentError("component must be 1 or 2"))

    if op isa Inti.Laplace{2}
        proj_cauchy = component == 1 ? real : z -> -imag(z)
        proj_hyper = component == 1 ? z -> -imag(z) : z -> -real(z)

        return (
            G_L = (x, y) -> zero(T),
            G_C = (x, y) -> zero(T),
            G_H = (x, y) -> one(T) / (2π),
            proj_cauchy = proj_cauchy,
            proj_hyper = proj_hyper,
        )
    elseif op isa Inti.Yukawa{2}
        λ = op.λ
        proj_cauchy = component == 1 ? real : z -> -imag(z)
        proj_hyper = component == 1 ? z -> -imag(z) : z -> -real(z)

        G_log = (x, y) -> begin
            x_coor = Inti.coords(x)
            y_coor = Inti.coords(y)
            r = x_coor - y_coor
            d = norm(r)
            d ≤ Inti.SAME_POINT_TOLERANCE && return zero(T)
            ny = Inti.normal(y)
            rdny = dot(r, ny)
            a = λ / (2π) * besseli(1, λ * d) / d
            da = λ^2 / (2π) * besseli(0, λ * d) / d - λ / π * besseli(1, λ * d) / d^2
            return a * ny[component] + (r[component] / d) * da * rdny
        end

        G_cauchy = (x, y) -> begin
            x_coor = Inti.coords(x)
            y_coor = Inti.coords(y)
            r = x_coor - y_coor
            d = norm(r)
            d ≤ Inti.SAME_POINT_TOLERANCE && return zero(T)
            ny = Inti.normal(y)
            return λ / (2π) * besseli(1, λ * d) * dot(r, ny) / d
        end

        return (
            G_L = G_log,
            G_C = G_cauchy,
            G_H = (x, y) -> one(T) / (2π),
            proj_cauchy = proj_cauchy,
            proj_hyper = proj_hyper,
        )
    end

    error("Kernel-split double-layer gradients are currently implemented only for Inti.Laplace(; dim = 2) and Inti.Yukawa(; dim = 2)")
end

function _panel_parametric_interval(source_el, boundary_inv, panel_idx, parametric_length)
    t_a = boundary_inv(source_el[panel_idx](0))
    t_b = boundary_inv(source_el[panel_idx](1))
    if (t_b - t_a) < -parametric_length / 2
        t_b += parametric_length
    end
    return t_a, t_b
end

function _panel_complex_data(
        source_quad,
        source_el,
        boundary_inv,
        velocity_fn,
        panel_idx,
        n_quad_pt,
        w_leg_ref,
        parametric_length,
    )
    idx_global = (panel_idx - 1) * n_quad_pt
    panel_quad = source_quad[(idx_global + 1):(idx_global + n_quad_pt)]

    node_a = source_el[panel_idx](0)
    node_b = source_el[panel_idx](1)
    node_a_complex = node_a[1] + im * node_a[2]
    node_b_complex = node_b[1] + im * node_b[2]

    t_a, t_b = _panel_parametric_interval(source_el, boundary_inv, panel_idx, parametric_length)
    panel_nodes = Inti.coords.(panel_quad)
    panel_nodes_complex = [node[1] + im * node[2] for node in panel_nodes]
    panel_normals_complex = [Inti.normal(q)[1] + im * Inti.normal(q)[2] for q in panel_quad]
    panel_weights = Inti.weight.(panel_quad)
    panel_params = boundary_inv.(panel_nodes)
    velocity_weights = (t_b - t_a) / 2 .* velocity_fn.(panel_params) .* w_leg_ref
    conj_tangent = conj.(velocity_weights ./ abs.(velocity_weights))

    return (
        idx_global = idx_global,
        quad = panel_quad,
        node_a_complex = node_a_complex,
        node_b_complex = node_b_complex,
        nodes_complex = panel_nodes_complex,
        normals_complex = panel_normals_complex,
        weights = panel_weights,
        velocity_weights = velocity_weights,
        conj_tangent = conj_tangent,
    )
end

"""
    kernel_split_double_layer_gradient_correction(
        op,
        source_quad,
        source_quad_connectivity,
        source_el,
        velocity_fn,
        boundary_inv,
        Lop,
        target;
        component,
        maxdist = 0.1,
        target_location = nothing,
        parametric_length = 1.0,
    )

Construct the kernel-split near-field correction for the `component`-th
Cartesian component of the off-surface double-layer gradient.

Currently this implements the 2D Laplace and Yukawa cases, where the singular
correction is handled by `wLCHinit`.
"""
function kernel_split_double_layer_gradient_correction(
        op,
        source_quad::Quadrature,
        source_quad_connectivity,
        source_el,
        velocity_fn,
        boundary_inv,
        Lop,
        target;
        component,
        maxdist = 0.1,
        target_location = nothing,
        parametric_length = 1.0,
    )
    target === source_quad &&
        throw(ArgumentError("double-layer gradient kernel split is implemented only for off-surface targets"))
    isnothing(target_location) &&
        throw(ArgumentError("missing target_location field in correction"))

    T = eltype(Lop)
    kernels = _get_ksplit_gradient_kernels(op, component, T)
    G_L = kernels.G_L
    G_C = kernels.G_C
    G_H = kernels.G_H
    proj_cauchy = kernels.proj_cauchy
    proj_hyper = kernels.proj_hyper

    Is, Js, Ls = Int[], Int[], T[]

    n_target, n_source = size(Lop)
    n_quad_pt = size(source_quad_connectivity, 1)
    n_el = size(source_quad_connectivity, 2)
    _, w_Leg_ref = gausslegendre(n_quad_pt)

    dict_near = near_points_vec(target, source_quad; maxdist = maxdist)
    near_idx = first(values(dict_near))

    for panel_idx in 1:n_el
        near_targets = near_idx[panel_idx]
        isempty(near_targets) && continue

        panel = _panel_complex_data(
            source_quad,
            source_el,
            boundary_inv,
            velocity_fn,
            panel_idx,
            n_quad_pt,
            w_Leg_ref,
            parametric_length,
        )

        for i in near_targets
            target_node = Inti.coords(target[i])
            target_node_complex = target_node[1] + im * target_node[2]

            wcorrL, _, wcmpC_complex, wcmpH = wLCHinit(
                panel.node_a_complex,
                panel.node_b_complex,
                target_node_complex,
                panel.nodes_complex,
                panel.normals_complex,
                panel.velocity_weights,
                n_quad_pt;
                target_location = target_location,
            )

            for j in 1:n_quad_pt
                correction_term =
                    G_L(target[i], panel.quad[j]) * panel.weights[j] * wcorrL[j] +
                    proj_cauchy(
                        G_C(target[i], panel.quad[j]) *
                        panel.conj_tangent[j] *
                        wcmpC_complex[j],
                    ) +
                    proj_hyper(G_H(target[i], panel.quad[j]) * wcmpH[j])

                correction_term == 0 && continue
                push!(Is, i)
                push!(Js, panel.idx_global + j)
                push!(Ls, correction_term)
            end
        end
    end

    return sparse(Is, Js, Ls, n_target, n_source)
end

"""
    double_layer_gradient(; op, target, source::Quadrature, compression, correction)

Return the Cartesian gradient operators of the double-layer potential for
off-surface targets. The result is a tuple `(Dx, Dy)` where `Dx * σ` and
`Dy * σ` approximate `∂₁ D[σ]` and `∂₂ D[σ]`, respectively.

The current implementation supports the 2D Laplace and Yukawa operators, and
the kernel-split correction is available only for off-surface targets.
"""
function double_layer_gradient(;
        op,
        target,
        source::Quadrature,
        compression = (method = :none,),
        correction = (method = :none,),
    )
    target === source &&
        throw(ArgumentError("double_layer_gradient only supports off-surface targets"))
    (op isa Inti.Laplace{2} || op isa Inti.Yukawa{2}) ||
        throw(ArgumentError("double_layer_gradient currently supports only Inti.Laplace(; dim = 2) and Inti.Yukawa(; dim = 2)"))

    compression = _normalize_compression(compression, target, source)
    correction = merge((method = :none, parametric_length = 1.0), correction)

    compression.method ∈ (:none, :hmatrix) || throw(
        ArgumentError("double_layer_gradient supports only :none and :hmatrix compression"),
    )
    correction.method ∈ (:none, :ksplit) || throw(
        ArgumentError("double_layer_gradient supports only :none and :ksplit correction"),
    )

    if correction.method == :ksplit
        for key in (:connectivity, :elements, :velocity_fn, :boundary_inv, :maxdist, :target_location)
            haskey(correction, key) || throw(ArgumentError("missing correction.$key field for :ksplit"))
        end
    end

    gradient_kernel = if op isa Inti.Laplace{2}
        _laplace_double_layer_gradient_kernel
    else
        component -> _yukawa_double_layer_gradient_kernel(component, op.λ)
    end
    Dxop = IntegralOperator(gradient_kernel(1), target, source)
    Dyop = IntegralOperator(gradient_kernel(2), target, source)

    if compression.method == :hmatrix
        Dxmat = assemble_hmatrix(Dxop; rtol = compression.tol)
        Dymat = assemble_hmatrix(Dyop; rtol = compression.tol)
    else
        Dxmat = assemble_matrix(Dxop)
        Dymat = assemble_matrix(Dyop)
    end

    if correction.method == :none
        return Dxmat, Dymat
    end

    ksplit_kwargs = filter(
        kv -> kv[1] in (:maxdist, :target_location, :parametric_length),
        pairs(correction),
    )

    δDx = kernel_split_double_layer_gradient_correction(
        op,
        source,
        correction.connectivity,
        correction.elements,
        correction.velocity_fn,
        correction.boundary_inv,
        Dxop,
        target;
        component = 1,
        ksplit_kwargs...,
    )
    δDy = kernel_split_double_layer_gradient_correction(
        op,
        source,
        correction.connectivity,
        correction.elements,
        correction.velocity_fn,
        correction.boundary_inv,
        Dyop,
        target;
        component = 2,
        ksplit_kwargs...,
    )

    if compression.method == :hmatrix
        Dx = LinearMap(Dxmat) + LinearMap(δDx)
        Dy = LinearMap(Dymat) + LinearMap(δDy)
    else
        Dx = axpy!(true, δDx, Dxmat)
        Dy = axpy!(true, δDy, Dymat)
    end

    return Dx, Dy
end
