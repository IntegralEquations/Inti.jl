include("ksplit_utils.jl")

"""
    kernel_split_correction(
    op,
    source_quad::Quadrature,
    source_quad_connectivity,
    source_el,
    velocity_fn,
    curvature_fn,
    boundary_inv,
    Lop,
    target=nothing;
    kwargs...
    )
)

Given an operator `op` and a Gauss Legendre Nyström discretization `Lop`
defined on a source quadrature `source_quad`, compute a correction `δL` such
that `Lop + δL` is a more accurate approximation of the underlying integral
operator.

This function implements a high-order quadrature correction based on variants
of Helsing's kernel split method. It explicitly splits the kernel `G` into a
smooth part, a log-singular part, and a Cauchy-singular part `G_C`:

G(x, y) = G_S(x, y) + G_L(x, y) log|x - y| + G_C(x, y) frac{(x - y) ⋅ n(y)}{|x - y|^2}

and uses product integration to integrate each singular part analytically.

See also [`helsing2008evaluation`](@ref) and [`helsing2015variants`](@ref).

A good rule of thumb for choosing the mesh size `h` relative to the wavenumber `k`
for a Helmholtz operator when a 16-point Gauss Legendre quadrature is `h ≈ π / k`.
This choice gives about 12 digits of accuracy for interior Dirichlet problems on 
a circle of radius 1. For more complicated geometries, a finer mesh may be needed.

# Arguments

## Required:

- `op`: Must be an [`AbstractDifferentialOperator`](@ref) (e.g., `Inti.Laplace`).
- `source_quad`: A [`Quadrature`](@ref) object for the source boundary `Y`.
- `source_quad_connectivity`: Connectivity matrix for `source_quad` points.
- `source_el`: A vector of panel parametrization functions for `source_quad`.
- `velocity_fn`: Function `t -> SVector` for the boundary's parametric velocity `v(t)`.
- `curvature_fn`: Function `t -> Float64` for the boundary's parametric signed curvature `κ(t)`.
- `boundary_inv`: Function `x -> t` mapping physical points `x` to parameter `t`.
- `Lop`: Approximated integral operators (e.g., from `Inti.assemble_operator`) to be corrected.

## Optional:

- `target=nothing`: The target points `X`. If `nothing` or `=== source_quad`,
  computes the on-surface correction. Otherwise, computes the off-surface
  correction for near-field interactions.
- `n_panel_corr=3`: The total number of panels (self + neighbors) in the
   fixed neighborhood of the source panel for the on-surface correction. 
   Must be an odd integer. The default provides a good balance between accuracy
   and computational cost and works the majority of the time.
- `maxdist=0.1`: distance beyond which interactions are considered sufficiently far
  so that no correction is needed. This is used to determine a threshold for
  nearly-singular corrections when `X` and `Y` are different domains. When `X
  === Y`, this is not needed.
- `target_location=nothing`: Passed to `wLCinit` for off-surface correction
  (e.g., `:inside`, `:outside`).
- `layer_type=:single`: The layer potential type (`:single` or `:double`).
- `parametric_length=1.0`: The total length of the parametric domain (e.g.,
  `1.0` or `2π`).
"""

function kernel_split_correction(
        op,
        source_quad::Quadrature,
        source_quad_connectivity,
        source_el,
        velocity_fn,
        curvature_fn,
        boundary_inv,
        Lop,
        target = nothing;
        n_panel_corr = 3,
        maxdist = 0.1,
        target_location = nothing,
        layer_type = :single,
        parametric_length = 1.0,
    )
    if isnothing(target)
        target = source_quad
    end
    speed_fn(θ) = norm(velocity_fn(θ))
    T = eltype(Lop) # element type of the operator

    # set coefficient functions for the kernel split
    kernels = _get_ksplit_kernels(op, layer_type, T)
    G_S = kernels.G_S
    G_L = kernels.G_L
    G_C = kernels.G_C

    Is, Js, Ls = Int[], Int[], T[]

    n_quad_pt = size(source_quad_connectivity)[1]
    n_el = size(source_quad_connectivity)[2]
    t_Leg_ref, w_Leg_ref = gausslegendre(n_quad_pt)

    if target == source_quad
        # define the weight matrix for the product integrations
        n = size(Lop)[1]

        # Build the connectivity map to handle randomly ordered panels
        panel_to_nodes_map, node_to_panel_map = build_neighbor_information(source_el, n_el)

        # define the correction distance
        n_panel_corr_dist = round(Int, (n_panel_corr - 1) / 2)

        # Pre-compute parametric intervals and sizes for all panels
        panel_intervals = Vector{Tuple{Float64, Float64}}(undef, n_el)
        for i in 1:n_el
            node_a = source_el[i](0)
            node_b = source_el[i](1)
            t_a = boundary_inv(node_a)
            t_b = boundary_inv(node_b)
            if (t_b - t_a) < -parametric_length / 2 # Handle branch cut
                t_b += parametric_length
            end
            panel_intervals[i] = (t_a, t_b)
        end

        # loop over each element to compute the correction δL
        for j_el in 1:n_el
            j_el_vec = OffsetArray(
                Vector{Int}(undef, 2 * n_panel_corr_dist + 1),
                -n_panel_corr_dist:n_panel_corr_dist,
            )
            j_el_vec[0] = j_el

            # Find neighbors in the "forward" direction
            current_panel_idx = j_el
            (start_node_id, end_node_id) = panel_to_nodes_map[current_panel_idx]
            current_node_id = end_node_id
            for k in 1:n_panel_corr_dist
                connected_panels = node_to_panel_map[current_node_id]
                next_panel_idx = if (connected_panels[1] == current_panel_idx)
                    connected_panels[2]
                else
                    connected_panels[1]
                end
                j_el_vec[k] = next_panel_idx

                # Update for the next iteration
                current_panel_idx = next_panel_idx
                (next_start_id, next_end_id) = panel_to_nodes_map[current_panel_idx]
                current_node_id =
                    (next_start_id == current_node_id) ? next_end_id : next_start_id
            end

            # Find neighbors in the "backward" direction
            current_panel_idx = j_el
            (start_node_id, end_node_id) = panel_to_nodes_map[current_panel_idx]
            current_node_id = start_node_id
            for k in -1:-1:-n_panel_corr_dist
                connected_panels = node_to_panel_map[current_node_id]
                next_panel_idx = if (connected_panels[1] == current_panel_idx)
                    connected_panels[2]
                else
                    connected_panels[1]
                end
                j_el_vec[k] = next_panel_idx

                # Update for the next iteration
                current_panel_idx = next_panel_idx
                (next_start_id, next_end_id) = panel_to_nodes_map[current_panel_idx]
                current_node_id =
                    (next_start_id == current_node_id) ? next_end_id : next_start_id
            end

            idx_global_corr_vec = (j_el_vec .- 1) * n_quad_pt

            # Get source panel j_el info
            (t_a_j, t_b_j) = panel_intervals[j_el]
            H_j = (t_b_j - t_a_j) / 2 # Source panel half-length

            # The calculation for W_L_c_corr (the k=0 case) is based on a standard interval
            𝔚_L_diag = WfrakLinit(0, 1, t_Leg_ref, n_quad_pt)
            W_L_c_corr = zeros(n_quad_pt, n_quad_pt)
            for i in 1:n_quad_pt, j in 1:n_quad_pt
                W_L_c_corr[i, j] =
                    𝔚_L_diag[i, j] / w_Leg_ref[j] -
                    (i != j ? log(abs(t_Leg_ref[i] - t_Leg_ref[j])) : 0)
            end

            # Add extra singular correction for diagonal elements
            for i in 1:n_quad_pt
                quad_i = source_quad[idx_global_corr_vec[0] + i]
                t_i = boundary_inv(Inti.coords(quad_i))
                W_L_c_corr[i, i] += log(H_j * speed_fn(t_i))
                push!(Is, idx_global_corr_vec[0] + i)
                push!(Js, idx_global_corr_vec[0] + i)
                push!(
                    Ls,
                    G_C(quad_i, quad_i) * Inti.weight(quad_i) * (-curvature_fn(t_i) / 2),
                )
            end

            # Loop over neighbors and compute corrections dynamically
            for k in -n_panel_corr_dist:n_panel_corr_dist
                j_k = j_el_vec[k] # This is the target panel index

                local W_L_matrix
                if k == 0
                    W_L_matrix = W_L_c_corr
                else
                    # Get target panel j_k info
                    (t_a_k, t_b_k) = panel_intervals[j_k]
                    C_k = (t_a_k + t_b_k) / 2
                    H_k = (t_b_k - t_a_k) / 2

                    # Center of source panel
                    C_j = (t_a_j + t_b_j) / 2

                    # Correct distance calculation for periodic domain ---
                    dist = C_k - C_j
                    if dist > parametric_length / 2
                        dist -= parametric_length
                    elseif dist < -parametric_length / 2
                        dist += parametric_length
                    end
                    trans = dist / H_j
                    scale = H_k / H_j

                    𝔚_L_offdiag = WfrakLinit(trans, scale, t_Leg_ref, n_quad_pt)
                    W_L_matrix = zeros(n_quad_pt, n_quad_pt)

                    t_target_in_source_frame = trans .+ scale .* t_Leg_ref
                    for i in 1:n_quad_pt, j in 1:n_quad_pt
                        W_L_matrix[i, j] =
                            𝔚_L_offdiag[i, j] / w_Leg_ref[j] -
                            log(abs(t_target_in_source_frame[i] - t_Leg_ref[j]))
                    end
                end

                # Apply the correction using the dynamically computed W_L_matrix
                for i in 1:n_quad_pt
                    quad_i = source_quad[idx_global_corr_vec[k] + i]
                    for j in 1:n_quad_pt
                        quad_j = source_quad[idx_global_corr_vec[0] + j]
                        quad_weight_j = Inti.weight(quad_j)

                        correction_term =
                            G_L(quad_i, quad_j) * quad_weight_j * W_L_matrix[i, j]
                        if k == 0 && i == j
                            correction_term += G_S(quad_i, quad_i) * Inti.weight(quad_i)
                        end

                        push!(Is, idx_global_corr_vec[k] + i)
                        push!(Js, idx_global_corr_vec[0] + j)
                        push!(Ls, correction_term)
                    end
                end
            end
        end
        δL = sparse(Is, Js, Ls, n, n)
    else # target != source
        n_target = length(target)
        n_source = length(source_quad)

        dict_near = near_points_vec(target, source_quad; maxdist = maxdist)
        near_idx = first(dict_near)[2]

        for j_el in 1:n_el
            idx_global_corr = (j_el - 1) * n_quad_pt
            for i in near_idx[j_el]
                target_node_i = Inti.coords(target[i])
                target_node_i_complex = target_node_i[1] + im * target_node_i[2]

                node_a = source_el[j_el](0)
                node_b = source_el[j_el](1)
                node_a_complex = node_a[1] + im * node_a[2]
                node_b_complex = node_b[1] + im * node_b[2]
                t_a = boundary_inv(node_a)
                t_b = boundary_inv(node_b)

                quad_j_el = source_quad[(idx_global_corr + 1):(idx_global_corr + n_quad_pt)]

                quad_node_j_el = Inti.coords.(quad_j_el)
                quad_node_j_el_complex =
                    [quad_node_j[1] + im * quad_node_j[2] for quad_node_j in quad_node_j_el]
                quad_normal_j_el = Inti.normal.(quad_j_el)
                quad_normal_j_el_complex = [
                    quad_normal_j[1] + im * quad_normal_j[2] for
                        quad_normal_j in quad_normal_j_el
                ]
                quad_weight_j_el = Inti.weight.(quad_j_el)

                # tj = real(boundary_inv.(quad_node_j_el)) # this might be unnecessary
                tj = boundary_inv.(quad_node_j_el)

                # check for branch cut for the boundary_inv function and correct it if necessary
                if (t_b - t_a) < -parametric_length / 2
                    t_b += parametric_length
                end

                quad_velocity_weight_j_el = (t_b - t_a) / 2 .* velocity_fn.(tj) .* w_Leg_ref

                wcorrL, wcmpC = wLCinit(
                    node_a_complex,
                    node_b_complex,
                    target_node_i_complex,
                    quad_node_j_el_complex,
                    quad_normal_j_el_complex,
                    quad_velocity_weight_j_el,
                    n_quad_pt;
                    target_location = target_location,
                )

                ra, rb, r, rj, nuj, rpwj, npt = node_a_complex,
                    node_b_complex,
                    target_node_i_complex,
                    quad_node_j_el_complex,
                    quad_normal_j_el_complex,
                    quad_velocity_weight_j_el,
                    n_quad_pt

                # loop over each quadrature point to compute the correction δL
                for j in 1:n_quad_pt
                    push!(Is, i)
                    push!(Js, idx_global_corr + j)
                    push!(
                        Ls,
                        G_L(target[i], quad_j_el[j]) * quad_weight_j_el[j] * wcorrL[j] +
                            G_C(target[i], quad_j_el[j]) * wcmpC[j],
                    )
                end
            end
        end
        δL = sparse(Is, Js, Ls, n_target, n_source)
    end

    return δL
end

"""
    adaptive_kernel_split_correction(
    op,
    source_quad::Quadrature,
    source_quad_connectivity,
    source_el,
    velocity_fn,
    curvature_fn,
    boundary_inv,
    Lop,
    target=nothing;
    kwargs...
    )
)

Given an operator `op` and a Gauss Legendre Nyström discretization `Lop`
defined on a source quadrature `source_quad`, compute a correction `δL` such
that `Lop + δL` is a more accurate approximation of the underlying integral
operator.

This function implements an adaptive version of the Helsing's kernel split method
by Fryklund et al, specifically for a class of modified PDEs, such as the modified 
Helmholtz (Yukawa) equation. It uses per-target adaptive sampling of the source geometry.
The refinement is carried out through recursive bisection, maintaining accuracy for 
a wide range of the parameter α.

See also [`fryklund2022adaptive`](@ref).

# Arguments

## Required:

- `op`: Must be an [`AbstractDifferentialOperator`](@ref) (e.g., `Inti.Laplace`).
- `source_quad`: A [`Quadrature`](@ref) object for the source boundary `Y`.
- `source_quad_connectivity`: Connectivity matrix for `source_quad` points.
- `source_el`: A vector of panel parametrization functions for `source_quad`.
- `velocity_fn`: Function `t -> SVector` for the boundary's parametric velocity `v(t)`.
- `curvature_fn`: Function `t -> Float64` for the boundary's parametric signed curvature `κ(t)`.
- `boundary_inv`: Function `x -> t` mapping physical points `x` to parameter `t`.
- `Lop`: Approximated integral operators (e.g., from `Inti.assemble_operator`) to be corrected.

## Optional:

- `target=nothing`: The target points `X`. If `nothing` or `=== source_quad`,
  computes the on-surface correction. Otherwise, computes the off-surface
  correction for near-field interactions.
- `n_panel_corr=3`: The total number of panels (self + neighbors) in the
   fixed neighborhood of the source panel for the on-surface correction. 
   Must be an odd integer. The default provides a good balance between accuracy
   and computational cost and works the majority of the time.
- `maxdist=0.1`: distance beyond which interactions are considered sufficiently far
  so that no correction is needed. This is used to determine a threshold for
  nearly-singular corrections when `X` and `Y` are different domains. When `X
  === Y`, this is not needed.
- `target_location=nothing`: Passed to `wLCinit` for off-surface correction
  (e.g., `:inside`, `:outside`).
- `layer_type=:single`: The layer potential type (`:single` or `:double`).
- `Cε=3.5`: The accuracy constant for the kernel-split method, used in the subinterval
   length criterion: `Δt ≤ 2*Cε / (α*h)`. It limits the product `α*h` to ensure that
   the smooth, but exponentially-growing, coefficient functions (e.g., `G_L` from the
   kernel split) can be accurately approximated by polynomials. The default value works
   well in practice. See [`fryklund2022adaptive`](@ref) for details.
- `Rε=3.7`: The cutoff radius for the "near evaluation criterion", which determines
   whether to use kernel-split quadrature or standard Gauss Legendre quadrature. The
   default value works well in practice. See [`fryklund2022adaptive`](@ref) for details.
- `parametric_length=1.0`: The total length of the parametric domain (e.g.,
  `1.0` or `2π`).
"""

function adaptive_kernel_split_correction(
        op,
        source_quad,
        source_quad_connectivity,
        source_el,
        velocity_fn,
        curvature_fn,
        boundary_inv,
        Lop,
        target = nothing;
        n_panel_corr = 3,
        maxdist = 0.1,
        target_location = nothing,
        layer_type = :single,
        Cε = 3.5,
        Rε = 3.7,
        parametric_length = 1.0,
        affine_preimage::Bool = true,
    )
    if isnothing(target)
        target = source_quad
    end
    speed_fn(θ) = norm(velocity_fn(θ))
    T = eltype(Lop) # element type of the operator

    if layer_type == :single
        G = Inti.SingleLayerKernel(op)
    elseif layer_type == :double
        G = Inti.DoubleLayerKernel(op)
    else
        throw("Unsupported layer type: $layer_type")
    end

    # set coefficient functions for the kernel split
    kernels = _get_ksplit_kernels(op, layer_type, T)
    G_S = kernels.G_S
    G_L = kernels.G_L
    G_C = kernels.G_C

    Is, Js, Ls = Int[], Int[], T[]

    # define parameters for the adaptive kernel split
    n_quad_pt = size(source_quad_connectivity)[1]
    n_el = size(source_quad_connectivity)[2]
    t_Leg_ref, w_Leg_ref = gausslegendre(n_quad_pt)
    L_Leg = L_Legendre_matrix(n_quad_pt)
    w_bary = LegendreBaryWeights(n_quad_pt)

    # FIXME: placeholder for Laplace
    if op isa Inti.Laplace{2}
        α = 5π
        @warn(
            "Using placeholder value α = 5π for Laplace operator in adaptive kernel split",
        )
    elseif op isa Inti.Helmholtz{2, Float64}
        α = op.k
        @warn("Adaptive kernel split does not provide computational advantages for Helmholtz 
        operators. Consider using the standard kernel split with fixed number of points 
        per wavelength instead.")
    elseif op isa Inti.Yukawa{2, Float64}
        α = op.λ
    else
        throw("Operator type not supported for adaptive kernel split")
    end

    if target == source_quad
        n_source = n_target = size(Lop)[1] # total number of dofs

        # define the correction distance
        n_panel_corr_dist = round(Int, (n_panel_corr - 1) / 2)

        # Build connectivity maps and pre-compute panel intervals
        panel_to_nodes_map, node_to_panel_map = build_neighbor_information(source_el, n_el)

        panel_intervals = Vector{Tuple{Float64, Float64}}(undef, n_el)
        for i in 1:n_el
            t_a = boundary_inv(source_el[i](0))
            t_b = boundary_inv(source_el[i](1))
            if (t_b - t_a) < -parametric_length / 2
                t_b += parametric_length
            end
            panel_intervals[i] = (t_a, t_b)
        end

        # loop over each element to compute the correction δL
        for j_el in 1:n_el

            # Find neighbors using connectivity map
            j_el_vec = OffsetArray(
                Vector{Int}(undef, 2 * n_panel_corr_dist + 1),
                -n_panel_corr_dist:n_panel_corr_dist,
            )
            j_el_vec[0] = j_el
            # Find neighbors in the "forward" direction
            current_panel_idx = j_el
            (start_node_id, end_node_id) = panel_to_nodes_map[current_panel_idx]
            current_node_id = end_node_id
            for k_fwd in 1:n_panel_corr_dist
                connected_panels = node_to_panel_map[current_node_id]
                next_panel_idx = if (connected_panels[1] == current_panel_idx)
                    connected_panels[2]
                else
                    connected_panels[1]
                end
                j_el_vec[k_fwd] = next_panel_idx
                current_panel_idx = next_panel_idx
                (next_start_id, next_end_id) = panel_to_nodes_map[current_panel_idx]
                current_node_id =
                    (next_start_id == current_node_id) ? next_end_id : next_start_id
            end
            # Find neighbors in the "backward" direction
            current_panel_idx = j_el
            (start_node_id, end_node_id) = panel_to_nodes_map[current_panel_idx]
            current_node_id = start_node_id
            for k_bwd in -1:-1:-n_panel_corr_dist
                connected_panels = node_to_panel_map[current_node_id]
                next_panel_idx = if (connected_panels[1] == current_panel_idx)
                    connected_panels[2]
                else
                    connected_panels[1]
                end
                j_el_vec[k_bwd] = next_panel_idx
                current_panel_idx = next_panel_idx
                (next_start_id, next_end_id) = panel_to_nodes_map[current_panel_idx]
                current_node_id =
                    (next_start_id == current_node_id) ? next_end_id : next_start_id
            end

            idx_global_corr_vec = (j_el_vec .- 1) * n_quad_pt

            # Get source panel info
            (t_a_j, t_b_j) = panel_intervals[j_el]
            C_j = (t_a_j + t_b_j) / 2
            H_j = (t_b_j - t_a_j) / 2

            # This must be defined here, as it's specific to the source panel `j_el`
            panel_parametrization = t -> SVector(source_el[j_el](scale_fn(0, 1, t)))

            # Compute subdivisions for all neighbors relative to the current source panel `j_el`
            panel_subdivision_vec = OffsetArray(
                Vector{Vector{Vector{Tuple}}}(undef, n_panel_corr),
                -n_panel_corr_dist:n_panel_corr_dist,
            )

            ksplit_bool_vec = OffsetArray(
                Vector{Vector{Vector{Bool}}}(undef, n_panel_corr),
                -n_panel_corr_dist:n_panel_corr_dist,
            )

            for k in -n_panel_corr_dist:n_panel_corr_dist
                panel_subdivision_vec[k], ksplit_bool_vec[k] = adaptive_refinement(
                    source_quad[(idx_global_corr_vec[0] + 1):(idx_global_corr_vec[0] + n_quad_pt)],
                    source_el[j_el],
                    L_Leg,
                    α,
                    Cε,
                    Rε,
                    target[(idx_global_corr_vec[k] + 1):(idx_global_corr_vec[k] + n_quad_pt)],
                    affine_preimage = affine_preimage,
                )
            end

            if isodd(n_quad_pt)
                throw("n_quad_pt = odd case not implemented")
            end

            # compute the local correction matrix δL_local_vec
            for k in -n_panel_corr_dist:n_panel_corr_dist
                j_k = j_el_vec[k] # Target panel index

                # Get target panel info and compute trans and scale
                (t_a_k, t_b_k) = panel_intervals[j_k]
                C_k = (t_a_k + t_b_k) / 2
                H_k = (t_b_k - t_a_k) / 2

                dist = C_k - C_j
                if dist > parametric_length / 2
                    dist -= parametric_length
                elseif dist < -parametric_length / 2
                    dist += parametric_length
                end
                trans = dist / H_j
                scale = H_k / H_j

                for i in 1:n_quad_pt

                    quad_i = target[idx_global_corr_vec[k] + i]
                    t_i = boundary_inv(Inti.coords(quad_i))
                    M_sub = length(panel_subdivision_vec[k][i])

                    # No subdivision but kernel split needed
                    if M_sub == 1 && ksplit_bool_vec[k][i][1] == true
                        # Determine trans and scale for the current panel pair (j_el, j_k)
                        local trans, scale
                        if k == 0
                            trans = 0.0
                            scale = 1.0
                        else
                            (t_a_k, t_b_k) = panel_intervals[j_k]
                            C_k = (t_a_k + t_b_k) / 2
                            H_k = (t_b_k - t_a_k) / 2
                            dist = C_k - C_j
                            if dist > parametric_length / 2
                                dist -= parametric_length
                            end
                            if dist < -parametric_length / 2
                                dist += parametric_length
                            end
                            trans = dist / H_j
                            scale = H_k / H_j
                        end

                        # Compute W_L_matrix with a single, conditional log term
                        𝔚_L_matrix = WfrakLinit(trans, scale, t_Leg_ref, n_quad_pt)
                        W_L_matrix = zeros(n_quad_pt, n_quad_pt)
                        t_target_in_source_frame = trans .+ scale .* t_Leg_ref

                        for i_w in 1:n_quad_pt, j_w in 1:n_quad_pt
                            log_arg = abs(t_target_in_source_frame[i_w] - t_Leg_ref[j_w])
                            # The log is only singular when k=0 and i_w=j_w.
                            log_term = (k == 0 && i_w == j_w) ? 0.0 : log(log_arg)
                            W_L_matrix[i_w, j_w] =
                                𝔚_L_matrix[i_w, j_w] / w_Leg_ref[j_w] - log_term
                        end

                        if k == 0
                            W_L_matrix[i, i] += log(H_j * speed_fn(t_i))
                            push!(Is, idx_global_corr_vec[0] + i)
                            push!(Js, idx_global_corr_vec[0] + i)
                            push!(
                                Ls,
                                G_C(quad_i, quad_i) *
                                    Inti.weight(quad_i) *
                                    (-curvature_fn(t_i) / 2),
                            )
                        end

                        for j in 1:n_quad_pt
                            quad_j = source_quad[idx_global_corr_vec[0] + j]
                            correction_term =
                                G_L(quad_i, quad_j) * Inti.weight(quad_j) * W_L_matrix[i, j]
                            if k == 0 && i == j
                                correction_term += G_S(quad_i, quad_i) * Inti.weight(quad_i)
                            end
                            push!(Is, idx_global_corr_vec[k] + i)
                            push!(Js, idx_global_corr_vec[0] + j)
                            push!(Ls, correction_term)
                        end

                    else # subdivision needed
                        # subtract regular coarse quadrature
                        for j in 1:n_quad_pt
                            quad_j = source_quad[idx_global_corr_vec[0] + j]
                            push!(Is, idx_global_corr_vec[k] + i)
                            push!(Js, idx_global_corr_vec[0] + j)
                            push!(Ls, -G(quad_i, quad_j) * Inti.weight(quad_j))
                        end

                        for k_sub in 1:M_sub
                            subpanel_node1, subpanel_node2 =
                                panel_subdivision_vec[k][i][k_sub]

                            t_Leg_sub = scale_fn(subpanel_node1, subpanel_node2, t_Leg_ref)

                            panel_curve = Inti.parametric_curve(
                                panel_parametrization_fn(source_el[j_el]),
                                subpanel_node1,
                                subpanel_node2,
                            )

                            panel_msh = Inti.meshgen(
                                panel_curve;
                                meshsize = (subpanel_node2 - subpanel_node1),
                            )
                            panel_quad =
                                Inti.Quadrature(panel_msh, Inti.GaussLegendre(n_quad_pt))

                            # Add regular refined quadrature contribution
                            for ĵ in 1:n_quad_pt
                                quad_ĵ = panel_quad[ĵ]
                                quad_weight_ĵ = Inti.weight(quad_ĵ)
                                bary_coef = Barycentric_coef(
                                    t_Leg_sub[ĵ];
                                    n = n_quad_pt,
                                    t_interp = t_Leg_ref,
                                    w = w_bary,
                                )
                                for j in 1:n_quad_pt
                                    push!(Is, idx_global_corr_vec[k] + i)
                                    push!(Js, idx_global_corr_vec[0] + j)
                                    push!(
                                        Ls,
                                        G(quad_i, quad_ĵ) * quad_weight_ĵ * bary_coef[j],
                                    )
                                end
                            end

                            # Add kernel split correction
                            if ksplit_bool_vec[k][i][k_sub] == 1
                                # The generalized target_node calculation is correct and remains.
                                dist_ti = t_i - C_j
                                if dist_ti > parametric_length / 2
                                    dist_ti -= parametric_length
                                end
                                if dist_ti < -parametric_length / 2
                                    dist_ti += parametric_length
                                end
                                τ_i = dist_ti / H_j

                                target_node =
                                    (2 * τ_i - (subpanel_node1 + subpanel_node2)) /
                                    (subpanel_node2 - subpanel_node1)

                                𝔚_L_vec = WfrakLinit(target_node, t_Leg_ref, n_quad_pt)
                                W_L_vec = zeros(n_quad_pt)
                                for j_w in 1:n_quad_pt
                                    W_L_vec[j_w] =
                                        𝔚_L_vec[j_w] / w_Leg_ref[j_w] -
                                        log(abs(target_node - t_Leg_ref[j_w]))
                                end

                                # Use the robust panel_quad object for the correction sum
                                for ĵ in 1:n_quad_pt
                                    quad_ĵ = panel_quad[ĵ]
                                    quad_weight_ĵ = Inti.weight(quad_ĵ)
                                    bary_coef = Barycentric_coef(
                                        t_Leg_sub[ĵ];
                                        n = n_quad_pt,
                                        t_interp = t_Leg_ref,
                                        w = w_bary,
                                    )
                                    correction_term =
                                        G_L(quad_i, quad_ĵ) * quad_weight_ĵ * W_L_vec[ĵ]
                                    for j in 1:n_quad_pt
                                        push!(Is, idx_global_corr_vec[k] + i)
                                        push!(Js, idx_global_corr_vec[0] + j)
                                        push!(Ls, correction_term * bary_coef[j])
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    else # target != source
        n_target = length(target)
        n_source = length(source_quad)

        # Pre-compute parametric intervals for all source panels
        panel_intervals = Vector{Tuple{Float64, Float64}}(undef, n_el)
        for i in 1:n_el
            t_a_i = boundary_inv(source_el[i](0))
            t_b_i = boundary_inv(source_el[i](1))
            if (t_b_i - t_a_i) < -parametric_length / 2
                t_b_i += parametric_length
            end
            panel_intervals[i] = (t_a_i, t_b_i)
        end

        dict_near = near_points_vec(target, source_quad; maxdist = maxdist)
        near_idx = first(dict_near)[2]

        for j_el in 1:n_el
            idx_global_corr = (j_el - 1) * n_quad_pt
            panel_subdivision_vec, ksplit_bool_vec = adaptive_refinement(
                source_quad[(idx_global_corr + 1):(idx_global_corr + n_quad_pt)],
                source_el[j_el],
                L_Leg,
                α,
                Cε,
                Rε,
                target[near_idx[j_el]],
                affine_preimage = affine_preimage,
            )

            # Get pre-computed, corrected panel information
            (t_a, t_b) = panel_intervals[j_el]
            node_a = source_el[j_el](0)
            node_a_complex = node_a[1] + im * node_a[2]
            node_b = source_el[j_el](1)
            node_b_complex = node_b[1] + im * node_b[2]
            panel_parametrization = panel_parametrization_fn(source_el[j_el])

            n_near_pt = length(near_idx[j_el])
            for i in 1:n_near_pt # Loop over each nearby target point
                near_idx_i = near_idx[j_el][i]
                target_i = target[near_idx_i]
                target_node_i_complex =
                    Inti.coords(target_i)[1] + im * Inti.coords(target_i)[2]

                M_sub = length(panel_subdivision_vec[i])

                if M_sub == 1 && ksplit_bool_vec[i][1] == true # No subdivision, but kernel split
                    quad_j_el = source_quad[(idx_global_corr + 1):(idx_global_corr + n_quad_pt)]
                    quad_node_j_el = Inti.coords.(quad_j_el)
                    quad_node_j_el_complex = [q[1] + im * q[2] for q in quad_node_j_el]
                    quad_normal_j_el_complex =
                        [Inti.normal(q)[1] + im * Inti.normal(q)[2] for q in quad_j_el]

                    tj = boundary_inv.(quad_node_j_el)
                    quad_velocity_weight_j_el =
                        (t_b - t_a) / 2 .* velocity_fn.(tj) .* w_Leg_ref

                    wcorrL, wcmpC = wLCinit(
                        node_a_complex,
                        node_b_complex,
                        target_node_i_complex,
                        quad_node_j_el_complex,
                        quad_normal_j_el_complex,
                        quad_velocity_weight_j_el,
                        n_quad_pt;
                        target_location = target_location,
                    )

                    for j in 1:n_quad_pt
                        push!(Is, near_idx_i)
                        push!(Js, idx_global_corr + j)
                        push!(
                            Ls,
                            G_L(target_i, quad_j_el[j]) *
                                Inti.weight(quad_j_el[j]) *
                                wcorrL[j] + G_C(target_i, quad_j_el[j]) * wcmpC[j],
                        )
                    end

                elseif M_sub > 1 # Subdivision needed
                    # Subtract original coarse quadrature
                    for j in 1:n_quad_pt
                        quad_j = source_quad[idx_global_corr + j]
                        push!(Is, near_idx_i)
                        push!(Js, idx_global_corr + j)
                        push!(Ls, -G(target_i, quad_j) * Inti.weight(quad_j))
                    end

                    for k_sub in 1:M_sub # Add contributions from each subpanel
                        subpanel_node1, subpanel_node2 = panel_subdivision_vec[i][k_sub]

                        # Create the subpanel quadrature object using meshgen
                        t_Leg_sub = scale_fn(subpanel_node1, subpanel_node2, t_Leg_ref)
                        panel_curve = Inti.parametric_curve(
                            panel_parametrization,
                            subpanel_node1,
                            subpanel_node2,
                        )
                        panel_msh = Inti.meshgen(
                            panel_curve;
                            meshsize = (subpanel_node2 - subpanel_node1),
                        )
                        panel_quad =
                            Inti.Quadrature(panel_msh, Inti.GaussLegendre(n_quad_pt))

                        # Add regular refined quadrature contribution using the robust panel_quad
                        for ĵ in 1:n_quad_pt
                            quad_ĵ = panel_quad[ĵ]
                            quad_weight_ĵ = Inti.weight(quad_ĵ)
                            bary_coef = Barycentric_coef(
                                t_Leg_sub[ĵ];
                                n = n_quad_pt,
                                t_interp = t_Leg_ref,
                                w = w_bary,
                            )

                            for j in 1:n_quad_pt
                                push!(Is, near_idx_i)
                                push!(Js, idx_global_corr + j)
                                push!(
                                    Ls,
                                    G(target_i, quad_ĵ) * quad_weight_ĵ * bary_coef[j],
                                )
                            end
                        end

                        # Add kernel split correction if needed
                        if ksplit_bool_vec[i][k_sub] == true
                            panel_node_a_complex =
                                panel_parametrization(subpanel_node1)[1] +
                                im * panel_parametrization(subpanel_node1)[2]
                            panel_node_b_complex =
                                panel_parametrization(subpanel_node2)[1] +
                                im * panel_parametrization(subpanel_node2)[2]

                            panel_node_complex = [
                                Inti.coords(q)[1] + im * Inti.coords(q)[2] for
                                    q in panel_quad
                            ]
                            panel_normal_complex = [
                                Inti.normal(q)[1] + im * Inti.normal(q)[2] for
                                    q in panel_quad
                            ]
                            tj_sub = boundary_inv.(Inti.coords.(panel_quad))
                            quad_velocity_weight =
                                (t_b - t_a) / 2 * (subpanel_node2 - subpanel_node1) / 2 .*
                                velocity_fn.(tj_sub) .* w_Leg_ref

                            wcorrL, wcmpC = wLCinit(
                                panel_node_a_complex,
                                panel_node_b_complex,
                                target_node_i_complex,
                                panel_node_complex,
                                panel_normal_complex,
                                quad_velocity_weight,
                                n_quad_pt;
                                target_location = target_location,
                            )

                            for ĵ in 1:n_quad_pt
                                quad_ĵ = panel_quad[ĵ]
                                quad_weight_ĵ = Inti.weight(quad_ĵ)
                                bary_coef = Barycentric_coef(
                                    t_Leg_sub[ĵ];
                                    n = n_quad_pt,
                                    t_interp = t_Leg_ref,
                                    w = w_bary,
                                )
                                correction_term =
                                    G_L(target_i, quad_ĵ) * quad_weight_ĵ * wcorrL[ĵ] +
                                    G_C(target_i, quad_ĵ) * wcmpC[ĵ]
                                for j in 1:n_quad_pt
                                    push!(Is, near_idx_i)
                                    push!(Js, idx_global_corr + j)
                                    push!(Ls, correction_term * bary_coef[j])
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    δL = sparse(Is, Js, Ls, n_target, n_source)

    # HACK: Clear Gmsh entities from Inti.meshgen in Inti to prevent memory leak
    # This is a temporary workaround and should be removed once the underlying issue is fixed.
    Inti.clear_entities!()

    return δL
end
