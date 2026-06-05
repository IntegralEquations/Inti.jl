"""
    const COMPRESSION_METHODS = [:none, :hmatrix, :fmm]

Available compression methods for the dense linear operators in [`Inti`](@ref).
"""
const COMPRESSION_METHODS = [:none, :hmatrix, :fmm]

"""
    const CORRECTION_METHODS = [:none, :dim, :adaptive]

Available correction methods for the singular and nearly-singular integrals in
[`Inti`](@ref).
"""
const CORRECTION_METHODS = [:none, :dim, :adaptive]

"""
    single_double_layer(; op, target, source::Quadrature, compression,
    correction, kernel_variant = :default)

Construct a discrete approximation to the single- and double-layer integral operators for
`op`, mapping values defined on the quadrature nodes of `source` to values defined on the
nodes of `target`. The `kernel_variant` keyword controls which pair of kernels is used:
`:default` gives the standard single- and double-layer kernels, `:neumann` gives the
adjoint double-layer and hypersingular kernels (i.e. the generalized Neumann trace of the
single- and double-layer), and `:gradient` gives the gradient kernels.

For finer control, you must choose a `compression` method and a `correction` method, as
described below.

# Compression

The `compression` argument is a named tuple with a `method` field followed by
method-specific fields. It specifies how the dense linear operators should be
compressed. The available options are:

  - `(method = :none, )`: no compression is performed, the resulting matrices
    are dense. This is the default, but not recommended for large problems.
  - `(method =:hmatrix, tol)`: the resulting operators are compressed using
    hierarchical matrices with an absolute tolerance `tol` (defaults to `1e-8`).
  - `(method = :fmm, tol)`: the resulting operators are compressed using the
    fast multipole method with an absolute tolerance `tol` (defaults to `1e-8`).

# Correction

The `correction` argument is a named tuple with a `method` field followed by
method-specific fields. It specifies how the singular and nearly-singular
integrals should be computed. The available options are:

  - `(method = :none, )`: no correction is performed. This is not recommended, as the
    resulting approximation will be inaccurate if the kernel is singular and source and
    target are not sufficiently far from each other.
  - `(method = :adaptive, maxdist, tol)`: correct interactions corresponding to entries of
    `target` and elements of `source` that are within `maxdist` of each other. The singular
    (including finite part) interactions are computed in polar coordinates, while the
    near-singular interactions are computed using an adaptive quadrature rule. The `tol`
    argument specifies the tolerance of the adaptive integration. See
    [`adaptive_correction`](@ref) for more details.
  - `(method = :dim, maxdist, target_location)`: use the density interpolation method to
    compute the correction. `maxdist` specifies the distance between source and target
    points above which no correction is performed (defaults to `Inf`). `target_location`
    should be either `:inside`, `:outside`, or `:on`, and specifies where the `target`
    points lie relative to the to the `source` curve/surface (which is assumed to be
    closed). When `target === source`, `target_location` is not needed. See
    [`bdim_correction`](@ref) and [`vdim_correction`](@ref) for more details.
"""
function single_double_layer(;
        op,
        target,
        source,
        compression = (method = :none,),
        correction = (method = :adaptive,),
        kernel_variant::Symbol = :default,
        derivative = nothing,
    )
    if !isnothing(derivative)
        Base.depwarn(
            "The `derivative` keyword is deprecated; use `kernel_variant = :neumann` instead.",
            :single_double_layer,
        )
        kernel_variant = derivative ? :neumann : :default
    end
    compression = _normalize_compression(compression, target, source)
    correction = _normalize_correction(correction, target, source)
    if kernel_variant === :gradient
        G = GradientSingleLayerKernel(op)
        dG = GradientDoubleLayerKernel(op)
    elseif kernel_variant === :neumann
        G = AdjointDoubleLayerKernel(op)
        dG = HyperSingularKernel(op)
    else
        G = SingleLayerKernel(op)
        dG = DoubleLayerKernel(op)
    end
    Sop = IntegralOperator(G, target, source)
    Dop = IntegralOperator(dG, target, source)
    # handle compression
    if compression.method == :hmatrix
        Smat = assemble_hmatrix(Sop; rtol = compression.tol)
        Dmat = assemble_hmatrix(Dop; rtol = compression.tol)
    elseif compression.method == :none
        Smat = assemble_matrix(Sop)
        Dmat = assemble_matrix(Dop)
    elseif compression.method == :fmm
        Smat = assemble_fmm(Sop; rtol = compression.tol)::LinearMap
        Dmat = assemble_fmm(Dop; rtol = compression.tol)::LinearMap
    else
        error("Unknown compression method. Available options: $COMPRESSION_METHODS")
    end

    # handle nearfield correction
    if correction.method == :none
        return Smat, Dmat # shortcircuit case without correction
    elseif correction.method == :dim
        if haskey(correction, :green_multiplier)
            @assert length(correction.green_multiplier) == length(target)
            green_multiplier = correction.green_multiplier
        else
            loc = target === source ? :on : correction.target_location
            μ = _green_multiplier(loc)
            green_multiplier = fill(μ, length(target))
        end
        dict_near = etype_to_nearest_points(target, source; correction.maxdist)
        # If target != source then we want to filter the near-field points and construct auxiliary
        # IntegralOperator with targets limited to those that will be corrected.
        if target !== source
            glob_near_trgs = Int[]
            for (E, qtags) in source.etype2qtags
                append!(glob_near_trgs, collect(Iterators.flatten(dict_near[E])))
            end
            glob_loc_near_trgs =
                Dict(glob_near_trgs[i] => i for i in eachindex(glob_near_trgs))

            # Set up new IntegralOperator maps for only the targets needing correction
            Sop_dim = IntegralOperator(G, target[glob_near_trgs], source)
            Dop_dim = IntegralOperator(dG, target[glob_near_trgs], source)
            # compress 'em
            if compression.method == :hmatrix
                Sop_dim_mat = assemble_hmatrix(Sop_dim; rtol = compression.tol)
                Dop_dim_mat = assemble_hmatrix(Dop_dim; rtol = compression.tol)
            elseif compression.method == :none
                Sop_dim_mat = assemble_matrix(Sop_dim)
                Dop_dim_mat = assemble_matrix(Dop_dim)
            elseif compression.method == :fmm
                Sop_dim_mat = assemble_fmm(Sop_dim; rtol = compression.tol)::LinearMap
                Dop_dim_mat = assemble_fmm(Dop_dim; rtol = compression.tol)::LinearMap
            else
                error("Unknown compression method. Available options: $COMPRESSION_METHODS")
            end

            filter_target_params = (
                dict_near = dict_near,
                num_trgs = length(target),
                glob_loc_near_trgs = glob_loc_near_trgs,
            )
            δS, δD = bdim_correction(
                op,
                target[glob_near_trgs],
                source,
                Sop_dim_mat,
                Dop_dim_mat;
                green_multiplier = green_multiplier[glob_near_trgs],
                correction.maxdist,
                kernel_variant,
                filter_target_params,
            )
        else
            δS, δD = bdim_correction(
                op,
                target,
                source,
                Smat,
                Dmat;
                green_multiplier,
                correction.maxdist,
                kernel_variant,
            )
        end
    elseif correction.method == :adaptive
        # strip `method` from correction and pass it on
        correction_kw = Base.structdiff(correction, NamedTuple{(:method,)})
        δS = adaptive_correction(Sop; correction_kw...)
        δD = adaptive_correction(Dop; correction_kw...)
    else
        error("Unknown correction method. Available options: $CORRECTION_METHODS")
    end

    # combine near and far field
    if compression.method == :none
        S = axpy!(true, δS, Smat)
        D = axpy!(true, δD, Dmat)
    elseif compression.method == :hmatrix
        if target === source
            S = axpy!(true, δS, Smat)
            D = axpy!(true, δD, Dmat)
        else
            S = LinearMap(Smat) + LinearMap(δS)
            D = LinearMap(Dmat) + LinearMap(δD)
        end
    elseif compression.method == :fmm
        S = Smat + LinearMap(δS)
        D = Dmat + LinearMap(δD)
    end
    return S, D
end

"""
    adj_double_layer_hypersingular(; op, target, source, compression,
    correction)

Similar to `single_double_layer`, but for the adjoint double-layer and
hypersingular operators. See the documentation of [`single_double_layer`] for a
description of the arguments.
"""
function adj_double_layer_hypersingular(;
        op,
        target,
        source = target,
        compression = (method = :none,),
        correction = (method = :adaptive,),
    )
    return single_double_layer(;
        op,
        target,
        source,
        compression,
        correction,
        kernel_variant = :neumann,
    )
end

"""
    single_double_layer_potential(; op, source)

Return the single- and double-layer potentials for `op` as
[`IntegralPotential`](@ref)s.
"""
function single_double_layer_potential(; op, source)
    G = SingleLayerKernel(op)
    dG = DoubleLayerKernel(op)
    𝒮 = IntegralPotential(G, source)
    𝒟 = IntegralPotential(dG, source)
    return 𝒮, 𝒟
end

"""
    volume_potential(; op, target, source::Quadrature, compression, correction)

Compute the volume potential operator for a given PDE.

## Arguments
- `op`: The PDE (Partial Differential Equation) to solve.
- `target`: The target domain where the potential is computed.
- `source`: The source domain where the potential is generated.
- `compression`: The compression method to use for the potential operator.
- `correction`: The correction method to use for the potential operator.

## Returns

The volume potential operator `V` that represents the interaction between the
target and source domains.

## Compression

The `compression` argument is a named tuple with a `method` field followed by
method-specific fields. It specifies how the dense linear operators should be
compressed. The available options are:

  - `(method = :none, )`: no compression is performed, the resulting matrices
    are dense.
  - `(method =:hmatrix, tol)`: the resulting operators are compressed using
    hierarchical matrices with an absolute tolerance `tol` (defaults to `1e-8`).
  - `(method = :fmm, tol)`: the resulting operators are compressed using the
    fast multipole method with an absolute tolerance `tol` (defaults to `1e-8`).

## Correction

The `correction` argument is a named tuple with a `method` field followed by
method-specific fields. It specifies how the singular and nearly-singular
integrals should be computed. The available options are:

  - `(method = :none, )`: no correction is performed. This is not recommented,
    as the resulting approximation will be inaccurate if the source and target
    are not sufficiently far apart.
  - `(method = :dim, maxdist, target_location)`: use the density interpolation
    method to compute the correction. `maxdist` specifies the distance between
    source and target points above which no correction is performed (defaults to
    `Inf`). `target_location` should be either `:inside`, `:outside`, or `:on`,
    and specifies where the `target`` points lie relative to the to the
    `source`'s boundary. When `target === source`, `target_location` is not
    needed.

## Details
The volume potential operator is computed by assembling the integral operator
`V` using the single-layer kernel `G`. The operator `V` is then compressed using
the specified compression method. If no compression is specified, the operator
is returned as is. If a correction method is specified, the correction is
computed and added to the compressed operator.
"""
function volume_potential(; op, target, source::Quadrature, compression, correction, kernel_variant::Symbol = :default)
    correction = _normalize_correction(correction, target, source)
    compression = _normalize_compression(compression, target, source)
    if kernel_variant === :gradient
        G = GradientSingleLayerKernel(op)
    elseif kernel_variant === :gradient_source
        # naive forward map of W[g] = -∫∇yG⋅g (vector density -> scalar). The kernel is
        # ∇yG; the leading minus of W is applied when the operator is assembled below.
        G = SourceGradientSingleLayerKernel(op)
    elseif kernel_variant === :hessian_source
        # forward map of X[g] = ∇W[g] (vector density -> vector). The PV part is
        # +∫∇ₓ∇ₓG⋅g, so the (target) Hessian single-layer kernel is assembled as-is.
        G = HessianSingleLayerKernel(op)
    else
        G = SingleLayerKernel(op)
    end
    V = IntegralOperator(G, target, source)
    # compress V
    if compression.method == :none
        Vmat = assemble_matrix(V)
    elseif compression.method == :hmatrix
        Vmat = assemble_hmatrix(V; rtol = compression.tol)
    elseif compression.method == :fmm
        Vmat = assemble_fmm(V; rtol = compression.tol, ndiv = compression.ndiv)
    else
        error("Unknown compression method. Available options: $COMPRESSION_METHODS")
    end
    # compute correction
    if correction.method == :none
        return Vmat
    elseif correction.method == :adaptive
        # strip `method` from correction and pass it on
        correction_kw = Base.structdiff(correction, NamedTuple{(:method,)})
        δV = adaptive_correction(V; correction_kw...)
    elseif correction.method == :dim
        if haskey(correction, :green_multiplier)
            green_multiplier = correction.green_multiplier
        else
            loc = target === source ? :inside : correction.target_location
            μ = _green_multiplier(loc)
            green_multiplier = fill(μ, length(target))
        end
        if haskey(correction, :boundary)
            boundary = correction.boundary
        elseif source.mesh isa SubMesh # attempt to find the boundary in the parent mesh
            Ω = domain(source)
            Γ = external_boundary(Ω)
            par_msh = source.mesh.parent # parent mesh, hopefully containing the boundary
            all(ent -> ent ∈ entities(par_msh), keys(Γ)) ||
                error("Boundary not found in parent mesh")
            qmax = maximum(order, values(source.etype2qrule))
            boundary = Quadrature(view(par_msh, Γ); qorder = 2 * qmax)
        else
            error("Missing correction.boundary field for :dim method on a volume potential")
        end
        # The W (3.25) and X (3.28) regularizations both use the standard scalar
        # single-/double-layer potentials, so for those variants the boundary
        # operators are built with the `:default` kernel.
        boundary_variant = kernel_variant in (:gradient_source, :hessian_source) ? :default : kernel_variant
        # Advanced usage: Use previously constructed layer operators for VDIM
        if !haskey(correction, :S_b2d) || !haskey(correction, :D_b2d)
            if haskey(correction, :green_multiplier)
                S, D = single_double_layer(;
                    op,
                    target,
                    source = boundary,
                    compression,
                    correction,
                    kernel_variant = boundary_variant,
                )
            else
                S, D = single_double_layer(;
                    op,
                    target,
                    source = boundary,
                    compression,
                    correction = (correction..., target_location = loc),
                    kernel_variant = boundary_variant,
                )
            end
        else
            S = correction.S_b2d
            D = correction.D_b2d
        end
        # The X regularization (3.28) additionally needs the gradient single-layer
        # ∇ₓS for the `-∇ₓS[gⱼν]` term; build it from the `:gradient` variant (its
        # single-layer return, `GS`, is the gradient single-layer).
        grad_single_layer = nothing
        if kernel_variant === :hessian_source
            if haskey(correction, :green_multiplier)
                grad_single_layer, _ = single_double_layer(;
                    op, target, source = boundary, compression, correction,
                    kernel_variant = :gradient,
                )
            else
                grad_single_layer, _ = single_double_layer(;
                    op, target, source = boundary, compression,
                    correction = (correction..., target_location = loc),
                    kernel_variant = :gradient,
                )
            end
        end
        # Volume operator used to build the correction.
        if kernel_variant === :gradient_source
            Vg = IntegralOperator(GradientSingleLayerKernel(op), target, source)
            if compression.method == :none
                Vcorr = assemble_matrix(Vg)
            elseif compression.method == :hmatrix
                Vcorr = assemble_hmatrix(Vg; rtol = compression.tol)
            else
                Vcorr = assemble_fmm(Vg; rtol = compression.tol)
            end
        elseif kernel_variant === :hessian_source && compression.method == :fmm
            # X under FMM: the forward `Vmat` is a dipole→gradient map (`SVector` output)
            # and cannot produce the Hessian `SMatrix` from a scalar monomial density. Build
            # a dedicated charge→Hessian volume operator (scalar→`SMatrix`) so the correction
            # applies it once per monomial (see `_vdim_correction_X`).
            Vh = IntegralOperator(HessianSingleLayerKernel(op), target, source)
            if (ambient_dimension(op) == 3 && op isa Laplace) || (ambient_dimension(op) == 2 && (op isa Laplace || op isa Helmholtz))
                Vcorr = assemble_fmm_chargehessian(Vh; rtol = compression.tol)
            else
                Vcorr = assemble_fmm(Vh; rtol = compression.tol)
            end
        else
            Vcorr = Vmat
        end
        interpolation_order = correction.interpolation_order
        δV = vdim_correction(
            op,
            target,
            source,
            boundary,
            S,
            D,
            Vcorr;
            green_multiplier,
            correction.maxdist,
            interpolation_order,
            kernel_variant,
            grad_single_layer,
        )
    else
        error("Unknown correction method. Available options: $CORRECTION_METHODS")
    end
    # add correction
    if kernel_variant === :gradient_source
        # W maps a vector (`SVector`) density to a scalar. `Vmat` assembles
        # `∫∇yG⋅g`; the operator W[g] = -∫∇yG⋅g carries a leading minus, hence the
        # `-Vmat` forward map. Wrap it together with the sparse correction in a
        # `VectorDensityOperator` so that `W * g` allocates a clean scalar output
        # (see the type's docstring).
        V = VectorDensityOperator{default_kernel_eltype(op)}(-Vmat, δV, size(δV))
    elseif kernel_variant === :hessian_source
        # X maps a vector (`SVector`) density to a vector (`SVector`) output. The
        # PV part of `X[g] = ∇W[g]` equals `+∫∇ₓ∇ₓG⋅g`, so the forward map is `Vmat`
        # (no leading minus). Wrap it with the sparse correction in a
        # `VectorDensityOperator` so that `X * g` allocates a clean `Vector{SVector}`
        # output (a plain `LinearMap` would infer an abstract `SArray` eltype).
        N = ambient_dimension(op)
        V = VectorDensityOperator{SVector{N, default_kernel_eltype(op)}}(Vmat, δV, size(δV))
    elseif compression.method ∈ (:hmatrix, :none)
        # TODO: in the hmatrix case, we may want to add the correction directly
        # to the HMatrix so that a direct solver can be later used
        V = LinearMap(Vmat) + LinearMap(δV)
        # if target === source
        #     V = axpy!(true, δV, Vmat)
        # else
        #     V = LinearMap(Vmat) + LinearMap(δV)
        # end
    elseif compression.method == :fmm
        V = Vmat + LinearMap(δV)
    end
    return V
end
