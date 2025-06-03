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
    correction, derivative = false)

Construct a discrete approximation to the single- and double-layer integral operators for
`op`, mapping values defined on the quadrature nodes of `source` to values defined on the
nodes of `target`. If `derivative = true`, return instead the adjoint double-layer and
hypersingular operators (which are the generalized Neumann trace of the single- and
double-layer, respectively).

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
    near-singular interactions are computing using an adaptive quadrature rule. The `tol`
    argument specifies the tolerance of the adaptive integration. See
    [`adaptive_correction`](@ref) for more details.
  - `(method = :dim, maxdist, target_location)`: use the density interpolation method to
    compute the correction. `maxdist` specifies the distance between source and target
    points above which no correction is performed (defaults to `Inf`). `target_location`
    should be either `:inside`, `:outside`, or `:on`, and specifies where the `target``
    points lie relative to the to the `source` curve/surface (which is assumed to be
    closed). When `target === source`, `target_location` is not needed. See
    [`bdim_correction`](@ref) and [`vdim_correction`] for more details.
"""
function single_double_layer(;
    op,
    target,
    source,
    compression = (method = :none,),
    correction = (method = :adaptive,),
    derivative = false,
)
    compression = _normalize_compression(compression, target, source)
    correction  = _normalize_correction(correction, target, source)
    G           = derivative ? AdjointDoubleLayerKernel(op) : SingleLayerKernel(op)
    dG          = derivative ? HyperSingularKernel(op) : DoubleLayerKernel(op)
    Sop         = IntegralOperator(G, target, source)
    Dop         = IntegralOperator(dG, target, source)
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
        loc = target === source ? :on : correction.target_location
        μ = _green_multiplier(loc)
        green_multiplier = fill(μ, length(target))
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
                green_multiplier,
                correction.maxdist,
                derivative,
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
                derivative,
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
        derivative = true,
    )
end

"""
    single_double_layer_potential(; op, source)

Return the single- and double-layer potentials for `op` as
[`IntegralPotential`](@ref)s.
"""
function single_double_layer_potential(; op, source)
    G    = SingleLayerKernel(op)
    dG   = DoubleLayerKernel(op)
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
function volume_potential(; op, target, source::Quadrature, compression, correction)
    correction = _normalize_correction(correction, target, source)
    compression = _normalize_compression(compression, target, source)
    G = SingleLayerKernel(op)
    V = IntegralOperator(G, target, source)
    # compress V
    if compression.method == :none
        Vmat = assemble_matrix(V)
    elseif compression.method == :hmatrix
        Vmat = assemble_hmatrix(V; rtol = compression.tol)
    elseif compression.method == :fmm
        Vmat = assemble_fmm(V; rtol = compression.tol)
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
        loc = target === source ? :inside : correction.target_location
        μ = _green_multiplier(loc)
        green_multiplier = fill(μ, length(target))
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
        # Advanced usage: Use previously constructed layer operators for VDIM
        if !haskey(correction, :S_b2d) || !haskey(correction, :D_b2d)
            S, D = single_double_layer(;
                op,
                target,
                source = boundary,
                compression,
                correction = (correction..., target_location = loc),
            )
        else
            S = correction.S_b2d
            D = correction.D_b2d
        end
        interpolation_order = correction.interpolation_order
        δV = vdim_correction(
            op,
            target,
            source,
            boundary,
            S,
            D,
            Vmat;
            green_multiplier,
            correction.maxdist,
            interpolation_order,
        )
    else
        error("Unknown correction method. Available options: $CORRECTION_METHODS")
    end
    # add correction
    if compression.method ∈ (:hmatrix, :none)
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
