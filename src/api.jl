"""
    single_double_layer(; pde, target, source::Quadrature, compression, correction)

Construct a discrete approximation to the single- and double-layer integral operators for `pde`,
mapping values defined on the quadrature nodes of `source` to values defined on
the nodes of `target`.

You  must choose a `compression` method and a `correction` method, as described below.

## Compression

The `compression` argument is a named tuple with a `method` field followed by
method-specific fields. It specifies how the dense linear operators should be
compressed. The available options are:

  - `(method = :none, )`: no compression is performed, the resulting matrices are dense.
  - `(method =:hmatrix, tol)`: the resulting operators are compressed using
    hierarchical matrices with an absolute tolerance `tol` (defaults to  `1e-8`).
  - `(method = :fmm, tol)`: the resulting operators are compressed using the fast multipole method
    with an absolute tolerance `tol` (defaults to `1e-8`).

## Correction

The `correction` argument is a named tuple with a `method` field followed by
method-specific fields. It specifies how the singular and nearly-singular
integrals should be computed. The available options are:

  - `(method = :none, )`: no correction is performed. This is not recommented,
    as the resulting approximation will be inaccurate if the source and target
    are not sufficiently far apart.
  - `(method = :dim, maxdist)`: use the density interpolation method to compute
    the correction. `maxdist` specifies the distance between
    source and target points above which no correction is performed
    (defaults to `Inf`).
"""
function single_double_layer(; pde, target, source, compression, correction)
    compression = _normalize_compression(compression)
    correction  = _normalize_correction(correction)
    G           = SingleLayerKernel(pde)
    dG          = DoubleLayerKernel(pde)
    Sop         = IntegralOperator(G, target, source)
    Dop         = IntegralOperator(dG, target, source)
    # handle compression
    if compression.method == :hmatrix
        Smat = assemble_hmatrix(Sop; atol = compression.tol)
        Dmat = assemble_hmatrix(Dop; atol = compression.tol)
    elseif compression.method == :none
        Smat = assemble_matrix(Sop)
        Dmat = assemble_matrix(Dop)
    elseif compression.method == :fmm
        Smat = assemble_fmm(Sop; atol = compression.tol)::LinearMap
        Dmat = assemble_fmm(Dop; atol = compression.tol)::LinearMap
    else
        error("Unknown compression method. Available options: $COMPRESSION_METHODS")
    end

    # handle nearfield correction
    if correction.method == :none
        return Smat, Dmat # shortcircuit case without correction
    elseif correction.method == :dim
        δS, δD =
            bdim_correction(pde, target, source, Smat, Dmat; maxdist = correction.maxdist)
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
    single_double_layer_potential(; pde, source)

Return the single- and double-layer potentials for `pde` as
[`IntegralPotential`](@ref)s.
"""
function single_double_layer_potential(; pde, source)
    G  = SingleLayerKernel(pde)
    dG = DoubleLayerKernel(pde)
    𝒮  = IntegralPotential(G, source)
    𝒟  = IntegralPotential(dG, source)
    return 𝒮, 𝒟
end

function volume_potential(; pde, target, source::Quadrature, compression, correction)
    correction = _normalize_correction(correction)
    compression = _normalize_compression(compression)
    G = SingleLayerKernel(pde)
    V = IntegralOperator(G, target, source)
    # compress V
    if compression.method == :none
        Vmat = assemble_matrix(V)
    elseif compression.method == :hmatrix
        Vmat = assemble_hmatrix(V; atol = compression.tol)
    elseif compression.method == :fmm
        Vmat = assemble_fmm(V; atol = compression.tol)
    else
        error("Unknown compression method. Available options: $COMPRESSION_METHODS")
    end
    # compute correction
    if correction.method == :none
        return Vmat
    elseif correction.method == :dim
        maxdist = correction.maxdist
        if haskey(correction, :boundary)
            boundary = correction.boundary
        else
            Ω = domain(source)
            Γ = external_boundary(Ω)
            msh = source.submesh.parent # parent mesh, hopefully containing the boundary
            Γ ∈ domain(msh) || error("Boundary not found in parent mesh")
            qmax = maximum(order, values(source.etype2qrule))
            boundary = Quadrature(msh, Γ; qorder = 2 * qmax)
        end
        S, D =
            single_double_layer(; pde, target, source = boundary, compression, correction)
        interpolation_order = correction.interpolation_order
        δV = vdim_correction(
            pde,
            target,
            source,
            boundary,
            S,
            D,
            Vmat;
            maxdist,
            interpolation_order,
        )
    else
        error("Unknown correction method. Available options: $CORRECTION_METHODS")
    end
    # add correction
    if compression.method ∈ (:hmatrix, :none)
        V = axpy!(true, δV, Vmat)
    elseif compression.method == :fmm
        V = Vmat + LinearMap(δV)
    end
    return V
end
