# The interpolation basis lives in `particular_basis.jl`. Everything below consumes it
# through one contract, and nothing else:
#
#   pb                = particular_basis(op, order);  length(pb) — how many functions
#   b, γ₀, γ₁, σ      = green_identity_terms(pb, variant, center, radius)
#   evaluate!(out, b, x)   /   evaluate!(out, γ₁, x, n)
#
# where `b(x)`, `γ₀(x)` and `σ(x)` return a length-`nb` vector of values at the physical point
# `x` and `γ₁(x, n)` the same given also a unit normal, such that the density interpolated in
# `b` satisfies `S[γ₁_β] - D[γ₀_β] + μσ_β + 𝒱[b_β] = 0` — `bdim`'s convention for `μ`, so the
# left-hand side is exactly what `Rt` accumulates below. Green's identity needs nothing more.
#
# `evaluate!` is that same value written into storage this file owns, and is what the loops
# below call: a term is evaluated once per patch node per element, so the vector `b(x)` would
# otherwise return is the dominant cost of the whole correction. It carries no new
# information — the fallback in `particular_basis.jl` is `out .= b(x)` — so a term needing
# nothing faster still satisfies the contract.
#
# In particular `lvdim` never learns what `b_β` *is*: not that it is polynomial, not that it
# is the monomials, not which operators have such a basis (that is `particular_basis`'s
# dispatch). Supporting a new operator means supplying that basis and nothing in this file.

# Each term of the identity is accumulated into `Rt` in one pass over the patch nodes. `Rt` is
# `num_basis × num_targets` — transposed relative to the correction it becomes — so that the
# innermost loop is a plain SIMD axpy.

# `𝒱[b_β] = Σ_j G(x, y_j) w_j b_β(y_j)`, the naive-quadrature side. This must be *exactly*
# `source`'s quadrature restricted to the patch elements, since `R = quad - exact` cancels the
# error the forward operator commits. So iterate `source`'s own nodes in place, which also picks
# up each type's own rule on a mixed patch for free.
function _lvdim_volume!(Rt, b, bbuf, X, etypes, patch_lists, source, G)
    for (E′, idxs) in zip(etypes, patch_lists)
        isempty(idxs) && continue
        qtags = get(source.etype2qtags, E′, nothing)
        isnothing(qtags) && error(
            "lvdim: patch element type $E′ is not integrated over by `source`, so no \
             consistent patch quadrature exists",
        )
        nq = size(qtags, 1)
        for idx in idxs
            for k in 1:nq
                y = source.qnodes[qtags[k, idx]]
                v = evaluate!(bbuf, b, coords(y))
                w = weight(y)
                @inbounds for i in eachindex(X)
                    g = G(X[i], y) * w
                    @simd for β in axes(Rt, 1)
                        Rt[β, i] += g * v[β]
                    end
                end
            end
        end
    end
    return Rt
end

# `S[γ₁P_β] - D[γ₀P_β]` over `∂Ωτ`, where the layer operators have been materialized as dense
# `(target, node)` matrices — the path taken when `∂Ωτ` meets `∂Ω` and `bdim` has to correct
# them. The interior path never comes through here: it evaluates the kernels on the fly in
# `_lvdim_layer_onfly!`, which has its own loop.
function _lvdim_layer!(Rt, Ψ, γ₁Ψ, vbuf, dvbuf, X, Ybdry, S, D)
    for l in 1:length(Ybdry)
        y = Ybdry[l]
        v = evaluate!(vbuf, Ψ, coords(y))
        dv = evaluate!(dvbuf, γ₁Ψ, coords(y), normal(y))
        @inbounds for i in eachindex(X)
            s, d = S[i, l], D[i, l]
            @simd for β in axes(Rt, 1)
                Rt[β, i] += s * dv[β] - d * v[β]
            end
        end
    end
    return Rt
end

# Same accumulation over the patch-boundary faces owned by one element type, mapping the
# reference-face quadrature through each face's true geometry on the fly — no boundary quadrature
# is stored. This is the interior-patch path, where `∂Ωτ` is a full layer of elements away from
# the targets so the free-space kernel evaluates cleanly and needs no `bdim`. A function barrier
# on the concrete `els` type keeps `boundary_element` type-stable (`E′` is a runtime `DataType`).
function _lvdim_layer_onfly!(Rt, Ψ, γ₁Ψ, vbuf, dvbuf, X, els, list, x̂, ŵ, G, dG)
    for (idx, k) in list
        bord = boundary_element(els[idx], k)
        for (x̂i, ŵi) in zip(x̂, ŵ)
            y = bord(x̂i)
            jac = jacobian(bord, x̂i)
            w = _integration_measure(jac) * ŵi
            n = _normal(jac)
            # `dG` reads the source normal, so the kernels need a node, not a bare point
            qy = QuadratureNode(y, w, n)
            v = evaluate!(vbuf, Ψ, y)
            dv = evaluate!(dvbuf, γ₁Ψ, y, n)
            @inbounds for i in eachindex(X)
                # both kernels at the same pair, so they are asked for together
                gs, gd = layer_pair_value(G, dG, X[i], qy)
                s, d = gs * w, gd * w
                @simd for β in axes(Rt, 1)
                    Rt[β, i] += s * dv[β] - d * v[β]
                end
            end
        end
    end
    return Rt
end

# `Rᵀ = quad - exact` for one element, over all near targets and all basis functions. Nothing of
# size `num_targets × num_nodes` is formed: a patch holds `O(1)` elements, so every kernel entry
# is used exactly once. The exception is a patch whose boundary carries a piece of `∂Ω`, which is
# delegated to `single_double_layer` and allocates its own `S`/`D`.
#
# The branch is purely *topological*: an interior patch always has `∂Ωτ` a full layer of elements
# away from `τ`'s nodes, so plain quadrature suffices, and only where `∂Ωτ` runs along `∂Ω` can
# it pass through `τ` itself. Distance to `∂Ωτ` would be sharper but needs a length scale to
# compare against, and any such threshold can silently under-correct on an untested mesh.
function _lvdim_element_R!(Rt, ctx, scr, γ₀Ψ, γ₁Ψ, σ, b, near, need_layer_corr)
    op, msh, source = ctx.op, ctx.msh, ctx.source
    X, μ = view(ctx.target, near), view(ctx.green_multiplier, near)
    Gv, Gs, Gd = ctx.volume_kernel, ctx.layer_kernels...
    # the free term `μ(x)σ_β(x)` seeds `Rt`, hence `=` and not `+=`
    for i in eachindex(X)
        v = evaluate!(scr.σbuf, σ, coords(X[i]))
        @inbounds @simd for β in axes(Rt, 1)
            Rt[β, i] = μ[i] * v[β]
        end
    end
    _lvdim_volume!(Rt, b, scr.bbuf, X, scr.etypes, scr.patch_lists, source, Gv)
    if need_layer_corr
        # `∂Ωτ` runs along `∂Ω`, so the layer potentials over it are nearly singular at the
        # targets: materialize the boundary quadrature and correct it with `bdim`
        Ybdry = Quadrature{ctx.N, Float64}(OrderedDict{DataType, ReferenceQuadrature}())
        for (E′, list) in zip(scr.etypes, scr.owner_lists)
            isempty(list) && continue
            _lvdim_add_faces!(Ybdry, elements(msh, E′), list, ctx.bdry_qrule)
        end
        S, D = single_double_layer(;
            op, target = X, source = Ybdry,
            compression = (method = :none,),
            correction = (
                method = :dim,
                green_multiplier = collect(Float64, μ),
                maxdist = Inf,
            ),
            kernel_variant = ctx.layer_variant,
        )
        _lvdim_layer!(Rt, γ₀Ψ, γ₁Ψ, scr.vbuf, scr.dvbuf, X, Ybdry, S, D)
    else
        for (E′, list) in zip(scr.etypes, scr.owner_lists)
            isempty(list) && continue
            _lvdim_layer_onfly!(
                Rt, γ₀Ψ, γ₁Ψ, scr.vbuf, scr.dvbuf, X,
                elements(msh, E′), list, ctx.bdry_x̂, ctx.bdry_ŵ, Gs, Gd,
            )
        end
    end
    return Rt
end

# ---------------------------------------------------------------------------
# Local patch geometry
# ---------------------------------------------------------------------------

# Topological boundary of the (possibly mixed-type) patch: the faces appearing exactly once.
# Returns the sorted node tuples (for comparing against `∂Ω`) and, for each, the
# `(type, element, local face)` owning it — the geometry then comes from the element itself via
# `boundary_element`, never rebuilt from the node coordinates. Faces are fixed-width
# `SVector{Nf,Int}` in pre-sized flat buffers, which keeps the sort/compare allocation-free (a
# `Vector{Int}`-per-face version was ~8x slower); in a fixed ambient dimension every element type
# shares the same face-node count `Nf`.
#
# The owning element type travels as an index into the caller's list of element
# types, not as the `DataType` itself: a tuple holding a type is not `isbits`,
# so a `Vector{PatchFace}` of them heap-allocates once per face — and a patch's
# faces are rebuilt for every element in the mesh.
const PatchFace = Tuple{Int, Int, Int}

function _local_patch_boundary(etypes, patch_lists, conns)
    nfaces = Nf = 0
    for (E′, idxs) in zip(etypes, patch_lists)
        isempty(idxs) && continue
        bdi = boundary_idxs(E′)
        Nf = length(first(bdi))
        nfaces += length(idxs) * length(bdi)
    end
    nfaces == 0 && return SVector{Nf, Int}[], PatchFace[]
    return _local_patch_boundary(etypes, patch_lists, conns, Val(Nf))
end

function _local_patch_boundary(etypes, patch_lists, conns, ::Val{Nf}) where {Nf}
    faces, owners = SVector{Nf, Int}[], PatchFace[]
    keep = _patch_boundary_faces!(faces, owners, Int[], Int[], etypes, patch_lists, conns)
    return faces[keep], owners[keep]
end

# The `keep` half of the above, writing into buffers the caller owns and returning the
# surviving indices rather than gathered copies. This is what the element loop runs: a
# patch's topology is rebuilt for every element, so the four arrays here are the ones worth
# handing back to the next element rather than to the collector.
function _patch_boundary_faces!(
        faces::Vector{SVector{Nf, Int}}, owners, perm, keep, etypes, patch_lists, conns,
    ) where {Nf}
    nfaces = 0
    for (E′, idxs) in zip(etypes, patch_lists)
        isempty(idxs) && continue
        nfaces += length(idxs) * length(boundary_idxs(E′))
    end
    resize!(faces, nfaces)
    resize!(owners, nfaces)
    resize!(perm, nfaces)
    empty!(keep)
    nfaces == 0 && return keep
    j = 1
    for (it, E′) in enumerate(etypes)
        idxs = patch_lists[it]
        isempty(idxs) && continue
        # barrier specializing the fill loop on the concrete connectivity-matrix and
        # `boundary_idxs` types (`E′` is a runtime `DataType`)
        j = _fill_patch_faces!(
            faces, owners, conns[E′], boundary_idxs(E′), it, idxs, j, Val(Nf),
        )
    end
    sortperm!(perm, faces)
    i = 1
    while i <= nfaces
        if i < nfaces && faces[perm[i]] == faces[perm[i + 1]]
            i += 2 # interior face, shared by two patch elements: drop both
        else
            push!(keep, perm[i])
            i += 1
        end
    end
    return keep
end

@inline function _fill_patch_faces!(
        faces, owners, conn::AbstractMatrix, bdi, it::Int, idxs, j, ::Val{Nf},
    ) where {Nf}
    for el in idxs
        for (k, face) in enumerate(bdi)
            faces[j] = sort(SVector(ntuple(i -> conn[face[i], el], Val(Nf))))
            owners[j] = (it, el, k)
            j += 1
        end
    end
    return j
end

# Topological boundary `∂Ωτ` of the patch around one element: the owning `(element, local face)`
# of each boundary face, grouped by owning element type so both consumers can take that type
# behind a function barrier. Also reports whether `∂Ωτ` contains a face of `∂Ω`, in which case the
# layer potentials over it need a `bdim_correction`.
#
# No quadrature is built here: the faces carry their own geometry through the owning element and
# are realized on the fly, or gathered into an `Ybdry` only when `∂Ωτ` meets `∂Ω`. The patch
# *volume* quadrature is likewise never materialized — it is `source`'s own nodes.
function _lvdim_patch_boundary!(scr, conns, bdry_faces)
    keep = _patch_boundary_faces!(
        scr.faces, scr.owners, scr.perm, scr.keep, scr.etypes, scr.patch_lists, conns,
    )
    for list in scr.owner_lists
        empty!(list)
    end
    touches_bdry = false
    for i in keep
        touches_bdry |= scr.faces[i] in bdry_faces
        it, el, k = scr.owners[i]
        push!(scr.owner_lists[it], (el, k))
    end
    return touches_bdry
end

# function barrier: `E′` is a runtime `DataType`, so specialize the face construction on the
# concrete element type once per type group
function _lvdim_add_faces!(Ybdry, els, list, bdry_qrule)
    bords = [boundary_element(els[el], k) for (el, k) in list]
    return _build_quadrature!(Ybdry, bords, ones(Int, length(bords)), bdry_qrule)
end

# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------

# Which kernels a variant's identity is written against: one for the naive volume quadrature, a
# pair for the layer potentials over `∂Ωτ`, and the `kernel_variant` that pair corresponds to
# (needed when `∂Ωτ` meets `∂Ω` and the layers go through `single_double_layer`). The rows of the
# table in `green_identity_terms`, read for their operators rather than their traces.
function _lvdim_kernels(op, variant::Symbol)
    variant === :default &&
        return SingleLayerKernel(op),
        (SingleLayerKernel(op), DoubleLayerKernel(op)), :default
    variant === :gradient &&
        return GradientSingleLayerKernel(op),
        (GradientSingleLayerKernel(op), GradientDoubleLayerKernel(op)), :gradient
    # `W[g] = -∫∇_yG·g = +∫∇ₓG·g`, so the volume kernel is the *target* gradient — the same one
    # `:gradient` uses — while the layers stay the plain scalar pair
    variant === :gradient_source &&
        return GradientSingleLayerKernel(op),
        (SingleLayerKernel(op), DoubleLayerKernel(op)), :default
    return error(
        "lvdim does not support kernel_variant = :$variant. `:hessian` needs two by-parts \
         boundary terms on two different operators, one more than the Green identity this \
         method assembles carries; use `correction = (method = :dim, ...)` for it.",
    )
end

# `Rt`'s element type, and `δV`'s. They differ only for `W`, whose Green identity is an
# `SVector` over the density directions while the correction it becomes contracts a density to
# a scalar.
function _lvdim_eltypes(op, variant::Symbol)
    T, N = default_kernel_eltype(op), ambient_dimension(op)
    variant === :default && return T, T
    variant === :gradient && return SVector{N, T}, SVector{N, T}
    variant === :gradient_source && return SVector{N, T}, Transpose{T, SVector{N, T}}
    return error(
        "lvdim does not support kernel_variant = :$variant. See the docstring for \
         `lvdim_correction`.",
    )
end

"""
    lvdim_correction(op, target, source; green_multiplier, kwargs...)

Return the sparse `δV` such that `V + δV` is an accurate approximation of the
volume-potential operator `V : source → target`, computed with the local (patch-based)
density interpolation method described in the header of `src/lvdim.jl`. The patches are built on
`mesh(source)`, the mesh that `source` integrates over, and each element is rescaled to its own
frame by [`translation_and_scaling`](@ref).

The local counterpart of [`vdim_correction`](@ref), and deliberately its signature minus the
operators: `S`, `D` and `V` are what the global method needs accelerated over the whole mesh,
whereas here every potential is evaluated directly on a patch of `O(1)` elements.

## Optional `kwargs`:
- `interpolation_order`: order of the local polynomial interpolation; defaults to
  [`_dim_interpolation_order`](@ref)`(source)`, as it does for [`vdim_correction`](@ref).
- `bdry_qorder`: order of the quadrature rule on the patch boundary `∂Ωτ`; defaults to the
  lowest quadrature order in `source` plus three. The patch *volume* quadrature is not a
  parameter: it is `source`'s own quadrature restricted to the patch.
- `nneighbors`: depth of the element patch `Ωτ`, in layers of topological neighbors (see
  [`topological_neighbors`](@ref)). Defaults to `1`. A deeper patch pushes `∂Ωτ` further from
  the targets, so the layer potentials over it are less nearly singular, at the cost of a
  patch that grows with the depth.
- `kernel_variant`: which volume operator is being corrected — `:default` for `V`,
  `:gradient` for `∇V`, or `:gradient_source` for `W[g] = -∫∇yG⋅g`. Unlike
  [`vdim_correction`](@ref) no operators are passed in, so this selects the *kernels* the
  patch quadrature and the layer potentials over `∂Ωτ` are built from. `:hessian` is not
  available: it needs two by-parts boundary terms on two different operators, one more than
  the identity assembled here carries (see [`green_identity_terms`](@ref)).
- `maxdist`: a target further than this from the nearest source quadrature node is left
  uncorrected. Only meaningful when `target !== source`: for `target === source` the near
  list is each element's own quadrature nodes and `maxdist` is never consulted. Purely an
  efficiency filter, and it does not affect which `green_multiplier` is valid, since a target
  is always assigned to the element owning its nearest node and hence lies in that element's
  patch.
"""
function lvdim_correction(
        op,
        target,
        source::Quadrature;
        green_multiplier::Vector{<:Real},
        interpolation_order = nothing,
        bdry_qorder = nothing,
        nneighbors = 1,
        maxdist = Inf,
        kernel_variant::Symbol = :default,
    )
    m, n = length(target), length(source)
    N = ambient_dimension(op)
    @assert ambient_dimension(source) == N "lvdim only works for volume potentials"
    volume_kernel, layer_kernels, layer_variant = _lvdim_kernels(op, kernel_variant)
    # the entries of `δV` are kernel values, so they follow the kernel's own type — complex
    # for Helmholtz — exactly as `vdim_correction` takes them from `eltype(Vop)`
    Rtype, Eltype = _lvdim_eltypes(op, kernel_variant)
    iorder = something(interpolation_order, _dim_interpolation_order(source))
    # the ∂Ωτ rule integrates the traces of `P_β` against a nearly-singular kernel, so it
    # needs more than the volume rule; the weakest element type sets the pace
    isnothing(bdry_qorder) &&
        (bdry_qorder = minimum(order, values(source.etype2qrule)) + 3)

    # the patches are built on the very mesh `source` integrates over: the element indices
    # below index both, so they cannot be different meshes
    msh = mesh(source)

    dict_near = etype_to_nearest_points(target, source; maxdist)
    neighbors = topological_neighbors(msh, nneighbors)
    # `nodes`/`connectivity` are recomputed from scratch on every call for a `SubMesh`, so
    # hoist them out of the element loop
    conns = OrderedDict{DataType, Matrix{Int}}(
        E => connectivity(msh, E) for E in element_types(msh)
    )
    # faces of the whole mesh appearing exactly once are the faces of `∂Ω`: the same
    # criterion `_local_patch_boundary` applies to a patch, so reuse it over every element
    etypes = collect(DataType, element_types(msh))
    etype_index = Dict{DataType, Int}(E => i for (i, E) in enumerate(etypes))
    whole_mesh = [collect(axes(conns[E], 2)) for E in etypes]
    bdry_faces = Set(first(_local_patch_boundary(etypes, whole_mesh, conns)))

    # Patch quadrature rule (shared, read-only). Chosen here rather than via
    # `_qrule_for_reference_shape`, which returns a Vioreanu-Rokhlin rule on a triangle: VR
    # rules are *interpolation* rules and are tabulated only at select orders, so the default
    # `bdry_qorder` could ask for one that does not exist. Nothing on `∂Ωτ` is interpolated,
    # so a plain Gauss rule is both the right tool and available at every order.
    bdry_qrule = if N == 3
        Gauss(; domain = :triangle, order = gauss_triangle_order(bdry_qorder))
    else
        GaussLegendre(; order = bdry_qorder)
    end
    # reference-face nodes/weights, mapped on the fly onto each patch-boundary face
    bdry_x̂, bdry_ŵ = qcoords(bdry_qrule), qweights(bdry_qrule)

    # the element-independent half of the basis: every polynomial solve happens here, once
    # for the whole mesh, and each element only centres and scales it
    pb = particular_basis(op, iorder)
    # asked of the basis rather than recomputed from `(op, iorder)`, so the two cannot disagree
    nb = length(pb)

    # everything below the element loop is read-only and the same for every element. The
    # variant travels as a `Val` so that the terms of its identity stay inferrable per element
    ctx = (;
        op, nb, N, msh, source, target, green_multiplier,
        neighbors, conns, etypes, etype_index, bdry_faces, bdry_qrule, bdry_x̂, bdry_ŵ, pb,
        variant = Val(kernel_variant),
        volume_kernel, layer_kernels, layer_variant, Rtype,
        # prototypes for the reusable buffers; see `_lvdim_scratch`
        face_proto = SVector{length(first(boundary_idxs(first(element_types(msh))))), Int}[],
        _lvdim_term_prototypes(pb, Val(kernel_variant), msh, source, N)...,
    )

    # The elements partition `δV`'s columns, so the whole sparsity structure is known before
    # any entry is: lay out the CSC arrays up front and let every element write its own
    # columns straight into them (see `_dim_csc_storage`).
    colptr, rowval, nzval = _dim_csc_storage(source, dict_near, Eltype)
    for (E, qtags) in source.etype2qtags
        els = elements(msh, E)
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        ne == 0 && continue
        # Elements are the unit of parallelism: each owns a disjoint set of columns, so no
        # lock, and a `δV` independent of the thread count. Tasks are spawned per *chunk* so
        # that the scratch each element writes through (see `_lvdim_scratch`) is built once
        # per task; more chunks than threads keeps the boundary elements, which cost a
        # `bdim_correction` the interior ones do not, from unbalancing the split.
        maxnear = maximum(length, near_list)
        chunk = max(1, cld(ne, 8 * Threads.nthreads()))
        @sync for els_chunk in Iterators.partition(1:ne, chunk)
            Threads.@spawn begin
                scr = _lvdim_scratch(ctx, Rtype, nq, maxnear)
                for el in els_chunk
                    _lvdim_element!(
                        colptr, rowval, nzval, ctx, scr, E, els, qtags, el, near_list[el],
                    )
                end
            end
        end
    end
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

# One element's columns of `δV`, written into the caller's preallocated CSC arrays at the
# offsets `colptr` already fixes. A function rather than the `@threads` body inlined, so that
# what it captures stays concretely typed and `E` — a runtime `DataType` — is specialized on
# once per element.
function _lvdim_element!(colptr, rowval, nzval, ctx, scr, E, els, qtags, el, near)
    isempty(near) && return nothing
    # the terms of this element's Green identity, centred and scaled to the element
    b, γ₀Ψ, γ₁Ψ, σ =
        green_identity_terms(ctx.pb, ctx.variant, translation_and_scaling(els[el])...)
    # on a curved mesh a single patch mixes straight `LagrangeElement`s with curved
    # `ParametricElement`s, so indices must stay paired with their type. The lists are keyed
    # on every type in the mesh once and emptied per element, so a patch costs no allocation
    for idxs in scr.patch_lists
        empty!(idxs)
    end
    for (E′, idx) in ctx.neighbors[(E, el)]
        push!(scr.patch_lists[ctx.etype_index[E′]], idx)
    end
    need_corr = _lvdim_patch_boundary!(scr, ctx.conns, ctx.bdry_faces)
    Rt = view(scr.Rt, :, 1:length(near))
    _lvdim_element_R!(Rt, ctx, scr, γ₀Ψ, γ₁Ψ, σ, b, near, need_corr)
    jglob = @view qtags[:, el]
    # `b` is a *batched* basis, so the Vandermonde is one column per node
    L = vandermonde!(scr.L, b, (coords(q) for q in view(ctx.source, jglob)))
    # the same interpolation solve `vdim_correction` performs, on a locally computed `Rt`
    _dim_solve!(colptr, rowval, nzval, L, Rt, jglob, near, scr.solve)
    return nothing
end

"""
    _lvdim_scratch(ctx, Rtype, nq, maxnear) -> scr

Everything one task reuses across the elements it owns: the patch-topology
buffers, the per-element `Rᵀ` and Vandermonde, and one output vector per term of
the Green identity.

This keeps the element loop allocation-free; the loop below spawns one task per
chunk of elements.

The term buffers are sized and typed from `ctx`'s prototypes rather than from
`Rtype`: how long a term's vector is and what it holds is not something this
file is told (see the header), so it is read off one evaluation instead of
derived.
"""
function _lvdim_scratch(ctx, ::Type{Rtype}, nq, maxnear) where {Rtype}
    nt = length(ctx.etypes)
    return (;
        # positional, not keyed: `_fill_patch_faces!` stores the owning type as its index
        etypes = ctx.etypes,
        patch_lists = [Int[] for _ in 1:nt],
        owner_lists = [Tuple{Int, Int}[] for _ in 1:nt],
        faces = similar(ctx.face_proto, 0),
        owners = PatchFace[],
        perm = Int[],
        keep = Int[],
        # `Rtype`, not `Float64`: the contract does not promise `b` is real-valued
        Rt = Matrix{Rtype}(undef, ctx.nb, maxnear),
        L = similar(ctx.b_proto, ctx.nb, nq),
        solve = _dim_work(similar(ctx.b_proto, ctx.nb, nq), Matrix{Rtype}(undef, 0, 0), maxnear),
        bbuf = similar(ctx.b_proto),
        σbuf = similar(ctx.σ_proto),
        vbuf = similar(ctx.γ₀_proto),
        dvbuf = similar(ctx.γ₁_proto),
    )
end

# One evaluation of each term of the identity, for the buffer sizes and element
# types the element loop then reuses, executed on the first element of the first
# type.
function _lvdim_term_prototypes(pb, variant, msh, source, N)
    E = first(keys(source.etype2qtags))
    c, r = translation_and_scaling(elements(msh, E)[1])
    b, γ₀, γ₁, σ = green_identity_terms(pb, variant, c, r)
    ν = SVector(ntuple(i -> i == 1 ? 1.0 : 0.0, N))
    return (
        b_proto = collect(b(c)), γ₀_proto = collect(γ₀(c)),
        γ₁_proto = collect(γ₁(c, ν)), σ_proto = collect(σ(c)),
    )
end
