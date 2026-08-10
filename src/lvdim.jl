# The interpolation basis lives in `particular_basis.jl`. Everything below consumes it
# through one contract, and nothing else:
#
#   pb                = particular_basis(op, order);  length(pb) — how many functions
#   b, γ₀, γ₁, σ      = green_identity_terms(pb, variant, center, radius)
#
# where `b(x)`, `γ₀(x)` and `σ(x)` return a length-`nb` vector of values at the physical point
# `x` and `γ₁(x, n)` the same given also a unit normal, such that the density interpolated in
# `b` satisfies `S[γ₁_β] - D[γ₀_β] + μσ_β + 𝒱[b_β] = 0` — `bdim`'s convention for `μ`, so the
# left-hand side is exactly what `Rt` accumulates below. Green's identity needs nothing more.
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
function _lvdim_volume!(Rt, b, X, patch_by_type, source, G)
    for (E′, idxs) in patch_by_type
        qtags = get(source.etype2qtags, E′, nothing)
        isnothing(qtags) && error(
            "lvdim: patch element type $E′ is not integrated over by `source`, so no \
             consistent patch quadrature exists",
        )
        nq = size(qtags, 1)
        for idx in idxs
            for k in 1:nq
                y = source.qnodes[qtags[k, idx]]
                v = b(coords(y))
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
function _lvdim_layer!(Rt, Ψ, γ₁Ψ, X, Ybdry, S, D)
    for l in 1:length(Ybdry)
        y = Ybdry[l]
        v = Ψ(coords(y))
        dv = γ₁Ψ(coords(y), normal(y))
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
function _lvdim_layer_onfly!(Rt, Ψ, γ₁Ψ, X, els, list, x̂, ŵ, G, dG)
    for (idx, k) in list
        bord = boundary_element(els[idx], k)
        for (x̂i, ŵi) in zip(x̂, ŵ)
            y = bord(x̂i)
            jac = jacobian(bord, x̂i)
            w = _integration_measure(jac) * ŵi
            n = _normal(jac)
            # `dG` reads the source normal, so the kernels need a node, not a bare point
            qy = QuadratureNode(y, w, n)
            v = Ψ(y)
            dv = γ₁Ψ(y, n)
            @inbounds for i in eachindex(X)
                s = G(X[i], qy) * w
                d = dG(X[i], qy) * w
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
function _lvdim_element_R(
        ctx, ::Type{Eltype}, γ₀Ψ, γ₁Ψ, σ, b, near, patch_by_type, owners_by_type,
        need_layer_corr,
    ) where {Eltype}
    op, msh, source = ctx.op, ctx.msh, ctx.source
    X, μ = view(ctx.target, near), view(ctx.green_multiplier, near)
    # `Eltype`, not `Float64`: the contract does not promise `b` is real-valued
    Rt = Matrix{Eltype}(undef, ctx.nb, length(near))
    Gv, Gs, Gd = ctx.volume_kernel, ctx.layer_kernels...
    # the free term `μ(x)σ_β(x)` seeds `Rt`, hence `=` and not `+=`
    for i in eachindex(X)
        v = σ(coords(X[i]))
        @inbounds @simd for β in axes(Rt, 1)
            Rt[β, i] = μ[i] * v[β]
        end
    end
    _lvdim_volume!(Rt, b, X, patch_by_type, source, Gv)
    if need_layer_corr
        # `∂Ωτ` runs along `∂Ω`, so the layer potentials over it are nearly singular at the
        # targets: materialize the boundary quadrature and correct it with `bdim`
        Ybdry = Quadrature{ctx.N, Float64}(OrderedDict{DataType, ReferenceQuadrature}())
        for (E′, list) in owners_by_type
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
        _lvdim_layer!(Rt, γ₀Ψ, γ₁Ψ, X, Ybdry, S, D)
    else
        for (E′, list) in owners_by_type
            _lvdim_layer_onfly!(
                Rt, γ₀Ψ, γ₁Ψ, X,
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
const PatchFace = Tuple{DataType, Int, Int}

function _local_patch_boundary(patch_by_type, conns)
    nfaces = Nf = 0
    for (E′, idxs) in patch_by_type
        bdi = boundary_idxs(E′)
        Nf = length(first(bdi))
        nfaces += length(idxs) * length(bdi)
    end
    nfaces == 0 && return SVector{Nf, Int}[], PatchFace[]
    return _local_patch_boundary(patch_by_type, conns, nfaces, Val(Nf))
end

function _local_patch_boundary(patch_by_type, conns, nfaces::Int, ::Val{Nf}) where {Nf}
    faces = Vector{SVector{Nf, Int}}(undef, nfaces)
    owners = Vector{PatchFace}(undef, nfaces)
    j = 1
    for (E′, idxs) in patch_by_type
        # barrier specializing the fill loop on the concrete connectivity-matrix and
        # `boundary_idxs` types (`E′` is a runtime `DataType`)
        j = _fill_patch_faces!(
            faces, owners, conns[E′], boundary_idxs(E′), E′, idxs, j, Val(Nf),
        )
    end
    perm = sortperm(faces)
    keep = Int[]
    sizehint!(keep, nfaces)
    i = 1
    while i <= nfaces
        if i < nfaces && faces[perm[i]] == faces[perm[i + 1]]
            i += 2 # interior face, shared by two patch elements: drop both
        else
            push!(keep, perm[i])
            i += 1
        end
    end
    return faces[keep], owners[keep]
end

@inline function _fill_patch_faces!(
        faces, owners, conn::AbstractMatrix, bdi, E′, idxs, j, ::Val{Nf},
    ) where {Nf}
    for el in idxs
        for (k, face) in enumerate(bdi)
            faces[j] = sort(SVector(ntuple(i -> conn[face[i], el], Val(Nf))))
            owners[j] = (E′, el, k)
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
function _lvdim_patch_boundary(patch_by_type, conns, bdry_faces)
    faces, owners = _local_patch_boundary(patch_by_type, conns)
    touches_bdry = any(in(bdry_faces), faces)
    by_type = OrderedDict{DataType, Vector{Tuple{Int, Int}}}()
    for (E′, el, k) in owners
        push!(get!(by_type, E′, Tuple{Int, Int}[]), (el, k))
    end
    return by_type, touches_bdry
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
    whole_mesh = OrderedDict{DataType, Vector{Int}}(
        E => collect(axes(conns[E], 2)) for E in element_types(msh)
    )
    bdry_faces = Set(first(_local_patch_boundary(whole_mesh, conns)))

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
        neighbors, conns, bdry_faces, bdry_qrule, bdry_x̂, bdry_ŵ, pb,
        variant = Val(kernel_variant),
        volume_kernel, layer_kernels, layer_variant, Rtype,
    )

    Is, Js, Vs = Int[], Int[], Eltype[]
    for (E, qtags) in source.etype2qtags
        els = elements(msh, E)
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        # Elements are the unit of parallelism: each one allocates everything it writes to,
        # so tasks share nothing writable. Each contributes exactly `nq * length(near)`
        # entries, so the output is sized up front and every task fills a disjoint slice — no
        # lock, and a `δV` independent of the thread count.
        counts = [nq * length(near) for near in near_list]
        offs = length(Is) .+ cumsum(counts) .- counts
        for v in (Is, Js, Vs)
            resize!(v, length(v) + sum(counts))
        end
        Threads.@threads for el in 1:ne
            _lvdim_element!(Is, Js, Vs, offs[el], ctx, E, els, qtags, el, near_list[el])
        end
    end
    return sparse(Is, Js, Vs, m, n)
end

# One element's block of `δV`, written into the caller's preallocated `(I, J, V)` starting at
# `off`. A function rather than the `@threads` body inlined, so that what it captures stays
# concretely typed and `E` — a runtime `DataType` — is specialized on once per element.
function _lvdim_element!(Is, Js, Vs, off, ctx, E, els, qtags, el, near)
    isempty(near) && return nothing
    # the terms of this element's Green identity, centred and scaled to the element
    b, γ₀Ψ, γ₁Ψ, σ =
        green_identity_terms(ctx.pb, ctx.variant, translation_and_scaling(els[el])...)
    # on a curved mesh a single patch mixes straight `LagrangeElement`s with curved
    # `ParametricElement`s, so indices must stay paired with their type
    patch_by_type = OrderedDict{DataType, Vector{Int}}()
    for (E′, idx) in ctx.neighbors[(E, el)]
        push!(get!(patch_by_type, E′, Int[]), idx)
    end
    owners_by_type, need_corr =
        _lvdim_patch_boundary(patch_by_type, ctx.conns, ctx.bdry_faces)
    Rt = _lvdim_element_R(
        ctx, ctx.Rtype, γ₀Ψ, γ₁Ψ, σ, b, near, patch_by_type, owners_by_type, need_corr,
    )
    jglob = @view qtags[:, el]
    # `b` is a *batched* basis, so the Vandermonde is one column per node
    L = vandermonde(b, (coords(q) for q in view(ctx.source, jglob)))
    # the same interpolation solve `vdim_correction` performs, on a locally computed `Rt`
    _dim_solve!(Is, Js, Vs, off, L, Rt, jglob, near)
    return nothing
end
