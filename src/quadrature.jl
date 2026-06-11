"""
    QuadratureNode{N,T<:Real}

A point in `ℝᴺ` with a `weight` for performing numerical integration. A
`QuadratureNode` can optionally store a `normal` vector.
"""
struct QuadratureNode{N, T <: Real}
    coords::SVector{N, T}
    weight::T
    normal::Union{Nothing, SVector{N, T}}
end

"""
    coords(q)

Return the spatial coordinates of `q`.
"""
function coords(x::T) where {T}
    if hasfield(T, :coords)
        return getfield(x, :coords)
    else
        error("type $T has no method nor field named `coords`.")
    end
end

"""
    normal(q)

Return the normal vector of `q`, if it exists.
"""
function normal(x::T) where {T}
    if hasfield(T, :normal)
        return getfield(x, :normal)
    else
        error("type $T has no method nor field named `normal`.")
    end
end

"""
    flip_normal(q::QuadratureNode)

Return a new `QuadratureNode` with the normal vector flipped.
"""
flip_normal(q::QuadratureNode) = QuadratureNode(q.coords, q.weight, -q.normal)

weight(q::QuadratureNode) = q.weight

# useful for using either a quadrature node or a just a simple point in
# `IntegralOperators`.
coords(x::Union{SVector, Tuple}) = SVector(x)

function Base.show(io::IO, q::QuadratureNode)
    println(io, "Quadrature node:")
    println(io, "-- coords: $(q.coords)")
    println(io, "-- normal: $(q.normal)")
    return print(io, "-- weight: $(q.weight)")
end

"""
    struct Quadrature{N,T} <: AbstractVector{QuadratureNode{N,T}}

A collection of [`QuadratureNode`](@ref)s used to integrate over an
[`AbstractMesh`](@ref).
"""
struct Quadrature{N, T} <: AbstractVector{QuadratureNode{N, T}}
    mesh::AbstractMesh{N, T}
    etype2qrule::OrderedDict{DataType, ReferenceQuadrature}
    qnodes::Vector{QuadratureNode{N, T}}
    etype2qtags::OrderedDict{DataType, Matrix{Int}}
end

# AbstractArray interface
Base.size(quad::Quadrature) = size(quad.qnodes)
Base.getindex(quad::Quadrature, i) = quad.qnodes[i]
Base.setindex!(quad::Quadrature, q, i) = (quad.qnodes[i] = q)

qnodes(quad::Quadrature) = quad.qnodes
mesh(quad::Quadrature) = quad.mesh
etype2qtags(quad::Quadrature, E) = quad.etype2qtags[E]

quadrature_rule(quad::Quadrature, E) = quad.etype2qrule[E]
ambient_dimension(::Quadrature{N}) where {N} = N

function Base.show(io::IO, quad::Quadrature)
    return print(io, " Quadrature with $(length(quad.qnodes)) quadrature nodes")
end

"""
    Quadrature(msh::AbstractMesh, etype2qrule::Dict)
    Quadrature(msh::AbstractMesh, qrule::ReferenceQuadrature)
    Quadrature(msh::AbstractMesh; qorder)

Construct a `Quadrature` for `msh`, where for each element type `E` in `msh` the
reference quadrature `q = etype2qrule[E]` is used. When a single `qrule` is
passed, it is used for all element types in `msh`.

If an `order` keyword is passed, a default quadrature of the desired order is
used for each element type using [`_qrule_for_reference_shape`](@ref).

For co-dimension one elements, the normal vector is also computed and stored in
the [`QuadratureNode`](@ref)s.
"""
function Quadrature(msh::AbstractMesh{N, T}, etype2qrule::OrderedDict) where {N, T}
    # initialize mesh with empty fields
    quad = Quadrature{N, T}(
        msh,
        etype2qrule,
        QuadratureNode{N, T}[],
        OrderedDict{DataType, Matrix{Int}}(),
    )
    # loop element types and generate quadrature for each
    for E in element_types(msh)
        els = elements(msh, E)
        ori = orientation(msh, E)
        qrule = etype2qrule[E]
        # dispatch to type-stable method
        _build_quadrature!(quad, els, ori, qrule)
    end
    return quad
end

function Quadrature(msh::AbstractMesh{N, T}, qrule::ReferenceQuadrature) where {N, T}
    etype2qrule = OrderedDict(E => qrule for E in element_types(msh))
    return Quadrature(msh, etype2qrule)
end

function Quadrature(msh::AbstractMesh; qorder)
    etype2qrule =
        OrderedDict(E => _qrule_for_reference_shape(domain(E), qorder) for E in element_types(msh))
    return Quadrature(msh, etype2qrule)
end

@noinline function _build_quadrature!(
        quad::Quadrature{N, T},
        els::AbstractVector{E},
        orientation::Vector{Int},
        qrule::ReferenceQuadrature,
    ) where {N, T, E}
    x̂ = map(x̂ -> T.(x̂), qcoords(qrule))
    ŵ = map(ŵ -> T.(ŵ), qweights(qrule))
    num_nodes = length(ŵ)
    M = geometric_dimension(domain(E))
    codim = N - M
    istart = length(quad.qnodes) + 1
    @assert length(els) == length(orientation)
    for (s, el) in zip(orientation, els)
        # and all qnodes for that element
        for (x̂i, ŵi) in zip(x̂, ŵ)
            x = el(x̂i)
            jac = jacobian(el, x̂i)
            μ = _integration_measure(jac)
            w = μ * ŵi
            ν = codim == 1 ? T.(s * _normal(jac)) : nothing
            qnode = QuadratureNode(T.(x), T.(w), ν)
            push!(quad.qnodes, qnode)
        end
    end
    iend = length(quad.qnodes)
    @assert !haskey(quad.etype2qtags, E)
    quad.etype2qtags[E] = reshape(collect(istart:iend), num_nodes, :)
    return quad
end

"""
    Quadrature(Ω::Domain; meshsize, qorder[, T = Float64])

Construct a `Quadrature` over the domain `Ω` with a mesh of size `meshsize` and quadrature
order `qorder`. The type parameter `T` controls the underlying data type; pass `T = Float32`
for single-precision.
"""
function Quadrature(Ω::Domain; meshsize, qorder, T = Float64)
    msh = meshgen(Ω; meshsize, T)
    Q = Quadrature(view(msh, Ω); qorder)
    return Q
end

"""
    domain(Q::Quadrature)

The [`Domain`](@ref) over which `Q` performs integration.
"""
domain(Q::Quadrature) = domain(Q.mesh)

entities(Q::Quadrature) = Q |> mesh |> entities

"""
    dom2qtags(Q::Quadrature, dom::Domain)

Given a domain, return the indices of the quadratures nodes in `Q` associated to
its quadrature.
"""
function dom2qtags(Q::Quadrature, dom::Domain)
    msh = Q.mesh
    tags = Int[]
    for E in element_types(msh)
        idxs = dom2elt(msh, dom, E)
        qtags = @view Q.etype2qtags[E][:, idxs]
        append!(tags, qtags)
    end
    return tags
end

"""
    _qrule_for_reference_shape(ref,order)

Given a `ref`erence shape and a desired quadrature `order`, return
an appropiate quadrature rule.
"""
function _qrule_for_reference_shape(ref, order)
    if ref === ReferenceLine() || ref === :line
        # return Fejer(; order)
        return GaussLegendre(; order)
    elseif ref === ReferenceSquare() || ref === :square
        qx = _qrule_for_reference_shape(ReferenceLine(), order)
        qy = qx
        return TensorProductQuadrature(qx, qy)
    elseif ref === ReferenceCube() || ref === :cube
        qx = _qrule_for_reference_shape(ReferenceLine(), order)
        qy = qz = qx
        return TensorProductQuadrature(qx, qy, qz)
    elseif ref isa ReferenceTriangle || ref === :triangle
        return VioreanuRokhlin(; domain = ref, order = order)
    elseif ref isa ReferenceTetrahedron || ref === :tetrahedron
        return VioreanuRokhlin(; domain = ref, order = order)
    else
        error("no appropriate quadrature rule found.")
    end
end

"""
    integrate(f,quad::Quadrature)

Compute `∑ᵢ f(qᵢ)wᵢ`, where the `qᵢ` are the quadrature nodes of `quad`,
and `wᵢ` are the quadrature weights.

Note that you must define `f(::QuadratureNode)`: use `q.coords` and `q.normal`
if you need to access the coordinate or normal vector at que quadrature node.
"""
function integrate(f, msh::Quadrature)
    return sum(q -> f(q) * q.weight, msh.qnodes)
end

"""
    etype_to_nearest_points(X,Y::Quadrature; maxdist)

For each element `el` in `Y.mesh`, return a list with the indices of all points
in `X` for which `el` is the nearest element. Ignore indices for which the
distance exceeds `maxdist`.
"""
function etype_to_nearest_points(X, Y::Quadrature; maxdist = Inf)
    if X === Y
        # when both surfaces are the same, the "near points" of an element are
        # simply its own quadrature points
        dict = OrderedDict{DataType, Vector{Vector{Int}}}()
        for (E, idx_dofs) in Y.etype2qtags
            dict[E] = map(i -> collect(i), eachcol(idx_dofs))
        end
    else
        pts = [coords(x) for x in X]
        dict = _etype_to_nearest_points(pts, Y, maxdist)
    end
    return dict
end

function _etype_to_nearest_points(X, Y::Quadrature, maxdist)
    y = [coords(q) for q in Y]
    kdtree = KDTree(y)
    dict = Dict(j => Int[] for j in 1:length(y))
    for i in eachindex(X)
        qtag, d = nn(kdtree, X[i])
        d > maxdist || push!(dict[qtag], i)
    end
    # dict[j] now contains indices in X for which the j quadrature node in Y is
    # the closest. Next we reverse the map
    etype2nearlist = OrderedDict{DataType, Vector{Vector{Int}}}()
    for (E, tags) in Y.etype2qtags
        nq, ne = size(tags)
        etype2nearlist[E] = nearlist = [Int[] for _ in 1:ne]
        for j in 1:ne # loop over each element of type E
            for q in 1:nq # loop over qnodes in the element
                qtag = tags[q, j]
                append!(nearlist[j], dict[qtag])
            end
        end
    end
    return etype2nearlist
end

"""
    quadrature_to_node_vals(Q::Quadrature, qvals::AbstractVector)

Given a vector `qvals` of scalar values at the quadrature nodes of `Q`, return a
vector `ivals` of scalar values at the interpolation nodes of `Q.mesh`.
"""
function quadrature_to_node_vals(Q::Quadrature, qvals::AbstractVector)
    msh = Q.mesh isa SubMesh ? collect(Q.mesh) : Q.mesh
    inodes = nodes(msh)
    ivals = zeros(eltype(qvals), length(inodes))
    areas = zeros(length(inodes)) # area of neighboring triangles
    for (E, mat) in etype2mat(msh)
        qrule = Q.etype2qrule[E]
        L = lagrange_basis(qrule)
        coords = reference_nodes(E)
        # precompute value of quadrature basis at the interpolation nodes
        Q2I = mapreduce(L, hcat, coords) |> transpose
        ni, nel = size(mat) # number of interpolation nodes by number of elements
        for n in 1:nel
            qtags = Q.etype2qtags[E][:, n]
            itags = mat[:, n]
            area = sum(q -> weight(q), view(Q.qnodes, qtags))
            ivals[itags] .+= area .* (Q2I * qvals[qtags])
            areas[itags] .+= area
        end
    end
    return ivals ./ areas
end

"""
    node_vals_to_quadrature(Q::Quadrature, ivals::AbstractVector)

Given a vector of `ivals` at the interpolation nodes of `Q.mesh`, return a vector of values
at the quadrature nodes of `Q`.
"""
function node_vals_to_quadrature(Q::Quadrature, ivals::AbstractVector)
    msh = Q.mesh isa SubMesh ? collect(Q.mesh) : Q.mesh
    qvals = zeros(eltype(ivals), length(Q.qnodes))
    for (E, mat) in etype2mat(msh)
        qrule = Q.etype2qrule[E]
        L = lagrange_basis(E)
        coords = qcoords(qrule)
        I2Q = mapreduce(L, hcat, coords) |> transpose
        _, nel = size(mat)
        for n in 1:nel
            qtags = Q.etype2qtags[E][:, n]
            itags = mat[:, n]
            qvals[qtags] .= (I2Q * ivals[itags])
        end
    end
    return qvals
end

"""
    principal_curvatures(Q::Quadrature; kwargs...)

Compute the `principal_curvatures` at each quadrature node in `Q`.
"""
principal_curvatures(Q::Quadrature; kwargs...) = _curvature((el, x̂) -> principal_curvatures(el, x̂; kwargs...), Q)

"""
    curvature(Q::Quadrature; kwargs...)

Compute the `curvature` at each quadrature node in `Q`.
"""
curvature(Q::Quadrature; kwargs...) = _curvature((el, x̂) -> curvature(el, x̂; kwargs...), Q)

"""
    mean_curvature(Q::Quadrature; kwargs...)

Compute the `mean_curvature` at each quadrature node in `Q`.
"""
mean_curvature(Q::Quadrature; kwargs...) = _curvature((el, x̂) -> mean_curvature(el, x̂; kwargs...), Q)

"""
    gauss_curvature(Q::Quadrature; kwargs...)

Compute the `gauss_curvature` at each quadrature node in `Q`.
"""
gauss_curvature(Q::Quadrature; kwargs...) = _curvature((el, x̂) -> gauss_curvature(el, x̂; kwargs...), Q)

# helper function for computing curvature
function _curvature(f, Q)
    msh = mesh(Q)
    isempty(Q.etype2qtags) && return []
    E1 = first(keys(Q.etype2qtags))
    q̂1, _ = quadrature_rule(Q, E1)()
    el1 = elements(msh, E1)[1]
    T = Base.promote_op(f, typeof(el1), eltype(q̂1))
    curv = Vector{T}(undef, length(Q))
    for (E, tags) in Q.etype2qtags
        qrule = quadrature_rule(Q, E)
        X̂ = vec(qcoords(qrule))
        els = elements(msh, E)
        for n in 1:size(tags, 2)
            curv[tags[:, n]] .= map(x̂ -> f(els[n], x̂), X̂)
        end
    end
    return curv
end

"""
    tangential_gradient_matrix(Q::Quadrature{N,T})

Return a sparse matrix `G` of size `(length(Q), length(Q))` with `SVector{N,T}`
entries such that `G * u` computes the surface gradient `∇_Γ u` at each
quadrature node for scalar values `u`.

The surface gradient is computed by locally interpolating `u` in parameter space
using the Lagrange basis on quadrature nodes, differentiating, and applying the
chain rule through the element parametrization:

```math
∇_Γ u = J (Jᵀ J)^{-1} ∇_{\\hat{x}} \\tilde{u}
```

where `J` is the Jacobian of the element map and `∇_{\\hat{x}} \\tilde{u}` is
the gradient of the interpolant in reference coordinates.
"""
function tangential_gradient_matrix(Q::Quadrature{N, T}) where {N, T}
    msh = mesh(Q)
    Is = Int[]
    Js = Int[]
    Vs = SVector{N, T}[]
    ntotal = sum(
        E -> length(qcoords(quadrature_rule(Q, E)))^2 * size(Q.etype2qtags[E], 2),
        element_types(msh)
    )
    sizehint!(Is, ntotal)
    sizehint!(Js, ntotal)
    sizehint!(Vs, ntotal)
    for (E, qtags_mat) in Q.etype2qtags
        _tangential_gradient_kernel!(Is, Js, Vs, Q, elements(msh, E), qtags_mat)
    end
    return sparse(Is, Js, Vs, length(Q), length(Q))
end

"""
    surface_gradient(u::AbstractVector, Q::Quadrature)

Compute the surface gradient `∇_Γ u` at each quadrature node, returning a
`Vector{SVector{N,T}}`.

See also: [`tangential_gradient_matrix`](@ref)
"""
function surface_gradient(u::AbstractVector, Q::Quadrature)
    return tangential_gradient_matrix(Q) * u
end

@noinline function _tangential_gradient_kernel!(
        Is,
        Js,
        Vs,
        Q::Quadrature{N, T},
        els::AbstractVector{E},
        qtags_mat::Matrix{Int},
    ) where {N, T, E}
    M = geometric_dimension(domain(E))
    qrule = quadrature_rule(Q, E)
    x̂_nodes = qcoords(qrule)
    nq = length(x̂_nodes)
    L = lagrange_basis(qrule)
    # precompute derivative of Lagrange basis at each reference node (shared across elements)
    # dL_rows[q][j] is the SVector{M,T} gradient of Lⱼ at x̂_q
    dL_rows = map(x̂_nodes) do x̂
        dL = ForwardDiff.jacobian(L, x̂)
        ntuple(j -> SVector{M, T}(ntuple(k -> T(dL[j, k]), M)), nq)
    end
    for n in 1:size(qtags_mat, 2)
        el = els[n]
        qtags = view(qtags_mat, :, n)
        for q in 1:nq
            J_q = SMatrix{N, M, T}(jacobian(el, x̂_nodes[q]))
            invA_q = inv(J_q' * J_q)
            B_q = J_q * invA_q
            i_global = qtags[q]
            for j in 1:nq
                coeff = B_q * dL_rows[q][j]
                push!(Is, i_global)
                push!(Js, qtags[j])
                push!(Vs, coeff)
            end
        end
    end
    return nothing
end

"""
    surface_divergence_matrix(Q::Quadrature{N,T})

Return a sparse matrix `D` of size `(length(Q), length(Q))` with `Adjoint{T, SVector{N,T}}`
entries such that `D * v` computes the surface divergence `∇_Γ ⋅ v` at each
quadrature node for vector fields `v`.
"""
function surface_divergence_matrix(Q::Quadrature{N, T}) where {N, T}
    msh = mesh(Q)
    Is = Int[]
    Js = Int[]
    Vs = Adjoint{T, SVector{N, T}}[]
    ntotal = sum(
        E -> length(qcoords(quadrature_rule(Q, E)))^2 * size(Q.etype2qtags[E], 2),
        element_types(msh)
    )
    sizehint!(Is, ntotal)
    sizehint!(Js, ntotal)
    sizehint!(Vs, ntotal)
    for (E, qtags_mat) in Q.etype2qtags
        _surface_divergence_kernel!(Is, Js, Vs, Q, elements(msh, E), qtags_mat)
    end
    return sparse(Is, Js, Vs, length(Q), length(Q))
end

"""
    surface_divergence(v::AbstractVector, Q::Quadrature)

Compute the surface divergence `∇_Γ ⋅ v` at each quadrature node, returning a
`Vector{T}`.
"""
function surface_divergence(v::AbstractVector, Q::Quadrature)
    return surface_divergence_matrix(Q) * v
end

@noinline function _surface_divergence_kernel!(
        Is, Js, Vs,
        Q::Quadrature{N, T},
        els::AbstractVector{E},
        qtags_mat::Matrix{Int},
    ) where {N, T, E}
    M = geometric_dimension(domain(E))
    qrule = quadrature_rule(Q, E)
    x̂_nodes = qcoords(qrule)
    nq = length(x̂_nodes)
    L = lagrange_basis(qrule)
    dL_rows = map(x̂_nodes) do x̂
        dL = ForwardDiff.jacobian(L, x̂)
        ntuple(j -> SVector{M, T}(ntuple(k -> T(dL[j, k]), M)), nq)
    end
    for n in 1:size(qtags_mat, 2)
        el = els[n]
        qtags = view(qtags_mat, :, n)
        for q in 1:nq
            J_q = SMatrix{N, M, T}(jacobian(el, x̂_nodes[q]))
            invA_q = inv(J_q' * J_q)
            B_q = J_q * invA_q
            i_global = qtags[q]
            for j in 1:nq
                coeff = B_q * dL_rows[q][j]
                push!(Is, i_global)
                push!(Js, qtags[j])
                push!(Vs, coeff')
            end
        end
    end
    return nothing
end

"""
    surface_laplacian_matrix(Q::Quadrature{N,T})

Return a sparse matrix `L` of size `(length(Q), length(Q))` with `T`
entries such that `L * u` computes the surface Laplacian `Δ_Γ u` at each
quadrature node for scalar fields `u`.
"""
function surface_laplacian_matrix(Q::Quadrature{N, T}) where {N, T}
    msh = mesh(Q)
    Is = Int[]
    Js = Int[]
    Vs = T[]
    ntotal = sum(
        E -> length(qcoords(quadrature_rule(Q, E)))^2 * size(Q.etype2qtags[E], 2),
        element_types(msh)
    )
    sizehint!(Is, ntotal)
    sizehint!(Js, ntotal)
    sizehint!(Vs, ntotal)
    for (E, qtags_mat) in Q.etype2qtags
        _surface_laplacian_kernel!(Is, Js, Vs, Q, elements(msh, E), qtags_mat)
    end
    return sparse(Is, Js, Vs, length(Q), length(Q))
end

"""
    surface_laplacian(u::AbstractVector, Q::Quadrature)

Compute the surface Laplacian `Δ_Γ u` at each quadrature node, returning a
`Vector{T}`.
"""
function surface_laplacian(u::AbstractVector, Q::Quadrature)
    return surface_laplacian_matrix(Q) * u
end

@noinline function _surface_laplacian_kernel!(
        Is, Js, Vs,
        Q::Quadrature{N, T},
        els::AbstractVector{E},
        qtags_mat::Matrix{Int},
    ) where {N, T, E}
    M = geometric_dimension(domain(E))
    qrule = quadrature_rule(Q, E)
    x̂_nodes = qcoords(qrule)
    nq = length(x̂_nodes)
    L = lagrange_basis(qrule)

    # Precompute gradients and hessians of Lagrange basis
    dL_rows = map(x̂_nodes) do x̂
        dL = ForwardDiff.jacobian(L, x̂)
        ntuple(j -> SVector{M, T}(ntuple(k -> T(dL[j, k]), M)), nq)
    end

    HL_rows = map(x̂_nodes) do x̂
        jac_func(x) = vec(ForwardDiff.jacobian(L, x))
        H_flat = ForwardDiff.jacobian(jac_func, x̂)
        ntuple(nq) do q_basis
            SMatrix{M, M, T}(ntuple(i -> H_flat[q_basis + ((i - 1) % M) * nq, div(i - 1, M) + 1], Val(M * M)))
        end
    end

    for n in 1:size(qtags_mat, 2)
        el = els[n]
        qtags = view(qtags_mat, :, n)
        for q in 1:nq
            x̂ = x̂_nodes[q]
            J_q = SMatrix{N, M, T}(jacobian(el, x̂))
            g = J_q' * J_q
            g_inv = inv(g)

            H_x = hessian(el, x̂)

            # Christoffel symbols Gamma^k = \sum_m g^{km} \sum_n H^x_n J_{nm}
            Gamma = ntuple(Val(M)) do k
                sum(1:M) do m
                    g_inv[k, m] * sum(1:N) do d
                        H_x[d, :, :][:, :] * J_q[d, m]
                    end
                end
            end

            i_global = qtags[q]
            for j in 1:nq
                grad_u_j = dL_rows[q][j]
                H_u_j = HL_rows[q][j]

                # \Delta_\Gamma L_j = g^{ab} ( H_u_j[a,b] - \sum_k \Gamma^k[a,b] grad_u_j[k] )
                val = sum(1:M) do a
                    sum(1:M) do b
                        g_inv[a, b] * (
                            H_u_j[a, b] - sum(1:M) do k
                                Gamma[k][a, b] * grad_u_j[k]
                            end
                        )
                    end
                end

                push!(Is, i_global)
                push!(Js, qtags[j])
                push!(Vs, val)
            end
        end
    end
    return nothing
end
