"""
    QuadratureNode{N,T<:Real}

A point in `ℝᴺ` with a `weight` for performing numerical integration. A
`QuadratureNode` can optionally store a `normal` vector.
"""
struct QuadratureNode{N,T<:Real}
    coords::SVector{N,T}
    weight::T
    normal::Union{Nothing,SVector{N,T}}
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

weight(q::QuadratureNode) = q.weight

# useful for using either a quadrature node or a just a simple point in
# `IntegralOperators`.
coords(x::Union{SVector,Tuple}) = SVector(x)

"""
    struct Quadrature{N,T} <: AbstractVector{QuadratureNode{N,T}}

A collection of [`QuadratureNode`](@ref)s used to integrate over an
[`AbstractMesh`](@ref).
"""
struct Quadrature{N,T} <: AbstractVector{QuadratureNode{N,T}}
    submesh::SubMesh{N,T}
    etype2qrule::Dict{DataType,ReferenceQuadrature}
    qnodes::Vector{QuadratureNode{N,T}}
    etype2qtags::Dict{DataType,Matrix{Int}}
end

# AbstractArray interface
Base.size(quad::Quadrature) = size(quad.qnodes)
Base.getindex(quad::Quadrature, i) = quad.qnodes[i]

ambient_dimension(quad::Quadrature{N}) where {N} = N

function Base.show(io::IO, quad::Quadrature)
    return print(io, " Quadrature with $(length(quad.qnodes)) quadrature nodes")
end

"""
    Quadrature(msh::LagrangeMesh, Ω::Domain, etype2qrule::Dict)
    Quadrature(msh::LagrangeMesh, Ω::Domain; qorder)

Construct a `Quadrature` of `Ω` using the elements in `msh`, where for each
element type `E` in `msh` the reference quadrature `q = etype2qrule[E]` is used.
If an `order` keyword is passed, a default quadrature of the desired order is
used for each element type usig [`_qrule_for_reference_shape`](@ref).

Alternatively, a [`SubMesh`](@ref) can be passed as a first argument in lieu of
the `msh` and `Ω` arguments. In this case, the quadrature is constructed using
the `domain` of the `submesh`.

For co-dimension one entities in `Ω`, the normal vector is computed and stored
in the `QuadratureNode`s.
"""
function Quadrature(submesh::SubMesh{N,T}, etype2qrule::Dict) where {N,T}
    # initialize mesh with empty fields
    quad = Quadrature{N,T}(
        submesh,
        etype2qrule,
        QuadratureNode{N,T}[],
        Dict{DataType,Matrix{Int}}(),
    )
    # loop element types and generate quadrature for each
    for E in element_types(submesh)
        els   = elements(submesh, E)
        qrule = etype2qrule[E]
        # dispatch to type-stable method
        _build_quadrature!(quad, els, qrule)
    end
    return quad
end
function Quadrature(msh::LagrangeMesh, Ω::Domain, args...; kwargs...)
    return Quadrature(view(msh, Ω), args...; kwargs...)
end

function Quadrature(submsh::SubMesh; qorder)
    etype2qrule = Dict(
        E => _qrule_for_reference_shape(domain(E), qorder) for E in element_types(submsh)
    )
    return Quadrature(submsh, etype2qrule)
end

@noinline function _build_quadrature!(
    quad,
    els::ElementIterator{E},
    qrule::ReferenceQuadrature,
) where {E}
    N = ambient_dimension(quad)
    x̂, ŵ = qrule() # nodes and weights on reference element
    num_nodes = length(ŵ)
    M = geometric_dimension(domain(E))
    codim = N - M
    istart = length(quad.qnodes) + 1
    for el in els
        # and all qnodes for that element
        for (x̂i, ŵi) in zip(x̂, ŵ)
            x = el(x̂i)
            jac = jacobian(el, x̂i)
            μ = _integration_measure(jac)
            w = μ * ŵi
            ν = codim == 1 ? _normal(jac) : nothing
            qnode = QuadratureNode(x, w, ν)
            push!(quad.qnodes, qnode)
        end
    end
    iend = length(quad.qnodes)
    @assert !haskey(quad.etype2qtags, E)
    quad.etype2qtags[E] = reshape(collect(istart:iend), num_nodes, :)
    return quad
end

"""
    domain(Q::Quadrature)

The [`Domain`](@ref) over which `Q` performs integration.
"""
domain(Q::Quadrature) = Q.submesh.domain

"""
    dom2qtags(Q::Quadrature, dom::Domain)

Given a domain, return the indices of the quadratures nodes in `Q` associated to
its quadrature.
"""
function dom2qtags(Q::Quadrature, dom::Domain)
    msh = Q.mesh
    tags = Int[]
    for E in element_types(Q)
        idxs  = dom2elt(msh, dom, E)
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
        return Fejer(; order)
        # return Fejer(; order)
    elseif ref === ReferenceSquare() || ref === :square
        qx = _qrule_for_reference_shape(ReferenceLine(), order)
        qy = qx
        return TensorProductQuadrature(qx, qy)
    elseif ref === ReferenceCube() || ref === :cube
        qx = _qrule_for_reference_shape(ReferenceLine(), order)
        qy = qz = qx
        return TensorProductQuadrature(qx, qy, qz)
    elseif ref isa ReferenceTriangle || ref === :triangle
        return Gauss(; domain = ref, order = order)
    elseif ref isa ReferenceTetrahedron || ref === :tetrahedron
        return Gauss(; domain = ref, order = order)
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
    etype_to_nearest_points(X,Y::Quadrature; tol)

For each element `el` in `Y.mesh`, return a list with the indices of all points
in `X` for which `el` is the nearest element. Ignore indices for which the
distance exceeds `tol`.
"""
function etype_to_nearest_points(X, Y::Quadrature; tol = Inf)
    if X === Y
        # when both surfaces are the same, the "near points" of an element are
        # simply its own quadrature points
        dict = Dict{DataType,Vector{Vector{Int}}}()
        for (E, idx_dofs) in Y.etype2qtags
            dict[E] = map(i -> collect(i), eachcol(idx_dofs))
        end
    else
        pts = [coords(x) for x in X]
        dict = _etype_to_nearest_points(pts, Y, tol)
    end
    return dict
end

function _etype_to_nearest_points(X, Y::Quadrature, tol = Inf)
    y = [coords(q) for q in Y]
    kdtree = KDTree(y)
    dict = Dict(j => Int[] for j in 1:length(y))
    for i in eachindex(X)
        qtag, d = nn(kdtree, X[i])
        d > tol || push!(dict[qtag], i)
    end
    # dict[j] now contains indices in X for which the j quadrature node in Y is
    # the closest. Next we reverse the map
    etype2nearlist = Dict{DataType,Vector{Vector{Int}}}()
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
