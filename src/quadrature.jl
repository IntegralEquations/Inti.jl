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

"""
    flip_normal(q::QuadratureNode)

Return a new `QuadratureNode` with the normal vector flipped.
"""
flip_normal(q::QuadratureNode) = QuadratureNode(q.coords, q.weight, -q.normal)

weight(q::QuadratureNode) = q.weight

translate(q::QuadratureNode, x) = QuadratureNode(coords(q) + x, weight(q), normal(q))

# useful for using either a quadrature node or a just a simple point in
# `IntegralOperators`.
coords(x::Union{SVector,Tuple}) = SVector(x)

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
struct Quadrature{N,T} <: AbstractVector{QuadratureNode{N,T}}
    mesh::AbstractMesh{N,T}
    etype2qrule::Dict{DataType,ReferenceQuadrature}
    qnodes::Vector{QuadratureNode{N,T}}
    etype2qtags::Dict{DataType,Matrix{Int}}
end

# AbstractArray interface
Base.size(quad::Quadrature) = size(quad.qnodes)
Base.getindex(quad::Quadrature, i) = quad.qnodes[i]
Base.setindex!(quad::Quadrature, q, i) = (quad.qnodes[i] = q)

qnodes(quad::Quadrature) = quad.qnodes
mesh(quad::Quadrature) = quad.mesh
etype2qtags(quad::Quadrature, E) = quad.etype2qtags[E]

quadrature_rule(quad::Quadrature, E) = quad.etype2qrule[E]
ambient_dimension(quad::Quadrature{N}) where {N} = N

function Base.show(io::IO, quad::Quadrature)
    return print(io, " Quadrature with $(length(quad.qnodes)) quadrature nodes")
end

"""
    Quadrature(msh::AbstractMesh, etype2qrule::Dict)
    Quadrature(msh::AbstractMesh; qorder)

Construct a `Quadrature` for `msh`, where for each element type `E` in `msh` the
reference quadrature `q = etype2qrule[E]` is used. If an `order` keyword is
passed, a default quadrature of the desired order is used for each element type
usig [`_qrule_for_reference_shape`](@ref).

For co-dimension one elements, the normal vector is also computed and stored in
the [`QuadratureNode`](@ref)s.
"""
function Quadrature(msh::AbstractMesh{N,T}, etype2qrule::Dict) where {N,T}
    # initialize mesh with empty fields
    quad = Quadrature{N,T}(
        msh,
        etype2qrule,
        QuadratureNode{N,T}[],
        Dict{DataType,Matrix{Int}}(),
    )
    # loop element types and generate quadrature for each
    for E in element_types(msh)
        els = elements(msh, E)
        qrule = etype2qrule[E]
        # dispatch to type-stable method
        _build_quadrature!(quad, els, qrule)
    end
    # check for entities with negative orientation and flip normal vectors if
    # present
    for ent in entities(msh)
        if (sign(tag(ent)) < 0) && (N - geometric_dimension(ent) == 1)
            @debug "Flipping normals of $ent"
            tags = dom2qtags(quad, Domain(ent))
            for i in tags
                quad[i] = flip_normal(quad[i])
            end
        end
    end
    return quad
end

# Quadrature constructor for list of volume elements for local vdim
function Quadrature(
    msh::AbstractMesh{N,T},
    elementlist::AbstractVector{E};
    qorder,
    center::SVector{N,Float64} = zero(SVector{N,Float64}),
    scale::Float64 = 1.0,
) where {N,T,E}
    if domain(E) isa Inti.ReferenceTriangle
        Q = Inti.VioreanuRokhlin(; domain = :triangle, order = qorder)
        etype2qrule = Dict(E => Q)
    elseif domain(E) isa Inti.ReferenceTetrahedron
        Q = Inti.VioreanuRokhlin(; domain = :tetrahedron, order = qorder)
        etype2qrule = Dict(E => Q)
    else
        etype2qrule = Dict(E => _qrule_for_reference_shape(domain(E), qorder))
    end

    # initialize mesh with empty fields
    quad = Quadrature{N,T}(
        msh,
        etype2qrule,
        QuadratureNode{N,T}[],
        Dict{DataType,Matrix{Int}}(),
    )
    # loop element types and generate quadrature for each
    qrule = etype2qrule[E]
    _build_quadrature!(quad, elementlist, qrule; center, scale)

    # check for entities with negative orientation and flip normal vectors if
    # present
    for ent in entities(msh)
        if (sign(tag(ent)) < 0) && (N - geometric_dimension(ent) == 1)
            @debug "Flipping normals of $ent"
            tags = dom2qtags(quad, Domain(ent))
            for i in tags
                quad[i] = flip_normal(quad[i])
            end
        end
    end
    return quad
end

function Quadrature(msh::AbstractMesh; qorder)
    etype2qrule =
        Dict(E => _qrule_for_reference_shape(domain(E), qorder) for E in element_types(msh))
    return Quadrature(msh, etype2qrule)
end

@noinline function _build_quadrature!(
    quad::Quadrature{N,T},
    els::AbstractVector{E},
    qrule::ReferenceQuadrature;
    center::SVector{N,Float64} = zero(SVector{N,Float64}),
    scale::Float64 = 1.0,
) where {E,N,T}
    #if isdefined(Main, :Infiltrator)
    #    Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)
    #end
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
            qnode = QuadratureNode((x - center) / scale, w / scale^M, ν)
            push!(quad.qnodes, qnode)
        end
    end
    iend = length(quad.qnodes)
    @assert !haskey(quad.etype2qtags, E)
    quad.etype2qtags[E] = reshape(collect(istart:iend), num_nodes, :)
    return quad
end

"""
    Quadrature(Ω::Domain; meshsize, qorder)

Construct a `Quadrature` over the domain `Ω` with a mesh of size `meshsize` and
quadrature order `qorder`.
"""
function Quadrature(Ω::Domain; meshsize, qorder)
    msh = meshgen(Ω; meshsize)
    Q = Quadrature(view(msh, Ω); qorder)
    return Q
end

"""
    domain(Q::Quadrature)

The [`Domain`](@ref) over which `Q` performs integration.
"""
domain(Q::Quadrature) = domain(Q.mesh)

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
    etype_to_nearest_points(X,Y::Quadrature; maxdist)

For each element `el` in `Y.mesh`, return a list with the indices of all points
in `X` for which `el` is the nearest element. Ignore indices for which the
distance exceeds `maxdist`.
"""
function etype_to_nearest_points(X, Y::Quadrature; maxdist = Inf)
    if X === Y
        # when both surfaces are the same, the "near points" of an element are
        # simply its own quadrature points
        dict = Dict{DataType,Vector{Vector{Int}}}()
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

"""
    etype_to_near_elements(X,Y::Quadrature; tol)

Return `Nl = [[el in Y.mesh && dist(x, el) ≤ tol] for x in X]`
"""
function etype_to_near_elements(X, Y::Quadrature; tol)
    y = [coords(q) for q in Y]
    tree = BallTree(y)
    etype2nearlist = Dict{DataType,Vector{Set{Int}}}()
    for (E, Q) in Y.etype2qtags
        P, N = size(Q)
        etype2nearlist[E] = source2el = [Set{Int}() for _ in 1:length(X)]
        quad2source = [Set{Int}() for _ in 1:length(y)]
        for (l, x) in enumerate(X)
            for q in inrange(tree, x, tol)
                push!(quad2source[q], l)
            end
        end
        for n in 1:N
            for i in 1:P
                for l in quad2source[Q[i, n]]
                    push!(source2el[l], n)
                end
            end
        end
    end
    return etype2nearlist
end

"""
    near_elements(Y::Quadrature; tol)

Return `Nl = [[el_j in Y.mesh && dist(el_j, el_i) ≤ tol] for el_i in Y.mesh]`
"""
function near_elements(Y::Quadrature; tol)
    y = [coords(q) for q in Y]
    tree = BallTree(y)
    el2el = Dict{Tuple{DataType,Int},Set{Tuple{DataType,Int}}}()
    quad2el = [Set{Tuple{DataType,Int}}() for _ in 1:length(y)]
    # for each element, loop over its qnodes, find q∈y close to one of its qnodes, add the element to quad2el[q]
    # quad2el[q] is the set of elements whose qnodes are close to q 
    for (E, Q) in Y.etype2qtags
        P, N = size(Q)
        for n in 1:N
            for i in 1:P
                for q in inrange(tree, coords(qnodes(Y)[Q[i, n]]), tol)
                    push!(quad2el[q], (E, n))
                end
            end
        end
    end
    # for each element, the set of elements close to it is the 
    # union of the sets of elements close to its qnodes
    for (E, Q) in Y.etype2qtags
        P, N = size(Q)
        for n in 1:N
            el2el[(E, n)] = union([quad2el[Q[i, n]] for i in 1:P]...)
        end
    end
    return el2el
end

"""
    near_components(Y::Quadrature; tol)

Calculate the connected components of each near_elements
"""
function near_components(Y::Quadrature; tol)
    topo_neighbor = topological_neighbors(Y.mesh)
    dist_neighbor = near_elements(Y; tol)
    return Dict(E => connected_components(nl, topo_neighbor) for (E, nl) in dist_neighbor)
end

# function _geometric_center_circum_radius(Y::Quadrature, E, Q, P, N)
#     C = [(sum(1:P) do i
#             coords(qnodes(Y)[Q[i,n]]) .* weight(qnodes(Y)[Q[i,n]])
#           end) /
#          (sum(1:P) do i
#             weight(qnodes(Y)[Q[i,n]])
#           end) for n in 1:N]
#     r = [maximum(i->norm(C[n]-nodes(mesh(Y))[i]), connectivity(mesh(Y), E)[:,n]) for n in 1:N]
#     return C, r
# end

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
        V = mapreduce(lagrange_basis(E), hcat, qcoords(qrule)) |> Matrix
        ni, nel = size(mat) # number of interpolation nodes by number of elements
        for n in 1:nel
            qtags = Q.etype2qtags[E][:, n]
            itags = mat[:, n]
            area = sum(q -> weight(q), view(Q.qnodes, qtags))
            ivals[itags] .+= area .* (transpose(V) \ qvals[qtags])
            areas[itags] .+= area
        end
    end
    return ivals ./ areas
end

"""
    mean_curvature(Q::Quadrature)

Compute the `mean_curvature` at each quadrature node in `Q`.
"""
mean_curvature(Q::Quadrature) = _curvature(mean_curvature, Q)

"""
    gauss_curvature(Q::Quadrature)

Compute the `gauss_curvature` at each quadrature node in `Q`.
"""
gauss_curvature(Q::Quadrature) = _curvature(gauss_curvature, Q)

# helper function for computing curvature
function _curvature(f, Q)
    msh = mesh(Q)
    curv = zeros(length(Q))
    for (E, tags) in Q.etype2qtags
        qrule = quadrature_rule(Q, E)
        q̂, _ = qrule()
        els = elements(msh, E)
        for n in 1:size(tags, 2)
            el = els[n]
            for i in 1:size(tags, 1)
                qtag = tags[i, n]
                curv[qtag] = f(el, q̂[i])
            end
        end
    end
    return curv
end
