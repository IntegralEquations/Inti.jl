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
    mesh::AbstractMesh{N,T}
    etype2qrule::Dict{DataType,ReferenceQuadrature}
    qnodes::Vector{QuadratureNode{N,T}}
    etype2qtags::Dict{DataType,Matrix{Int}}
end

# AbstractArray interface
Base.size(quad::Quadrature) = size(quad.qnodes)
Base.getindex(quad::Quadrature, i) = quad.qnodes[i]

qnodes(quad::Quadrature) = quad.qnodes
qcoords(quad::Quadrature) = [q.coords for q in qnodes(quad)]
qweights(quad::Quadrature) = [q.weight for q in qnodes(quad)]
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
        els   = elements(msh, E)
        qrule = etype2qrule[E]
        # dispatch to type-stable method
        _build_quadrature!(quad, els, qrule)
    end
    return quad
end

function Quadrature(msh::AbstractMesh; qorder)
    etype2qrule =
        Dict(E => _qrule_for_reference_shape(domain(E), qorder) for E in element_types(msh))
    return Quadrature(msh, etype2qrule)
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
domain(Q::Quadrature) = domain(Q.mesh)

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

function _etype_to_nearest_points(X, Y::Quadrature, maxdist = Inf)
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
    farfield_distance(Q::Quadrature, K, tol, max_iter = 10)

Given a quadrature `Q`, an `AbstractKernel` `K`, and a tolerance `tol`, return
an estimate of the distance `d` such that the (absolute) quadrature error of the
integrand `y -> K(x,y)` is below `tol` for `x` at a distance `d` from the center
of the largest element in `Q`. The estimate is computed by finding the first
integer `n` such that the quadrature error on the largest element `τ` lies below
`tol` for points `x` satisfying `dist(x,center(τ)) = n*radius(τ)`.

Note that the desired tolerance may not be achievable if the quadrature rule is
not accurate enough, or if `τ` is not sufficiently small, and therefore a
maximum number of iterations `maxiter` is provided to avoid an infinite loops.
In such cases, it is recommended that you either increase the quadrature order,
or decrease the mesh size.

**Note**: this is obviously a heuristic, and may not be accurate in all cases.
"""
function farfield_distance(Q::Quadrature, K, tol, maxiter = 10)
    msh = mesh(Q)
    dict = Dict{DataType,Float64}()
    for E in element_types(msh)
        els = elements(msh, E)
        qrule = quadrature_rule(Q, E)
        # pick the biggest element as a reference
        qtags = etype2qtags(Q, E)
        a, i = @views findmax(j -> sum(weight, Q[qtags[:, j]]), 1:size(qtags, 2))
        dict[E] = _farfield_distance(els[i], K, qrule, tol, maxiter)
    end
    # TODO: it may be useful to return a maxdist per element type so that we can
    # be more efficient in cases where different elements of different orders
    # and sizes are used. That is why a dictionary is created here. For the time
    # being, however, we just return the maximum distance.
    return maximum(values(dict))
end

function _farfield_distance(el, K, qrule, tol, maxiter)
    x₀ = center(el) # center
    h = radius(el)  # reasonable scale
    f = (x, ŷ) -> begin
        y   = el(ŷ)
        jac = jacobian(el, ŷ)
        ν   = _normal(jac)
        τ′  = _integration_measure(jac)
        return K(x, (coords = y, normal = ν)) * τ′
    end
    τ̂ = domain(el)
    N = length(x₀)
    er = 0.0
    n = 0
    while n < maxiter
        n += 1
        # explore a few directions and pick a maximum distance
        er = 0.0
        for dir in -N:N
            iszero(dir) && continue
            k    = abs(dir)
            x    = setindex(x₀, x₀[k] + sign(N) * n * h, k)
            I, E = hcubature(ŷ -> f(x, ŷ), τ̂; atol = tol / 2)
            @assert E < tol / 2 "hcubature did not converge"
            Ia = integrate(ŷ -> f(x, ŷ), qrule)
            er = max(er, norm(Ia - I))
        end
        @debug n, er
        (er < tol / 2) && break # attained desired tolerance
    end
    msg = """failed to attained desired tolerance when computing maxdist. Your
    quadrature may not be accurate enough, or your meshsize not small enough, to
    achieve the requested tolerance on the far field."""
    er > tol / 2 && @warn msg
    return n * h
end
