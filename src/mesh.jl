"""
    abstract type AbstractMesh{N,T}

An abstract mesh structure in dimension `N` with primite data of type `T` (e.g.
`Float64` for double precision representation).

Concrete subtypes of `AbstractMesh` should implement [`ElementIterator`](@ref)
for accessing the mesh elements.

See also: [`Mesh`](@ref)
"""
abstract type AbstractMesh{N,T} end

ambient_dimension(::AbstractMesh{N}) where {N} = N

function Base.show(io::IO, msh::AbstractMesh)
    print(io, "$(typeof(msh)) containing:")
    for E in element_types(msh)
        iter = elements(msh, E)
        print(io, "\n\t $(length(iter)) elements of type ", E)
    end
    return io
end

"""
    elements(msh::AbstractMesh [, E::DataType])

Return the elements of a `msh`. Passing and element type `E` will restricts to
elements of that type.

A common pattern to avoid type-instabilies in performance critical parts of the
code is to use a [function
barrier](https://docs.julialang.org/en/v1/manual/performance-tips/#kernel-functions),
as illustrated below:

```julia
for E in element_types(msh)
    _long_computation(elements(msh, E), args...)
end

@noinline function _long_computation(iter, args...)
    for el in iter # the type of el is known at compile time
        # do something with el
    end
end
```

where a dynamic dispatch is performed only on the element types (typically small
for a given mesh).
"""
function elements end

"""
    element_types(msh::AbstractMesh)

Return the element types present in the `msh`.
"""
function element_types end

"""
    struct Mesh{N,T} <: AbstractMesh{N,T}

Unstructured mesh defined by a set of `nodes`` (of type `SVector{N,T}`), and a
dictionary mapping element types to connectivity matrices. Each columns of a
given connectivity matrix stores the integer tags of the nodes in the mesh
comprising the element.

Additionally, the mesh contains a mapping from [`EntityKey`](@ref)s to the tags
of the elements composing the entity. This can be used to extract submeshes from
a given mesh using e.g. `view(msh,Γ)` or `msh[Γ]`, where `Γ` is a
[`Domain`](@ref).

See [`elements`](@ref) for a way to iterate over the elements of a mesh.
"""
struct Mesh{N,T} <: AbstractMesh{N,T}
    nodes::Vector{SVector{N,T}}
    # for each element type (key), return the connectivity matrix
    etype2mat::Dict{DataType,Matrix{Int}}
    # store abstract vector of elements
    etype2els::Dict{DataType,AbstractVector}
    # mapping from entity to a dict containing (etype=>tags)
    ent2etags::Dict{EntityKey,Dict{DataType,Vector{Int}}}
end

# empty constructor
function Mesh{N,T}() where {N,T}
    return Mesh{N,T}(
        SVector{N,T}[],
        Dict{DataType,Matrix{Int}}(),
        Dict{DataType,AbstractVector{<:ReferenceInterpolant}}(),
        Dict{EntityKey,Dict{DataType,Vector{Int}}}(),
    )
end

element_types(msh::Mesh) = keys(msh.etype2mat)

elements(msh::Mesh, E::DataType) = msh.etype2els[E]

# generic implementation
function elements(msh::AbstractMesh)
    return Iterators.flatten(elements(msh, E) for E in element_types(msh))
end

nodes(msh::Mesh) = msh.nodes

"""
    ent2etags(msh::AbstractMesh)

Return a dictionary mapping entities to a dictionary of element types to element
tags.
"""
ent2etags(msh::Mesh) = msh.ent2etags
etype2mat(msh::Mesh) = msh.etype2mat
etype2els(msh::Mesh) = msh.etype2els

"""
    connectivity(msh::AbstractMesh,E::DataType)

Return the connectivity matrix for elements of type `E` in `msh`. The integer
tags in the matrix refer to the points in `nodes(msh)`
"""
connectivity(msh::Mesh, E::DataType) = msh.etype2mat[E]

function ent2nodetags(msh::Mesh, ent::EntityKey)
    tags = Int[]
    for (E, t) in msh.ent2etags[ent]
        append!(tags, view(msh.etype2mat[E], :, t))
    end
    return tags
end

entities(msh::Mesh) = keys(msh.ent2etags)

"""
    domain(msh::AbstractMesh)

Return a [`Domain`] containing of all entities covered by the mesh.
"""
domain(msh::AbstractMesh) = Domain(entities(msh))

"""
    dom2elt(m::Mesh,Ω,E)::Vector{Int}

Compute the element indices `idxs` of the elements of type `E` composing `Ω`.
"""
function dom2elt(m::AbstractMesh, Ω::Domain, E::DataType)
    idxs = Int[]
    for k in keys(Ω)
        tags = get(ent2etags(m)[k], E, Int[])
        append!(idxs, tags)
    end
    return idxs
end

function Base.getindex(msh::Mesh{N,T}, Ω::Domain) where {N,T}
    new_msh = Mesh{N,T}()
    (; nodes, etype2mat, etype2els, ent2etags) = new_msh
    foreach(k -> ent2etags[k] = Dict{DataType,Vector{Int}}(), keys(Ω))
    glob2loc = Dict{Int,Int}()
    for E in element_types(msh)
        E <: Union{LagrangeElement,SVector,ParametricElement} ||
            error("unsupported element type $E")
        els = elements(msh, E)
        # create new element iterator
        els_new = etype2els[E] = E <: ParametricElement ? E[] : ElementIterator(new_msh, E)
        # create new connectivity
        connect = msh.etype2mat[E]::Matrix{Int}
        np, _ = size(connect)
        mat = Int[]
        etag_loc = 0 # tag of the element in the new mesh
        for k in keys(Ω)
            # check if parent has elements of type E for the entity
            haskey(msh.ent2etags[k], E) || continue
            etags_glob = msh.ent2etags[k][E]
            etags_loc = get!(ent2etags[k], E, Int[])
            for iglob in etags_glob
                etag_loc += 1
                # add nodes and connectivity matrix
                for jglob in view(connect, :, iglob)
                    if !haskey(glob2loc, jglob) # new node
                        push!(nodes, msh.nodes[jglob])
                        glob2loc[jglob] = length(nodes)
                    end
                    push!(mat, glob2loc[jglob]) # push local index of node
                end
                # add new tag for the element
                push!(etags_loc, etag_loc)
                # push new element if not inferable from connectivity
                isa(els_new, Vector) && push!(etype2els[E], els[iglob])
            end
        end
        isempty(mat) || (etype2mat[E] = reshape(mat, np, :))
    end
    return new_msh
end
Base.getindex(msh::Mesh, ent::EntityKey) = getindex(msh, Domain(ent))

"""
    meshgen(Ω, n; T = Float64)
    meshgen(Ω, n_dict; T = Float64)
    meshgen(Ω; meshsize, T = Float64)

Generate a `Mesh` for the domain `Ω` where each curve is meshed using
`n` elements. Passing a dictionary allows for a finer control; in such cases,
`n_dict[ent]` should return an integer for each entity `ent` in `Ω` of
`geometric_dimension` one.

Alternatively, a `meshsize` can be passed, in which case, the number of elements
is computed as so as to obtain an *average* mesh size of `meshsize`. Note that
the actual mesh size may vary significantly for each element if the
parametrization is far from uniform.

The mesh is created with primitive data of type `T`.

This function requires the entities forming `Ω` to have an explicit
parametrization.

!!! warning "Mesh quality"
    The quality of the generated mesh created using `meshgen` depends on
    the quality of the underlying parametrization. For complex surfaces, you are
    better off using a proper mesher such as `gmsh`.
"""
function meshgen(Ω::Domain, args...; T = Float64, kwargs...)
    # extract the ambient dimension for these entities (i.e. are we in 2d or
    # 3d). Only makes sense if all entities have the same ambient dimension.
    N = ambient_dimension(first(Ω))
    @assert all(p -> ambient_dimension(p) == N, entities(Ω)) "Entities must have the same ambient dimension"
    mesh = Mesh{N,T}()
    meshgen!(mesh, Ω, args...; kwargs...)
    return mesh
end

meshgen(e::EntityKey, args...; kwargs...) = meshgen(Domain(e), args...; kwargs...)

"""
    meshgen!(mesh,Ω,sz)

Similar to [`meshgen`](@ref), but append entries to `mesh`.
"""
function meshgen!(msh::Mesh, Ω::Domain, num_elements::Int)
    e1d = filter(k -> geometric_dimension(k) == 1, all_keys(Ω))
    return meshgen!(msh, Ω, Dict(e => num_elements for e in e1d))
end
function meshgen!(msh::Mesh, Ω::Domain; meshsize)
    # get all 1d entities (i.e. lines)
    e1d = filter(k -> geometric_dimension(k) == 1, all_keys(Ω))
    # normalize meshsize to a dictionary
    isa(meshsize, Real) && (meshsize = Dict(e => meshsize for e in e1d))
    isa(meshsize, Dict) || error("meshsize must be a number or a dictionary")
    # compute the length of each curve using an adaptive quadrature
    dict = Dict(k => ceil(Int, measure(k) / meshsize[k]) for k in e1d)
    # curves which are opposite sides of a surface must have the same number of
    # elements. If they differ, we take the maximum. Because there can be chains
    # of dependencies (i.e. l1 is opposite to l2 and l2 is opposite to l3), we
    # simple iterate until all are consistent.
    e2d = filter(k -> geometric_dimension(k) == 2, all_keys(Ω))
    while true
        consistent = true
        for k in e2d
            b1, b2, b3, b4 = boundary(global_get_entity(k))
            n1, n2, n3, n4 = map(b -> dict[b], (b1, b2, b3, b4))
            if !(n1 == n3 && n2 == n4)
                consistent = false
                dict[b1] = dict[b3] = max(n1, n3)
                dict[b2] = dict[b4] = max(n2, n4)
            end
        end
        consistent && break
    end
    return meshgen!(msh, Ω, dict)
end
function meshgen!(msh::Mesh, Ω::Domain, dict::Dict)
    d = geometric_dimension(Ω)
    for k in keys(Ω)
        if d == 1
            haskey(dict, k) || error(msg1(k))
            sz = dict[k]
            _meshgen!(msh, k, (sz,))
        elseif d == 2
            # mesh the boundary first, then the interior
            ent = global_get_entity(k)
            @assert length(boundary(ent)) == 4
            b1, b2, b3, b4 = boundary(ent)
            # check consistency between the size of a boundary curve and its
            # opposite side
            pairs = ((b1, b3), (b2, b4))
            n = map(pairs) do (l, op_l)
                if haskey(dict, l)
                    haskey(dict, op_l) &&
                        dict[l] != dict[op_l] &&
                        @warn "num_elements for $l and $op_l must be the same"
                    return dict[l]
                elseif haskey(dict, l_op)
                    return dict[l_op]
                else
                    error("missing num_elements entry for entity $b1")
                end
            end
            # mesh boundary (if needed)
            b1 ∈ entities(msh) || _meshgen!(msh, b1, (n[1],))
            b2 ∈ entities(msh) || _meshgen!(msh, b2, (n[2],))
            b3 ∈ entities(msh) || _meshgen!(msh, b3, (n[1],))
            b4 ∈ entities(msh) || _meshgen!(msh, b4, (n[2],))
            # mesh area
            _meshgen!(msh, k, n)
        else
            error("meshgen! not implemented on volumes")
        end
    end
    _build_connectivity!(msh)
    return msh
end

# a simple mesh generation for GeometricEntity objects containing a
# parametrization
function _meshgen!(mesh::Mesh, key::EntityKey, sz)
    ent = global_get_entity(key)
    d = domain(ent)
    hasparametrization(ent) || error("$key has no parametrization")
    @assert ambient_dimension(ent) == ambient_dimension(mesh)
    N = geometric_dimension(ent)
    @assert length(sz) == N
    # extract relevant fields and mesh the entity
    f = parametrization(ent)
    d = domain(ent)
    @assert d isa HyperRectangle "reference domain must be a HyperRectangle for meshgen"
    els = _meshgen(f, d, sz)
    # push related information to mesh
    E = eltype(els)
    vals = get!(mesh.etype2els, E, Vector{E}())
    istart = length(vals) + 1
    append!(vals, els)
    iend = length(vals)
    haskey(mesh.ent2etags, key) && @warn "$key already present in mesh"
    mesh.ent2etags[key] = Dict(E => collect(istart:iend)) # add key
    return mesh
end

"""
    _meshgen(f,d::HyperRectangle,sz)

Create `prod(sz)` elements of [`ParametricElement`](@ref) type representing the
push forward of `f` on each of the subdomains defined by a uniform cartesian
mesh of `d` of size `sz`.
"""
function _meshgen(f, d::HyperRectangle, sz::NTuple)
    lc, hc = low_corner(d), high_corner(d)
    Δx = (hc - lc) ./ sz
    map(CartesianIndices(sz)) do I
        low = lc + (Tuple(I) .- 1) .* Δx
        high = low .+ Δx
        return ParametricElement(f, HyperRectangle(low, high))
    end |> vec
end

function _build_connectivity!(msh::Mesh{N,T}, tol = 1e-8) where {N,T}
    nodes = msh.nodes
    connect_dict = msh.etype2mat
    # first build a naive connectivity matrix where duplicate points are present
    for E in keys(msh.etype2els)
        E <: ParametricElement || continue
        connect = Int[]
        E <: SVector && continue # skip points
        # map to equivalent Meshes type depending on the ReferenceShape
        x̂ = vertices(domain(E))
        nv = length(x̂)
        map(elements(msh, E)) do el
            istart = length(nodes) + 1
            for i in 1:nv
                push!(nodes, el(x̂[i]))
            end
            iend = length(nodes)
            return append!(connect, istart:iend)
        end
        connect_dict[E] = reshape(connect, nv, :)
    end
    remove_duplicate_nodes!(msh, tol)
    return msh
end

"""
    struct ElementIterator{E,M} <: AbstractVector{E}

Structure to lazily access elements of type `E` in a mesh of type `M`. This is
particularly useful for [`LagrangeElement`](@ref)s, where the information to
reconstruct the element is stored in the mesh connectivity matrix.
"""
struct ElementIterator{E,M} <: AbstractVector{E}
    mesh::M
end

ElementIterator(m, E::DataType) = ElementIterator{E,typeof(m)}(m)

# implement the interface for ElementIterator of lagrange elements on a generic
# mesh. The elements are constructed on the flight based on the global nodes and
# the connectivity list stored
function Base.length(iter::ElementIterator{E,<:Mesh}) where {E}
    tags = iter.mesh.etype2mat[E]::Matrix{Int}
    _, Nel = size(tags)
    return Nel
end
Base.size(iter::ElementIterator{E,<:Mesh}) where {E} = (length(iter),)

function Base.getindex(iter::ElementIterator{E,<:Mesh}, i::Int) where {E<:LagrangeElement}
    tags = iter.mesh.etype2mat[E]::Matrix{Int}
    node_tags = view(tags, :, i)
    vtx = view(iter.mesh.nodes, node_tags)
    el = E(vtx)
    return el
end

# convert a mesh to 2d by ignoring third component. Note that this also requires
# converting various element types to their 2d counterpart. These are needed
# because some meshers like gmsh always create three-dimensional objects, so we
# must convert after importing the mesh
function _convert_to_2d(mesh::Mesh{3,T}) where {T}
    msh2d = Mesh{2,T}()
    # create new dictionaries for elements and ent2etagsdict with 2d elements as keys
    new_etype2mat = msh2d.etype2mat
    new_ent2etags = msh2d.ent2etags
    new_etype2els = msh2d.etype2els
    for (E, tags) in mesh.etype2mat
        E2d = _convert_to_2d(E)
        new_etype2mat[E2d] = tags
        new_etype2els[E2d] = ElementIterator(msh2d, E2d)
    end
    for (ent, dict) in mesh.ent2etags
        new_dict = empty(dict)
        for (E, tags) in dict
            E2d = _convert_to_2d(E)
            new_dict[E2d] = tags
        end
        new_ent2etags[ent] = new_dict
    end
    nodes2d = [x[1:2] for x in mesh.nodes]
    append!(msh2d.nodes, nodes2d)
    return msh2d
end

function _convert_to_2d(::Type{LagrangeElement{R,N,SVector{3,T}}}) where {R,N,T}
    return LagrangeElement{R,N,SVector{2,T}}
end
_convert_to_2d(::Type{SVector{3,T}}) where {T} = SVector{2,T}

function remove_duplicate_nodes!(msh::Mesh, tol)
    nodes = copy(msh.nodes)
    new_nodes = empty!(msh.nodes)
    connect_dict = msh.etype2mat
    btree = BallTree(nodes)
    prox = inrange(btree, nodes, tol)
    old2new = Dict{Int,Int}()
    glob2loc = Dict{Int,Int}()
    for (i, jnear) in enumerate(prox)
        # favor the point with smallest index
        iglob = minimum(jnear)
        inew = if haskey(glob2loc, iglob)
            glob2loc[iglob]
        else
            push!(new_nodes, nodes[iglob])
            glob2loc[iglob] = length(new_nodes)
        end
        old2new[i] = inew
    end
    for mat in values(connect_dict)
        replace!(i -> old2new[i], mat)
    end
    return msh
end

"""
    struct SubMesh{N,T} <: AbstractMesh{N,T}

View into a `parent` mesh over a given `domain`.

A submesh implements the interface for `AbstractMesh`; therefore you can iterate
over elements of the submesh just like you would with a mesh.

Construct `SubMesh`s using `view(parent,Ω::Domain)`.
"""
struct SubMesh{N,T} <: AbstractMesh{N,T}
    parent::Mesh{N,T}
    domain::Domain
    # etype2etags maps E => indices of elements in parent mesh contained in the
    # submesh
    etype2etags::Dict{DataType,Vector{Int}}
    function SubMesh(mesh::Mesh{N,T}, Ω::Domain) where {N,T}
        etype2etags = Dict{DataType,Vector{Int}}()
        for E in element_types(mesh)
            # add the indices of the elements of type E in the submesh. Skip if
            # empty
            idxs = dom2elt(mesh, Ω, E)
            isempty(idxs) || (etype2etags[E] = idxs)
        end
        return new{N,T}(mesh, Ω, etype2etags)
    end
end

Base.view(m::Mesh, Ω::Domain) = SubMesh(m, Ω)
Base.view(m::Mesh, ent::EntityKey) = SubMesh(m, Domain(ent))

Base.collect(msh::SubMesh) = msh.parent[msh.domain]

Base.parent(msh::SubMesh) = msh.parent

ambient_dimension(::SubMesh{N}) where {N} = N

geometric_dimension(msh::AbstractMesh) = geometric_dimension(domain(msh))

domain(msh::SubMesh) = msh.domain
entities(msh::SubMesh) = entities(domain(msh))

function ent2etags(msh::SubMesh)
    par_ent2etags = ent2etags(parent(msh))
    g2l = Dict{Int,Int}() # global (parent) to local element index
    for (E, tags) in msh.etype2etags
        for (iloc, iglob) in enumerate(tags)
            g2l[iglob] = iloc
        end
    end
    new_ent2etags = empty(par_ent2etags)
    for ent in entities(msh)
        par_etags = par_ent2etags[ent]
        new_ent2etags[ent] = etags = empty(par_etags)
        for (E, tags) in par_etags
            etags[E] = [g2l[t] for t in tags]
        end
    end
    return new_ent2etags
end

element_types(msh::SubMesh) = keys(msh.etype2etags)

ent2nodetags(msh::SubMesh, ent::EntityKey) = ent2nodetags(msh.parent, ent)

function elements(msh::SubMesh, E::DataType)
    tags = msh.etype2etags[E]
    p_els = elements(msh.parent, E)
    return view(p_els, tags)
end

"""
    nodetags(msh::SubMesh)

Return the tags of the nodes in the parent mesh belonging to the submesh.
"""
function nodetags(msh::SubMesh)
    tags = Int[]
    for ent in entities(msh)
        append!(tags, ent2nodetags(msh, ent))
    end
    return unique!(tags)
end
nodetags(msh::Mesh) = collect(1:length(msh.nodes))

"""
    nodes(msh::SubMesh)

A view of the nodes of the parent mesh belonging to the submesh. The ordering is
given by the [`nodetags`](@ref) function.
"""
function nodes(msh::SubMesh)
    tags = nodetags(msh)
    return view(msh.parent.nodes, tags)
end

function connectivity(msh::SubMesh, E::DataType)
    tags = nodetags(msh) # tags of the nodes relative to the parent mesh
    g2l = Dict(zip(tags, 1:length(tags))) # global to local index
    eltags = msh.etype2etags[E] # indices of elements in submesh
    # connectity matrix
    return map(t -> g2l[t], view(msh.parent.etype2mat[E], :, eltags))
end

"""
    elements_to_near_targets(X,Y::AbstractMesh; tol)

For each element `el` of type `E` in `Y`, return the indices of the points in
`X` which are closer than `tol` to the `center` of `el`.

This function returns a dictionary where e.g. `dict[E][5] --> Vector{Int}` gives
the indices of points in `X` which are closer than `tol` to the center of the
fifth element of type `E`.

If `tol` is a `Dict`, then `tol[E]` is the tolerance for elements of type `E`.
"""
function elements_to_near_targets(
    X::AbstractVector{<:SVector{N}},
    Y::AbstractMesh{N};
    tol,
) where {N}
    @assert isa(tol, Number) || isa(tol, Dict) "tol must be a number or a dictionary mapping element types to numbers"
    # for each element type, build the list of targets close to a given element
    dict = Dict{DataType,Vector{Vector{Int}}}()
    balltree = BallTree(X)
    for E in element_types(Y)
        els = elements(Y, E)
        tol_ = isa(tol, Number) ? tol : tol[E]
        idxs = _elements_to_near_targets(balltree, els, tol_)
        dict[E] = idxs
    end
    return dict
end

@noinline function _elements_to_near_targets(balltree, els, tol)
    centers = map(center, els)
    return inrange(balltree, centers, tol)
end

"""
    Domain(f::Function, msh::AbstractMesh)

Call `Domain(f, ents)` on `ents = entities(msh).`
"""
Domain(f::Function, msh::AbstractMesh) = Domain(f, entities(msh))

"""
    target_to_near_elements(X::AbstractVector{<:SVector{N}}, Y::AbstractMesh{N};
    tol)

For each target `x` in `X`, return a vector of tuples `(E, i)` where `E` is the
type of the element in `Y` and `i` is the index of the element in `Y` such that
`x` is closer than `tol` to the center of the element.
"""
function target_to_near_elements(
    X::AbstractVector{<:SVector{N}},
    Y::AbstractMesh{N};
    tol,
) where {N}
    @assert isa(tol, Number) || isa(tol, Dict) "tol must be a number or a dictionary mapping element types to numbers"
    dict = Dict{Int,Vector{Tuple{DataType,Int}}}()
    balltree = BallTree(X)
    for E in element_types(Y)
        els = elements(Y, E)
        tol_ = isa(tol, Number) ? tol : tol[E]
        idxs = _target_to_near_elements(balltree, els, tol_)
        for (i, idx) in enumerate(idxs)
            dict[i] = get!(dict, i, Vector{Tuple{DataType,Int}}())
            for j in idx
                push!(dict[i], (E, j))
            end
        end
    end
    return dict
end

"""
    topological_neighbors(msh::Mesh, k=1)

Return the `k` neighbors of each element in `msh`. The one-neighbors are the
elements that share a common vertex with the element, `k` neighbors are the
one-neighbors of the `k-1` neighbors.

This function returns a dictionary where the key is an `(Eᵢ,i)` tuple denoting
the element `i` of type `E` in the mesh, and the value is a set of tuples
`(Eⱼ,j)` where `Eⱼ` is the type of the element and `j` its index.
"""
function topological_neighbors(msh::AbstractMesh, k = 1)
    # dictionary mapping a node index to all elements containing it. Note
    # that the elements are stored as a tuple (type, index)
    T = Tuple{DataType,Int}
    node2els = Dict{Int,Vector{T}}()
    for E in element_types(msh)
        mat = connectivity(msh, E)::Matrix{Int} # connectivity matrix
        np, Nel = size(mat)
        for n in 1:Nel
            for i in 1:np
                idx = mat[i, n]
                els = get!(node2els, idx, Vector{T}())
                push!(els, (E, n))
            end
        end
    end
    # now revert the map to get the neighbors
    one_neighbors = Dict{T,Set{T}}()
    for (_, els) in node2els
        for el in els
            nei = get!(one_neighbors, el, Set{T}())
            for el′ in els
                push!(nei, el′)
            end
        end
    end
    # Recursively compute the neighbors from the one-neighbors
    k_neighbors = deepcopy(one_neighbors)
    while k > 1
        # update neighborhood of each element
        for el in keys(one_neighbors)
            knn = k_neighbors[el]
            for el′ in copy(knn)
                union!(knn, one_neighbors[el′])
            end
        end
        k -= 1
    end
    return k_neighbors
end
