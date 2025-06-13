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
    # keep track if the element orientation should be inverted
    etype2orientation::Dict{DataType,Vector{Int}}
end

# empty constructor
function Mesh{N,T}() where {N,T}
    return Mesh{N,T}(
        SVector{N,T}[],
        Dict{DataType,Matrix{Int}}(),
        Dict{DataType,AbstractVector{<:ReferenceInterpolant}}(),
        Dict{EntityKey,Dict{DataType,Vector{Int}}}(),
        Dict{DataType,Vector{Int}}(),
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

"""
    orientation(msh::AbstractMesh,E::DataType)

Return the orientation of the elements of type `E` in `msh` (`1` if normal and `-1` if
inverted).
"""
orientation(msh::AbstractMesh, E::DataType) = msh.etype2orientation[E]

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

function build_orientation!(msh::AbstractMesh)
    # allocate the orientation vector for each element type
    for E in element_types(msh)
        E <: SVector && continue # skip points
        haskey(msh.etype2orientation, E) && (@warn "recomputing orientation for $E")
        msh.etype2orientation[E] = Vector{Int}(undef, length(elements(msh, E)))
    end
    # for each entity, set the orientation of the elements
    e2t = ent2etags(msh)
    for ent in entities(msh)
        geometric_dimension(ent) == 0 && continue # skip points
        s = sign(tag(ent))
        for (E, tags) in e2t[ent]
            for i in tags
                msh.etype2orientation[E][i] = s
            end
        end
    end
end

function Base.getindex(msh::Mesh{N,T}, Ω::Domain) where {N,T}
    new_msh = Mesh{N,T}()
    (; nodes, etype2mat, etype2els, ent2etags) = new_msh
    foreach(k -> ent2etags[k] = Dict{DataType,Vector{Int}}(), keys(Ω))
    glob2loc = Dict{Int,Int}()
    for E in element_types(msh)
        E <: Union{LagrangeElement,SVector,ParametricElement} || error()
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
    build_orientation!(new_msh)
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
            haskey(dict, k) || error("missing num_elements entry for entity $k")
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
                elseif haskey(dict, op_l)
                    return dict[op_l]
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
            error("meshgen! not implemented volumes")
        end
    end
    _build_connectivity!(msh)
    build_orientation!(msh)
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
    orientation = iter.mesh.etype2orientation[E][i]
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
    etype2orientation::Dict{DataType,Vector{Int}}
    function SubMesh(mesh::Mesh{N,T}, Ω::Domain) where {N,T}
        etype2etags = Dict{DataType,Vector{Int}}()
        for E in element_types(mesh)
            # add the indices of the elements of type E in the submesh. Skip if empty
            idxs = dom2elt(mesh, Ω, E)
            isempty(idxs) || (etype2etags[E] = idxs)
        end
        etype2orientation = Dict{DataType,Vector{Bool}}()
        submsh = new{N,T}(mesh, Ω, etype2etags, etype2orientation)
        build_orientation!(submsh)
        return submsh
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
    near_interaction_list(X,Y::AbstractMesh; tol)

For each element `el` of type `E` in `Y`, return the indices of the points in
`X` which are closer than `tol` to the `center` of `el`.

This function returns a dictionary where e.g. `dict[E][5] --> Vector{Int}` gives
the indices of points in `X` which are closer than `tol` to the center of the
fifth element of type `E`.

If `tol` is a `Dict`, then `tol[E]` is the tolerance for elements of type `E`.
"""
function near_interaction_list(
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
        idxs = _near_interaction_list(balltree, els, tol_)
        dict[E] = idxs
    end
    return dict
end

@noinline function _near_interaction_list(balltree, els, tol)
    centers = map(center, els)
    return inrange(balltree, centers, tol)
end

"""
    Domain(f::Function, msh::AbstractMesh)

Call `Domain(f, ents)` on `ents = entities(msh).`
"""
Domain(f::Function, msh::AbstractMesh) = Domain(f, entities(msh))

function node2etags(msh)
    # dictionary mapping a node index to all elements containing it. Note
    # that the elements are stored as a tuple (type, index)
    T = Vector{Int}
    node2els = Dict{Int,Vector{T}}()
    for E in Inti.element_types(msh)
        mat = Inti.connectivity(msh, E)::Matrix{Int} # connectivity matrix
        np, Nel = size(mat)
        for n in 1:Nel
            tags = mat[:, n]
            for i in tags
                etags = get!(node2els, i, Vector{T}())
                push!(etags, tags)
            end
        end
    end
    return node2els
end

function elements_containing_nodes(n2e, nodes)
    els = map(i -> n2e[i], nodes)
    return intersect(els...)
end

function curve_mesh(
    msh::Mesh{2,Float64},
    ψ,
    order::Int,
    patch_sample_num::Int;
    face_element_on_curved_surface = nothing,
)
    order > 0 || error("smoothness order must be positive")
    # implemented up to order=5 below but need interpolation routines to turn
    # this on
    order <= 4 || notimplemented()
    if isnothing(face_element_on_curved_surface)
        face_element_on_curved_surface = (arg) -> true
    end
    # Sample from the patch
    t = LinRange(0, 1, patch_sample_num)

    param_disc = ψ.(t)
    kdt = KDTree(transpose(stack(param_disc; dims = 1)))

    nbdry_els = size(
        msh.etype2mat[Inti.LagrangeElement{
            Inti.ReferenceHyperCube{1},
            2,
            SVector{2,Float64},
        }],
    )[2]
    bdry_node_idx = Vector{Int64}()
    bdry_node_param_loc = Vector{Float64}()

    uniqueidx(v) = unique(i -> v[i], eachindex(v))

    crvmsh = Inti.Mesh{2,Float64}()
    (; nodes, etype2mat, etype2els, ent2etags) = crvmsh
    foreach(k -> ent2etags[k] = Dict{DataType,Vector{Int}}(), Inti.entities(msh))
    append!(nodes, msh.nodes)

    # Re-write nodes to lay on exact boundary
    for elind in 1:nbdry_els
        local node_indices = msh.etype2mat[Inti.LagrangeElement{
            Inti.ReferenceHyperCube{1},
            2,
            SVector{2,Float64},
        }][
            :,
            elind,
        ]
        local straight_nodes = crvmsh.nodes[node_indices]

        if face_element_on_curved_surface(straight_nodes)
            idxs, dists = nn(kdt, straight_nodes)
            crvmsh.nodes[node_indices[1]] = param_disc[idxs[1]]
            crvmsh.nodes[node_indices[2]] = param_disc[idxs[2]]
            push!(bdry_node_idx, node_indices[1])
            push!(bdry_node_idx, node_indices[2])
            push!(bdry_node_param_loc, t[idxs[1]])
            push!(bdry_node_param_loc, t[idxs[2]])
        end
    end
    I = uniqueidx(bdry_node_idx)
    bdry_node_idx = bdry_node_idx[I]
    bdry_node_param_loc = bdry_node_param_loc[I]
    node_to_param = Dict(zip(bdry_node_idx, bdry_node_param_loc))

    connect_straight = Int[]
    connect_curve = Int[]
    connect_curve_bdry = Int[]
    # TODO Could use an ElementIterator for straight elements
    els_straight = []
    els_curve = []
    els_curve_bdry = []

    for E in Inti.element_types(msh)
        # The purpose of this check is to see if other element types are present in
        # the mesh, such as e.g. quads; This code errors when encountering a quad,
        # but the method can be extended to transfer straight quads to the new mesh,
        # similar to how straight simplices are transferred below.
        E <: Union{
            Inti.LagrangeElement{Inti.ReferenceSimplex{2}},
            Inti.LagrangeElement{Inti.ReferenceHyperCube{1}},
            SVector,
        } || error()
        E <: SVector && continue
        E <: Inti.LagrangeElement{Inti.ReferenceHyperCube{1}} && continue
        E_straight_bdry =
            Inti.LagrangeElement{Inti.ReferenceHyperCube{1},2,SVector{2,Float64}}
        els = Inti.elements(msh, E)
        for elind in eachindex(els)
            node_indices = msh.etype2mat[E][:, elind]
            straight_nodes = crvmsh.nodes[node_indices]

            # First determine if straight or curved
            verts_on_bdry = findall(x -> x ∈ bdry_node_idx, node_indices)
            j = length(verts_on_bdry) # j in C. Bernardi SINUM Sec. 6
            if j > 1
                append!(connect_curve, node_indices)
                node_indices_on_bdry = node_indices[verts_on_bdry]
                append!(connect_curve_bdry, node_indices_on_bdry)

                # Need parametric coordinates of curved mapping to be consistent with straight simplex nodes
                α₁ = min(
                    node_to_param[node_indices_on_bdry[1]],
                    node_to_param[node_indices_on_bdry[2]],
                )
                α₂ = max(
                    node_to_param[node_indices_on_bdry[1]],
                    node_to_param[node_indices_on_bdry[2]],
                )
                # HACK: handle wrap-around in parameter space when using a global parametrization
                if abs(α₁ - α₂) > 0.5
                    α₁ = α₂
                    α₂ = 1.0
                end
                a₁ = ψ(α₁)
                a₂ = ψ(α₂)

                ## Interpolant πₖʲ construction from Inti
                α₁hat = 0.0
                α₂hat = 1.0
                f̂ₖ = (t) -> α₁ .+ (α₂ - α₁)*t
                f̂ₖ_comp = (x) -> f̂ₖ((x[1] * α₁hat + x[2] * α₂hat)/(x[1] + x[2]))

                # l = 1 projection onto linear FE space
                πₖ¹_nodes = Inti.reference_nodes(
                    Inti.LagrangeElement{Inti.ReferenceLine,2,SVector{2,Float64}},
                )
                πₖ¹ψ_reference_nodes = Vector{SVector{2,Float64}}(undef, length(πₖ¹_nodes))
                for i in eachindex(πₖ¹_nodes)
                    πₖ¹ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ¹_nodes[i][1]))
                end
                πₖ¹ψ_reference_nodes = SVector{2}(πₖ¹ψ_reference_nodes)
                πₖ¹ψ =
                    (x) -> Inti.LagrangeElement{Inti.ReferenceLine}(πₖ¹ψ_reference_nodes)(x)

                # l = 2 projection onto quadratic FE space
                if order > 1
                    πₖ²_nodes = Inti.reference_nodes(
                        Inti.LagrangeElement{Inti.ReferenceLine,3,SVector{3,Float64}},
                    )
                    πₖ²ψ_reference_nodes =
                        Vector{SVector{2,Float64}}(undef, length(πₖ²_nodes))
                    for i in eachindex(πₖ²_nodes)
                        πₖ²ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ²_nodes[i][1]))
                    end
                    πₖ²ψ_reference_nodes = SVector{3}(πₖ²ψ_reference_nodes)
                    πₖ²ψ =
                        (x) ->
                            Inti.LagrangeElement{Inti.ReferenceLine}(πₖ²ψ_reference_nodes)(
                                x,
                            )
                end

                # l = 3 projection onto cubic FE space
                if order > 2
                    πₖ³_nodes = Inti.reference_nodes(
                        Inti.LagrangeElement{Inti.ReferenceLine,4,SVector{4,Float64}},
                    )
                    πₖ³ψ_reference_nodes =
                        Vector{SVector{2,Float64}}(undef, length(πₖ³_nodes))
                    for i in eachindex(πₖ³_nodes)
                        πₖ³ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ³_nodes[i][1]))
                    end
                    πₖ³ψ_reference_nodes = SVector{4}(πₖ³ψ_reference_nodes)
                    πₖ³ψ =
                        (x) ->
                            Inti.LagrangeElement{Inti.ReferenceLine}(πₖ³ψ_reference_nodes)(
                                x,
                            )
                end

                # l = 4 projection onto quartic FE space
                if order > 3
                    πₖ⁴_nodes = Inti.reference_nodes(
                        Inti.LagrangeElement{Inti.ReferenceLine,5,SVector{5,Float64}},
                    )
                    πₖ⁴ψ_reference_nodes =
                        Vector{SVector{2,Float64}}(undef, length(πₖ⁴_nodes))
                    for i in eachindex(πₖ⁴_nodes)
                        πₖ⁴ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁴_nodes[i][1]))
                    end
                    πₖ⁴ψ_reference_nodes = SVector{5}(πₖ⁴ψ_reference_nodes)
                    πₖ⁴ψ =
                        (x) ->
                            Inti.LagrangeElement{Inti.ReferenceLine}(πₖ⁴ψ_reference_nodes)(
                                x,
                            )
                end

                # l = 5 projection onto quintic FE space
                if order > 4
                    πₖ⁵_nodes = Inti.reference_nodes(
                        Inti.LagrangeElement{Inti.ReferenceLine,6,SVector{6,Float64}},
                    )
                    πₖ⁵ψ_reference_nodes =
                        Vector{SVector{2,Float64}}(undef, length(πₖ⁵_nodes))
                    for i in eachindex(πₖ⁵_nodes)
                        πₖ⁵ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁵_nodes[i][1]))
                    end
                    πₖ⁵ψ_reference_nodes = SVector{6}(πₖ⁵ψ_reference_nodes)
                    πₖ⁵ψ =
                        (x) ->
                            Inti.LagrangeElement{Inti.ReferenceLine}(πₖ⁵ψ_reference_nodes)(
                                x,
                            )
                end

                # Nonlinear map

                # θ = 1
                if order == 1
                    Φₖ =
                        (x) ->
                            (x[1] + x[2])^3 * (
                                ψ(f̂ₖ_comp(x)) -
                                πₖ¹ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            )
                end

                # θ = 2
                if order == 2
                    Φₖ =
                        (x) ->
                            (x[1] + x[2])^4 * (
                                ψ(f̂ₖ_comp(x)) -
                                πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^2*(
                                πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ¹ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            )
                end

                # θ = 3
                if order == 3
                    Φₖ =
                        (x) ->
                            (x[1] + x[2])^5 * (
                                ψ(f̂ₖ_comp(x)) -
                                πₖ³ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^2*(
                                πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ¹ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^3*(
                                πₖ³ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            )
                end

                # θ = 4
                if order == 4
                    Φₖ =
                        (x) ->
                            (x[1] + x[2])^6 * (
                                ψ(f̂ₖ_comp(x)) -
                                πₖ⁴ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^2*(
                                πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ¹ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^3*(
                                πₖ³ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^4*(
                                πₖ⁴ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ³ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            )
                end

                # θ = 5
                if order == 5
                    Φₖ =
                        (x) ->
                            (x[1] + x[2])^6 * (
                                ψ(f̂ₖ_comp(x)) -
                                πₖ⁴ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^2*(
                                πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ¹ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^3*(
                                πₖ³ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ²ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^4*(
                                πₖ⁴ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ³ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            ) +
                            (
                                x[1] + x[2]
                            )^5*(
                                πₖ⁵ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2])) -
                                πₖ⁴ψ((x[1] * α₁hat + x[2]*α₂hat)/(x[1] + x[2]))
                            )
                end

                # Zlamal nonlinear map
                #Φₖ_Z = (x) -> x[2]/(1 - x[1]) * (ψ(x[1] * α₁ + (1 - x[1]) * α₂) - x[1] * a₁ - (1 - x[1])*a₂)

                # Affine map
                aₖ = crvmsh.nodes[node_indices_on_bdry[1]]
                bₖ = crvmsh.nodes[setdiff(node_indices, node_indices[verts_on_bdry])[1]]
                cₖ = crvmsh.nodes[node_indices_on_bdry[2]]
                F̃ₖ =
                    (x) -> [
                        (cₖ[1] - bₖ[1])*x[1] + (aₖ[1] - bₖ[1])*x[2] + bₖ[1],
                        (cₖ[2] - bₖ[2])*x[1] + (aₖ[2] - bₖ[2])*x[2] + bₖ[2],
                    ]

                # Full transformation
                Fₖ = (x) -> F̃ₖ(x) + Φₖ(x)
                D = Inti.ReferenceTriangle
                T = SVector{2,Float64}
                el = Inti.ParametricElement{D,T}(x -> Fₖ(x))
                push!(els_curve, el)
                ψₖ = (s) -> Fₖ([1.0 - s[1], s[1]])
                L = Inti.ReferenceHyperCube{1}
                bdry_el = Inti.ParametricElement{L,T}(s -> ψₖ(s))
                push!(els_curve_bdry, bdry_el)

                # loop over entities
                Ecurve = typeof(first(els_curve))
                Ecurvebdry = typeof(first(els_curve_bdry))
                for k in Inti.entities(msh)
                    # determine if the straight (LagrangeElement) mesh element
                    # belongs to the entity and, if so, add the curved
                    # (ParametricElement) element.
                    if haskey(msh.ent2etags[k], E)
                        n_straight_vol_els = size(msh.etype2mat[E])[2]
                        if any(
                            (i) -> node_indices == msh.etype2mat[E][:, i],
                            range(1, n_straight_vol_els),
                        )
                            haskey(ent2etags[k], Ecurve) ||
                                (ent2etags[k][Ecurve] = Vector{Int64}())
                            append!(ent2etags[k][Ecurve], length(els_curve))
                        end
                    end
                    # find entity that contains straight (LagrangeElement) face
                    # element which is now being replaced by a curved
                    # (ParametricElement) face element
                    if haskey(msh.ent2etags[k], E_straight_bdry)
                        k.dim == 1 || continue
                        n_straight_bdry_els = size(msh.etype2mat[E_straight_bdry])[2]
                        straight_entity_elementind = findall(
                            (i) ->
                                sort(node_indices_on_bdry) ==
                                sort(msh.etype2mat[E_straight_bdry][:, i]),
                            range(1, n_straight_bdry_els),
                        )
                        if !isempty(straight_entity_elementind)
                            haskey(ent2etags[k], Ecurvebdry) ||
                                (ent2etags[k][Ecurvebdry] = Vector{Int64}())
                            append!(ent2etags[k][Ecurvebdry], length(els_curve_bdry))
                        end
                    end
                end

            else
                append!(connect_straight, node_indices)
                el = Inti.LagrangeElement{Inti.ReferenceSimplex{2},3,SVector{2,Float64}}(
                    straight_nodes,
                )
                push!(els_straight, el)

                for k in Inti.entities(msh)
                    # determine if the straight mesh element belongs to the entity and, if so, add.
                    if haskey(msh.ent2etags[k], E)
                        n_straight_vol_els = size(msh.etype2mat[E])[2]
                        if any(
                            (i) -> node_indices == msh.etype2mat[E][:, i],
                            range(1, n_straight_vol_els),
                        )
                            haskey(ent2etags[k], E) || (ent2etags[k][E] = Vector{Int64}())
                            append!(ent2etags[k][E], length(els_straight))
                        end
                    end
                    # Note: This code does not consider the possibility of boundary
                    # entities that are the boundary of straight simplices.  This is
                    # because of the assumption above that if j > 1 the triangle is
                    # curved.
                end
            end
        end
    end

    nv = 3 # Number of vertices for connectivity information in the volume
    nv_bdry = 2 # Number of vertices for connectivity information on the boundary
    Ecurve = typeof(first(els_curve))
    Ecurvebdry = typeof(first(els_curve_bdry))
    Estraight = Inti.LagrangeElement{Inti.ReferenceSimplex{2},3,SVector{2,Float64}} # TODO fix this to auto be a P1 element type

    crvmsh.etype2mat[Ecurve] = reshape(connect_curve, nv, :)
    crvmsh.etype2els[Ecurve] = convert(Vector{Ecurve}, els_curve)
    crvmsh.etype2orientation[Ecurve] = ones(length(els_curve))

    crvmsh.etype2mat[Estraight] = reshape(connect_straight, nv, :)
    crvmsh.etype2els[Estraight] = convert(Vector{Estraight}, els_straight)
    crvmsh.etype2orientation[Estraight] = ones(length(els_straight))

    crvmsh.etype2mat[Ecurvebdry] = reshape(connect_curve_bdry, nv_bdry, :)
    crvmsh.etype2els[Ecurvebdry] = convert(Vector{Ecurvebdry}, els_curve_bdry)
    crvmsh.etype2orientation[Ecurvebdry] = ones(length(els_curve_bdry))

    return crvmsh
end

function curve_mesh(
    msh::Mesh{3,Float64},
    ψ,
    order::Int,
    patch_sample_num::Int;
    face_element_on_curved_surface = nothing,
)
    order > 0 || error("smoothness order must be positive")
    order <= 5 || notimplemented()
    if isnothing(face_element_on_curved_surface)
        face_element_on_curved_surface = (arg) -> true
    end
    # v = (θ, ϕ)
    θ = LinRange(0, 2*π, patch_sample_num)
    ϕ = LinRange(0, 2*π, patch_sample_num)
    function ψ⁻¹(v0, p)
        F = (v, p) -> ψ(v) - p
        prob = NonlinearSolve.NonlinearProblem(F, v0, p)
        return NonlinearSolve.solve(prob, NonlinearSolve.SimpleNewtonRaphson())
    end

    chart_1 = Array{SVector{3,Float64}}(undef, length(θ)*length(ϕ))
    chart_1_cart_idxs_θ = []
    chart_1_cart_idxs_ϕ = []
    for i in eachindex(θ)
        for j in eachindex(ϕ)
            chart_1[(i-1)*length(ϕ)+j] = [k for k in ψ((θ[i], ϕ[j]))]
            push!(chart_1_cart_idxs_θ, i)
            push!(chart_1_cart_idxs_ϕ, j)
        end
    end
    chart_1_kdt = KDTree(chart_1; reorder = false)

    n2e = Inti.node2etags(msh)

    nbdry_els = size(
        msh.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2},3,SVector{3,Float64}}],
    )[2]
    chart_1_bdry_node_idx = Vector{Int64}()
    chart_1_bdry_node_param_loc = Vector{Vector{Float64}}()

    uniqueidx(v) = unique(i -> v[i], eachindex(v))

    crvmsh = Inti.Mesh{3,Float64}()
    (; nodes, etype2mat, etype2els, ent2etags) = crvmsh
    foreach(k -> ent2etags[k] = Dict{DataType,Vector{Int}}(), Inti.entities(msh))
    append!(nodes, msh.nodes)

    # Set up chart <-> node Dict
    for elind in 1:nbdry_els
        node_indices = msh.etype2mat[Inti.LagrangeElement{
            Inti.ReferenceSimplex{2},
            3,
            SVector{3,Float64},
        }][
            :,
            elind,
        ]
        straight_nodes = crvmsh.nodes[node_indices]

        if face_element_on_curved_surface(straight_nodes)
            idxs, dists = nn(chart_1_kdt, straight_nodes)
            if node_indices[1] ∉ chart_1_bdry_node_idx
                crvmsh.nodes[node_indices[1]] = chart_1[idxs[1]]
                push!(
                    chart_1_bdry_node_param_loc,
                    Vector{Float64}([
                        θ[chart_1_cart_idxs_θ[idxs[1]]],
                        ϕ[chart_1_cart_idxs_ϕ[idxs[1]]],
                    ]),
                )
                push!(chart_1_bdry_node_idx, node_indices[1])
            end
            if node_indices[2] ∉ chart_1_bdry_node_idx
                crvmsh.nodes[node_indices[2]] = chart_1[idxs[2]]
                push!(
                    chart_1_bdry_node_param_loc,
                    Vector{Float64}([
                        θ[chart_1_cart_idxs_θ[idxs[2]]],
                        ϕ[chart_1_cart_idxs_ϕ[idxs[2]]],
                    ]),
                )
                push!(chart_1_bdry_node_idx, node_indices[2])
            end
            if node_indices[3] ∉ chart_1_bdry_node_idx
                crvmsh.nodes[node_indices[3]] = chart_1[idxs[3]]
                push!(
                    chart_1_bdry_node_param_loc,
                    Vector{Float64}([
                        θ[chart_1_cart_idxs_θ[idxs[3]]],
                        ϕ[chart_1_cart_idxs_ϕ[idxs[3]]],
                    ]),
                )
                push!(chart_1_bdry_node_idx, node_indices[3])
            end
        end
    end
    chart_1_node_to_param = Dict(zip(chart_1_bdry_node_idx, chart_1_bdry_node_param_loc))

    connect_straight = Int[]
    connect_curve = Int[]
    connect_curve_bdry = Int[]
    # TODO Could use an ElementIterator for straight elements
    els_straight = []
    els_curve = []
    els_curve_bdry = []

    for E in Inti.element_types(msh)
        # The purpose of this check is to see if other element types are present in
        # the mesh, such as e.g. cubes; This code errors when encountering a cube,
        # but the method can be extended to transfer straight cubes to the new mesh,
        # similar to how straight simplices are transferred below.
        E <: Union{
            Inti.LagrangeElement{Inti.ReferenceSimplex{3}},
            Inti.LagrangeElement{Inti.ReferenceSimplex{2}},
            Inti.LagrangeElement{Inti.ReferenceHyperCube{1}},
            SVector,
        } || error()
        E <: SVector && continue
        E <: Inti.LagrangeElement{Inti.ReferenceHyperCube{1}} && continue
        E <: Inti.LagrangeElement{Inti.ReferenceHyperCube{2}} && continue
        E <: Inti.LagrangeElement{Inti.ReferenceSimplex{2}} && continue
        E <: Inti.LagrangeElement{Inti.ReferenceSimplex{3},4,SVector{3,Float64}} ||
            (println(E); error())
        E_straight_bdry =
            Inti.LagrangeElement{Inti.ReferenceSimplex{2},3,SVector{3,Float64}}
        els = Inti.elements(msh, E)
        for elind in eachindex(els)
            node_indices = msh.etype2mat[E][:, elind]
            straight_nodes = crvmsh.nodes[node_indices]

            verts_on_bdry = findall(x -> x ∈ chart_1_bdry_node_idx, node_indices)
            # j in C. Bernardi SINUM Sec. 6
            j = 0
            if !isempty(verts_on_bdry)
                nverts_in_chart = length(verts_on_bdry)
                j = nverts_in_chart
            end
            if j > 1
                append!(connect_curve, node_indices)
                node_indices_on_bdry = copy(node_indices[verts_on_bdry])
                append!(connect_curve_bdry, node_indices_on_bdry)
                node_to_param = chart_1_node_to_param
                α₁ = copy(node_to_param[node_indices_on_bdry[1]])
                if nverts_in_chart >= 2
                    α₂ = copy(node_to_param[node_indices_on_bdry[2]])
                else
                    @assert false
                end
                if nverts_in_chart >= 3
                    α₃ = copy(node_to_param[node_indices_on_bdry[3]])
                else
                    # Find missing node α₃ that (non-uniquely) defines the curved face simplex containing α₁, α₂
                    candidate_els =
                        Inti.elements_containing_nodes(n2e, node_indices_on_bdry)
                    # Filter out volume elements; should be at most two face simplices remaining
                    candidate_els = candidate_els[length.(candidate_els) .== 3]
                    # Take the first face simplex; while either would work if j=2,
                    # if j=3 only one of the candidate face triangles will work, so
                    # find that one
                    if candidate_els[1][1] ∉ node_indices_on_bdry
                        p = copy(crvmsh.nodes[candidate_els[1][1]])
                    elseif candidate_els[1][2] ∉ node_indices_on_bdry
                        p = copy(crvmsh.nodes[candidate_els[1][2]])
                    elseif candidate_els[1][3] ∉ node_indices_on_bdry
                        p = copy(crvmsh.nodes[candidate_els[1][3]])
                    else
                        @assert false
                    end
                    if j == 3 && p ∉ straight_nodes
                        if candidate_els[2][1] ∉ node_indices_on_bdry
                            p = copy(crvmsh.nodes[candidate_els[2][1]])
                        elseif candidate_els[2][2] ∉ node_indices_on_bdry
                            p = copy(crvmsh.nodes[candidate_els[2][2]])
                        elseif candidate_els[2][3] ∉ node_indices_on_bdry
                            p = copy(crvmsh.nodes[candidate_els[2][3]])
                        else
                            @assert false
                        end
                    end
                    res = ψ⁻¹(α₂, p)
                    if res.retcode == ReturnCode.MaxIters
                        α₃ = copy(α₂)
                        @assert j == 2
                    else
                        α₃ = Vector{Float64}(ψ⁻¹(α₂, p).u)
                        @assert norm(ψ(α₃) - p) < 10^(-14)
                    end
                end
                atol = 10^(-4)
                # Try to handle periodicity in ϕ
                if (abs(α₂[2]) < atol) && (abs(α₂[2] - α₁[2]) > π || abs(α₂[2] - α₃[2]) > π)
                    α₂[2] = 2*π
                end
                if (abs(α₃[2]) < atol) && (abs(α₃[2] - α₁[2]) > π || abs(α₃[2] - α₂[2]) > π)
                    α₃[2] = 2*π
                end
                if (abs(α₁[2]) < atol) && (abs(α₁[2] - α₂[2]) > π || abs(α₁[2] - α₃[2]) > π)
                    α₁[2] = 2*π
                end
                if (α₂[2] ≈ 2*π) && (abs(α₂[2] - α₁[2]) > π || abs(α₂[2] - α₃[2]) > π)
                    α₂[2] = 0.0
                end
                if (α₃[2] ≈ 2*π) && (abs(α₃[2] - α₁[2]) > π || abs(α₃[2] - α₂[2]) > π)
                    α₃[2] = 0.0
                end
                if (α₁[2] ≈ 2*π) && (abs(α₁[2] - α₂[2]) > π || abs(α₁[2] - α₃[2]) > π)
                    α₁[2] = 0.0
                end

                # Try to handle periodicity in ϕ
                if (abs(α₂[1]) < atol) && (abs(α₂[1] - α₁[1]) > π || abs(α₂[1] - α₃[1]) > π)
                    α₂[1] = 2*π
                end
                if (abs(α₃[1]) < atol) && (abs(α₃[1] - α₁[1]) > π || abs(α₃[1] - α₂[1]) > π)
                    α₃[1] = 2*π
                end
                if (abs(α₁[1]) < atol) && (abs(α₁[1] - α₂[1]) > π || abs(α₁[1] - α₃[1]) > π)
                    α₁[1] = 2*π
                end
                if (α₂[1] ≈ 2*π) && (abs(α₂[1] - α₁[1]) > π || abs(α₂[1] - α₃[1]) > π)
                    α₂[1] = 0.0
                end
                if (α₃[1] ≈ 2*π) && (abs(α₃[1] - α₁[1]) > π || abs(α₃[1] - α₂[1]) > π)
                    α₃[1] = 0.0
                end
                if (α₁[1] ≈ 2*π) && (abs(α₁[1] - α₂[1]) > π || abs(α₁[1] - α₃[1]) > π)
                    α₁[1] = 0.0
                end

                # Try to handle periodicity in ϕ -- case of α straddling 2π
                if (abs(α₁[2] - α₂[2]) > π) ||
                   (abs(α₂[2] - α₃[2]) > π) ||
                   (abs(α₁[2] - α₃[2]) > π)
                    if α₁[2] < π && α₂[2] < π && α₃[2] > 2*(2*π)/3
                        α₃[2] -= 2*π
                    end
                    if α₂[2] < π && α₃[2] < π && α₁[2] > 2*(2*π)/3
                        α₁[2] -= 2*π
                    end
                    if α₁[2] < π && α₃[2] < π && α₂[2] > 2*(2*π)/3
                        α₂[2] -= 2*π
                    end
                end
                if (abs(α₁[2] - α₂[2]) > π) ||
                   (abs(α₂[2] - α₃[2]) > π) ||
                   (abs(α₁[2] - α₃[2]) > π)
                    if α₁[2] > 2*(2*π)/3 && α₂[2] > 2*(2*π)/3 && α₃[2] < π
                        α₃[2] += 2*π
                    end
                    if α₂[2] > 2*(2*π)/3 && α₃[2] > 2*(2*π)/3 && α₁[2] < π
                        α₁[2] += 2*π
                    end
                    if α₁[2] > 2*(2*π)/3 && α₃[2] > 2*(2*π)/3 && α₂[2] < π
                        α₂[2] += 2*π
                    end
                end
                # Try to handle periodicity in ϕ -- case of α straddling 2π
                if (abs(α₁[1] - α₂[1]) > π) ||
                   (abs(α₂[1] - α₃[1]) > π) ||
                   (abs(α₁[1] - α₃[1]) > π)
                    if α₁[1] < π && α₂[1] < π && α₃[1] > 2*(2*π)/3
                        α₃[1] -= 2*π
                    end
                    if α₂[1] < π && α₃[1] < π && α₁[1] > 2*(2*π)/3
                        α₁[1] -= 2*π
                    end
                    if α₁[1] < π && α₃[1] < π && α₂[1] > 2*(2*π)/3
                        α₂[1] -= 2*π
                    end
                end
                if (abs(α₁[1] - α₂[1]) > π) ||
                   (abs(α₂[1] - α₃[1]) > π) ||
                   (abs(α₁[1] - α₃[1]) > π)
                    if α₁[1] > 2*(2*π)/3 && α₂[1] > 2*(2*π)/3 && α₃[1] < π
                        α₃[1] += 2*π
                    end
                    if α₂[1] > 2*(2*π)/3 && α₃[1] > 2*(2*π)/3 && α₁[1] < π
                        α₁[1] += 2*π
                    end
                    if α₁[1] > 2*(2*π)/3 && α₃[1] > 2*(2*π)/3 && α₂[1] < π
                        α₂[1] += 2*π
                    end
                end
                @assert (
                    (abs(α₁[1] - α₂[1]) < π/8) &&
                    (abs(α₂[1] - α₃[1]) < π/8) &&
                    (abs(α₁[1] - α₃[1]) < π/8)
                )
                if !(
                    (abs(α₁[2] - α₂[2]) < π/8) &&
                    (abs(α₂[2] - α₃[2]) < π/8) &&
                    (abs(α₁[2] - α₃[2]) < π/8)
                )
                    @warn "Chart parametrization warning at element #",
                    elind,
                    " with ",
                    j,
                    "verts on bdry, at θ ≈ ",
                    max(α₁[1], α₂[1], α₃[1])
                end
                a₁ = SVector{3,Float64}(ψ(α₁))
                a₂ = SVector{3,Float64}(ψ(α₂))
                a₃ = SVector{3,Float64}(ψ(α₃))

                # Construction of the affine map with vertices (aₖ, bₖ, cₖ, dₖ).
                # Vertices aₖ and bₖ always lay on surface. Vertex dₖ always lays in volume.
                aₖ = a₁
                bₖ = a₂
                cₖ = a₃
                atol = 10^-12
                facenodes = [a₁, a₂, a₃]
                skipnode = 0
                if all(norm.(Ref(straight_nodes[1]) .- facenodes) .> atol)
                    dₖ = straight_nodes[1]
                    skipnode = 1
                elseif all(norm.(Ref(straight_nodes[2]) .- facenodes) .> atol)
                    dₖ = straight_nodes[2]
                    skipnode = 2
                elseif all(norm.(Ref(straight_nodes[3]) .- facenodes) .> atol)
                    dₖ = straight_nodes[3]
                    skipnode = 3
                elseif all(norm.(Ref(straight_nodes[4]) .- facenodes) .> atol)
                    dₖ = straight_nodes[4]
                    skipnode = 4
                else
                    error("Uhoh")
                end
                if j == 2
                    if all(norm.(Ref(straight_nodes[1]) .- facenodes) .> atol)
                        (skipnode == 1) || (cₖ = copy(straight_nodes[1]))
                    end
                    if all(norm.(Ref(straight_nodes[2]) .- facenodes) .> atol)
                        (skipnode == 2) || (cₖ = copy(straight_nodes[2]))
                    end
                    if all(norm.(Ref(straight_nodes[3]) .- facenodes) .> atol)
                        (skipnode == 3) || (cₖ = copy(straight_nodes[3]))
                    end
                    if all(norm.(Ref(straight_nodes[4]) .- facenodes) .> atol)
                        (skipnode == 4) || (cₖ = copy(straight_nodes[4]))
                    end
                    @assert norm(cₖ - a₃) > atol
                    @assert norm(cₖ - dₖ) > atol
                    @assert norm(dₖ - a₁) > atol
                    @assert norm(dₖ - a₂) > atol
                end
                @assert !all(norm.(Ref(a₁) .- straight_nodes) .> atol)
                @assert !all(norm.(Ref(a₂) .- straight_nodes) .> atol)
                if j == 3
                    @assert !all(norm.(Ref(a₃) .- straight_nodes) .> atol)
                end

                # The following ensures an ordering of the face nodes so that
                # the resulting normal vector is properly oriented.
                if det([aₖ-dₖ bₖ-dₖ cₖ-dₖ]) < 0
                    tmp = deepcopy(α₁)
                    α₁ = deepcopy(α₂)
                    α₂ = tmp
                    a₁ = SVector{3,Float64}(ψ(α₁))
                    a₂ = SVector{3,Float64}(ψ(α₂))
                    a₃ = SVector{3,Float64}(ψ(α₃))
                    aₖ = a₁
                    bₖ = a₂
                end

                α₁hat = SVector{2,Float64}(1.0, 0.0)
                α₂hat = SVector{2,Float64}(0.0, 1.0)
                α₃hat = SVector{2,Float64}(0.0, 0.0)

                πₖ¹_nodes = Inti.reference_nodes(
                    Inti.LagrangeElement{Inti.ReferenceTriangle,3,SVector{2,Float64}},
                )
                α_reference_nodes = Vector{SVector{2,Float64}}(undef, length(πₖ¹_nodes))
                α_reference_nodes[1] = SVector{2}(α₃)
                α_reference_nodes[2] = SVector{2}(α₁)
                α_reference_nodes[3] = SVector{2}(α₂)
                α_reference_nodes = SVector{3}(α_reference_nodes)
                f̂ₖ =
                    (x) ->
                        Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(α_reference_nodes)(x)

                @assert (f̂ₖ(α₁hat) ≈ α₁) && (f̂ₖ(α₂hat) ≈ α₂) && (f̂ₖ(α₃hat) ≈ α₃)
                @assert a₁ ≈ ψ(f̂ₖ(α₁hat))
                @assert a₂ ≈ ψ(f̂ₖ(α₂hat))
                @assert a₃ ≈ ψ(f̂ₖ(α₃hat))
                @assert a₁ ≈ straight_nodes[1] ||
                        a₁ ≈ straight_nodes[2] ||
                        a₁ ≈ straight_nodes[3] ||
                        a₁ ≈ straight_nodes[4]
                @assert a₂ ≈ straight_nodes[1] ||
                        a₂ ≈ straight_nodes[2] ||
                        a₂ ≈ straight_nodes[3] ||
                        a₂ ≈ straight_nodes[4]
                if j == 3
                    @assert a₃ ≈ straight_nodes[1] ||
                            a₃ ≈ straight_nodes[2] ||
                            a₃ ≈ straight_nodes[3] ||
                            a₃ ≈ straight_nodes[4]
                end
                @assert aₖ ≈ a₁
                @assert bₖ ≈ a₂
                F̃ₖ =
                    (x) -> [
                        (aₖ[1] - dₖ[1])*x[1] +
                        (bₖ[1] - dₖ[1])*x[2] +
                        (cₖ[1] - dₖ[1])*x[3] +
                        dₖ[1],
                        (aₖ[2] - dₖ[2])*x[1] +
                        (bₖ[2] - dₖ[2])*x[2] +
                        (cₖ[2] - dₖ[2])*x[3] +
                        dₖ[2],
                        (aₖ[3] - dₖ[3])*x[1] +
                        (bₖ[3] - dₖ[3])*x[2] +
                        (cₖ[3] - dₖ[3])*x[3] +
                        dₖ[3],
                    ]

                # l = 1
                πₖ¹_nodes = Inti.reference_nodes(
                    Inti.LagrangeElement{
                        Inti.ReferenceTriangle,
                        binomial(2+1, 2),
                        SVector{2,Float64},
                    },
                )
                πₖ¹ψ_reference_nodes = Vector{SVector{3,Float64}}(undef, length(πₖ¹_nodes))
                for i in eachindex(πₖ¹_nodes)
                    πₖ¹ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ¹_nodes[i]))
                end
                πₖ¹ψ_reference_nodes = SVector{binomial(2+1, 2)}(πₖ¹ψ_reference_nodes)
                πₖ¹ψ =
                    (x) ->
                        Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(πₖ¹ψ_reference_nodes)(
                            x,
                        )
                #l = 2
                if order > 1
                    πₖ²_nodes = Inti.reference_nodes(
                        Inti.LagrangeElement{
                            Inti.ReferenceTriangle,
                            binomial(2+2, 2),
                            SVector{2,Float64},
                        },
                    )
                    πₖ²ψ_reference_nodes =
                        Vector{SVector{3,Float64}}(undef, length(πₖ²_nodes))
                    for i in eachindex(πₖ²_nodes)
                        πₖ²ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ²_nodes[i]))
                    end
                    πₖ²ψ_reference_nodes = SVector{binomial(2+2, 2)}(πₖ²ψ_reference_nodes)
                    πₖ²ψ =
                        (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(
                            πₖ²ψ_reference_nodes,
                        )(
                            x,
                        )
                end
                #l = 3
                if order > 2
                    πₖ³_nodes = Inti.reference_nodes(
                        Inti.LagrangeElement{
                            Inti.ReferenceTriangle,
                            binomial(2+3, 2),
                            SVector{2,Float64},
                        },
                    )
                    πₖ³ψ_reference_nodes =
                        Vector{SVector{3,Float64}}(undef, length(πₖ³_nodes))
                    for i in eachindex(πₖ³_nodes)
                        πₖ³ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ³_nodes[i]))
                    end
                    πₖ³ψ_reference_nodes = SVector{binomial(2+3, 2)}(πₖ³ψ_reference_nodes)
                    πₖ³ψ =
                        (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(
                            πₖ³ψ_reference_nodes,
                        )(
                            x,
                        )
                end
                #l = 4
                if order > 3
                    πₖ⁴_nodes = Inti.reference_nodes(
                        Inti.LagrangeElement{
                            Inti.ReferenceTriangle,
                            binomial(2+4, 2),
                            SVector{2,Float64},
                        },
                    )
                    πₖ⁴ψ_reference_nodes =
                        Vector{SVector{3,Float64}}(undef, length(πₖ⁴_nodes))
                    for i in eachindex(πₖ⁴_nodes)
                        πₖ⁴ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁴_nodes[i]))
                    end
                    πₖ⁴ψ_reference_nodes = SVector{binomial(2+4, 2)}(πₖ⁴ψ_reference_nodes)
                    πₖ⁴ψ =
                        (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(
                            πₖ⁴ψ_reference_nodes,
                        )(
                            x,
                        )
                end
                #l = 5
                if order > 4
                    πₖ⁵_nodes = Inti.reference_nodes(
                        Inti.LagrangeElement{
                            Inti.ReferenceTriangle,
                            binomial(2+5, 2),
                            SVector{2,Float64},
                        },
                    )
                    πₖ⁵ψ_reference_nodes =
                        Vector{SVector{3,Float64}}(undef, length(πₖ⁵_nodes))
                    for i in eachindex(πₖ⁵_nodes)
                        πₖ⁵ψ_reference_nodes[i] = ψ(f̂ₖ(πₖ⁵_nodes[i]))
                    end
                    πₖ⁵ψ_reference_nodes = SVector{binomial(2+5, 2)}(πₖ⁵ψ_reference_nodes)
                    πₖ⁵ψ =
                        (x) -> Inti.LagrangeElement{Inti.ReferenceSimplex{2}}(
                            πₖ⁵ψ_reference_nodes,
                        )(
                            x,
                        )
                end

                # Nonlinear map
                if j == 3
                    f̂ₖ_comp =
                        (x) -> f̂ₖ(
                            (
                                x[1] * α₁hat + x[2] * α₂hat + x[3] * α₃hat
                            )/(x[1] + x[2] + x[3]),
                        )
                    if order == 1
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2] + x[3])^3 * (
                                    ψ(f̂ₖ_comp(x)) - πₖ¹ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                )
                            )
                    end
                    if order == 2
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2] + x[3])^4 * (
                                    ψ(f̂ₖ_comp(x)) - πₖ²ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^2 * (
                                    πₖ²ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ¹ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                )
                            )
                    end
                    if order == 3
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2] + x[3])^5 * (
                                    ψ(f̂ₖ_comp(x)) - πₖ³ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^2 * (
                                    πₖ²ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ¹ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^3 * (
                                    πₖ³ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ²ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                )
                            )
                    end
                    if order == 4
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2] + x[3])^6 * (
                                    ψ(f̂ₖ_comp(x)) - πₖ⁴ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^2 * (
                                    πₖ²ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ¹ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^3 * (
                                    πₖ³ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ²ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^4 * (
                                    πₖ⁴ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ³ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                )
                            )
                    end
                    if order == 5
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2] + x[3])^7 * (
                                    ψ(f̂ₖ_comp(x)) - πₖ⁵ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^2 * (
                                    πₖ²ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ¹ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^3 * (
                                    πₖ³ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ²ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^4 * (
                                    πₖ⁴ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ³ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                ) +
                                (x[1] + x[2] + x[3])^5 * (
                                    πₖ⁵ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    ) - πₖ⁴ψ(
                                        (x[1]*α₁hat + x[2]*α₂hat + x[3]*α₃hat) /
                                        (x[1] + x[2] + x[3]),
                                    )
                                )
                            )
                    end
                else
                    f̂ₖ_comp = (x) -> f̂ₖ((x[1] * α₁hat + x[2] * α₂hat)/(x[1] + x[2]))
                    if order == 1
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2])^3 * (
                                    ψ(f̂ₖ_comp(x)) -
                                    πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                )
                            )
                    end
                    if order == 2
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2])^4 * (
                                    ψ(f̂ₖ_comp(x)) -
                                    πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^2 * (
                                    πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                )
                            )
                    end
                    if order == 3
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2])^5 * (
                                    ψ(f̂ₖ_comp(x)) -
                                    πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^2 * (
                                    πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^3 * (
                                    πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                )
                            )
                    end
                    if order == 4
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2])^6 * (
                                    ψ(f̂ₖ_comp(x)) -
                                    πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^2 * (
                                    πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^3 * (
                                    πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^4 * (
                                    πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                )
                            )
                    end
                    if order == 5
                        Φₖ =
                            (x) -> (
                                (x[1] + x[2])^7 * (
                                    ψ(f̂ₖ_comp(x)) -
                                    πₖ⁵ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^2 * (
                                    πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ¹ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^3 * (
                                    πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ²ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^4 * (
                                    πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ³ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                ) +
                                (x[1] + x[2])^5 * (
                                    πₖ⁵ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2])) -
                                    πₖ⁴ψ((x[1]*α₁hat + x[2]*α₂hat) / (x[1] + x[2]))
                                )
                            )
                    end
                end

                # Full transformation
                Fₖ = (x) -> F̃ₖ(x) + Φₖ(x)
                @assert norm(Fₖ([1.0, 0.0, 0.0]) - a₁) < atol
                @assert norm(Fₖ([0.0, 1.0, 0.0]) - a₂) < atol
                @assert norm(Fₖ([1.0, 0.0, 0.0]) - aₖ) < atol
                @assert norm(Fₖ([0.0, 1.0, 0.0]) - bₖ) < atol
                @assert norm(Fₖ([0.0, 0.0000000000000001, 1.0]) - cₖ) < atol
                @assert norm(Fₖ([0.0, 0.0000000000000001, 0.0]) - dₖ) < atol
                if j == 3
                    @assert norm(a₃ - cₖ) < atol
                    @assert norm(Fₖ([0.0, 0.0, 1.0]) - cₖ) < atol
                    @assert norm(Fₖ([0.0, 0.0, 1.0]) - a₃) < atol
                    @assert norm(Φₖ([0.0, 0.0, 0.3])) < atol
                    @assert norm(Φₖ([0.0, 0.3, 0.0])) < atol
                    @assert norm(Φₖ([0.3, 0.0, 0.0])) < atol
                    @assert norm(
                        Φₖ([0.3, 0.45, 0.25]) -
                        (ψ(f̂ₖ_comp([0.3, 0.45, 0.25])) - 0.3*a₁ - 0.45*a₂ - 0.25*a₃),
                    ) < atol
                    @assert norm(
                        Φₖ([0.55, 0.45, 0.0]) -
                        (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂),
                    ) < atol
                end
                @assert norm(Φₖ([0.0, 0.0000000000000001, 0.3])) < atol
                @assert norm(Φₖ([0.0, 0.3, 0.0])) < atol
                @assert norm(Φₖ([0.3, 0.0, 0.0])) < atol
                if j == 2
                    @assert norm(Φₖ([0.6, 0.0, 0.4])) < atol
                    @assert norm(Φₖ([0.0, 0.6, 0.4])) < atol
                    @assert norm(
                        Φₖ([0.55, 0.45, 0.0]) -
                        (ψ(f̂ₖ_comp([0.55, 0.45, 0.0])) - 0.55*a₁ - 0.45*a₂),
                    ) < atol
                end

                D = Inti.ReferenceTetrahedron
                T = SVector{3,Float64}
                el = Inti.ParametricElement{D,T}(x -> Fₖ(x))
                push!(els_curve, el)
                if j == 3
                    ψₖ = (s) -> Fₖ([s[1], s[2], 1.0 - s[1] - s[2]])
                    F = Inti.ReferenceTriangle
                    bdry_el = Inti.ParametricElement{F,T}(s -> ψₖ(s))
                    push!(els_curve_bdry, bdry_el)
                    Ecurvebdry = typeof(first(els_curve_bdry))
                end

                Ecurve = typeof(first(els_curve))
                for k in Inti.entities(msh)
                    # determine if the straight (LagrangeElement) mesh element
                    # belongs to the entity and, if so, add the curved
                    # (ParametricElement) element.
                    if haskey(msh.ent2etags[k], E)
                        n_straight_vol_els = size(msh.etype2mat[E])[2]
                        if any(
                            (i) -> sort(node_indices) == sort(msh.etype2mat[E][:, i]),
                            range(1, n_straight_vol_els),
                        )
                            haskey(ent2etags[k], Ecurve) ||
                                (ent2etags[k][Ecurve] = Vector{Int64}())
                            append!(ent2etags[k][Ecurve], length(els_curve))
                        end
                    end
                    # find entity that contains straight (LagrangeElement) face
                    # element which is now being replaced by a curved
                    # (ParametricElement) face element
                    if (j == 3) && (haskey(msh.ent2etags[k], E_straight_bdry))
                        k.dim == 2 || continue
                        n_straight_bdry_els = size(msh.etype2mat[E_straight_bdry])[2]
                        if any(
                            (i) ->
                                sort(node_indices_on_bdry) ==
                                sort(msh.etype2mat[E_straight_bdry][:, i]),
                            range(1, n_straight_bdry_els),
                        )
                            haskey(ent2etags[k], Ecurvebdry) ||
                                (ent2etags[k][Ecurvebdry] = Vector{Int64}())
                            append!(ent2etags[k][Ecurvebdry], length(els_curve_bdry))
                        end
                    end
                end
            else
                append!(connect_straight, node_indices)
                el = Inti.LagrangeElement{Inti.ReferenceSimplex{3},4,SVector{3,Float64}}(
                    straight_nodes,
                )
                push!(els_straight, el)

                for k in Inti.entities(msh)
                    # determine if the straight mesh element belongs to the entity and, if so, add.
                    if haskey(msh.ent2etags[k], E)
                        n_straight_vol_els = size(msh.etype2mat[E])[2]
                        if any(
                            (i) -> node_indices == msh.etype2mat[E][:, i],
                            range(1, n_straight_vol_els),
                        )
                            haskey(ent2etags[k], E) || (ent2etags[k][E] = Vector{Int64}())
                            append!(ent2etags[k][E], length(els_straight))
                        end
                    end
                    # Note: This code does not consider the possibility of boundary
                    # entities that are the boundary of straight simplices.  This is
                    # because of the assumption above that if j > 1 the triangle is
                    # curved.
                end
            end
        end
    end

    nv = 4 # Number of vertices for connectivity information in the volume
    nv_bdry = 3 # Number of vertices for connectivity information on the boundary
    Ecurve = typeof(first(els_curve))
    Ecurvebdry = typeof(first(els_curve_bdry))
    Estraight = Inti.LagrangeElement{Inti.ReferenceSimplex{3},4,SVector{3,Float64}} # TODO fix this to auto be a P1 element type

    crvmsh.etype2mat[Ecurve] = reshape(connect_curve, nv, :)
    crvmsh.etype2els[Ecurve] = convert(Vector{Ecurve}, els_curve)
    crvmsh.etype2orientation[Ecurve] = ones(length(els_curve))

    crvmsh.etype2mat[Estraight] = reshape(connect_straight, nv, :)
    crvmsh.etype2els[Estraight] = convert(Vector{Estraight}, els_straight)
    crvmsh.etype2orientation[Estraight] = ones(length(els_straight))

    crvmsh.etype2mat[Ecurvebdry] = reshape(connect_curve_bdry, nv_bdry, :)
    crvmsh.etype2els[Ecurvebdry] = convert(Vector{Ecurvebdry}, els_curve_bdry)
    crvmsh.etype2orientation[Ecurvebdry] = ones(length(els_curve_bdry))

    return crvmsh
end
