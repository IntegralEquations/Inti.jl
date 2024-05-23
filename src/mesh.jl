"""
    abstract type AbstractMesh{N,T}

An abstract mesh structure in dimension `N` with primite data of type `T` (e.g.
`Float64` for double precision representation).

Concrete subtypes of `AbstractMesh` should implement [`ElementIterator`](@ref)
for accessing the mesh elements.

See also: [`LagrangeMesh`](@ref)
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
    struct LagrangeMesh{N,T} <: AbstractMesh{N,T}

Unstructured mesh is defined by a set of `nodes`` (of type `SVector{N,T}`), and
a dictionary mapping element types to connectivity matrices. Each columns of a
given connectivity matrix stores the integer tags of the nodes in the mesh
comprising the element.

Additionally, the mesh contains a mapping from [`EntityKey`](@ref)s to the tags
of the elements composing the entity. This can be used to extract submeshes from
a given mesh using e.g. `view(msh,Γ)` or `msh[Γ]`, where `Γ` is a
[`Domain`](@ref).

See [`elements`](@ref) for a way to iterate over the elements of a mesh.
"""
struct LagrangeMesh{N,T} <: AbstractMesh{N,T}
    nodes::Vector{SVector{N,T}}
    # for each element type (key), return the connectivity matrix
    etype2mat::Dict{DataType,Matrix{Int}}
    # store abstract vector of elements
    etype2els::Dict{DataType,AbstractVector}
    # mapping from entity to a dict containing (etype=>tags)
    ent2etags::Dict{EntityKey,Dict{DataType,Vector{Int}}}
end

# empty constructor
function LagrangeMesh{N,T}() where {N,T}
    return LagrangeMesh{N,T}(
        SVector{N,T}[],
        Dict{DataType,Matrix{Int}}(),
        Dict{DataType,AbstractVector{<:ReferenceInterpolant}}(),
        Dict{EntityKey,Dict{DataType,Vector{Int}}}(),
    )
end

element_types(msh::LagrangeMesh) = keys(msh.etype2mat)

elements(msh::LagrangeMesh, E::DataType) = msh.etype2els[E]

function elements(msh::LagrangeMesh)
    return Iterators.flatten(elements(msh, E) for E in element_types(msh))
end

nodes(msh::LagrangeMesh)     = msh.nodes
ent2etags(msh::LagrangeMesh) = msh.ent2etags
etype2mat(msh::LagrangeMesh) = msh.etype2mat
etype2els(msh::LagrangeMesh) = msh.etype2els

"""
    connectivity(msh::AbstractMesh,E::DataType)

Return the connectivity matrix for elements of type `E` in `msh`. The integer
tags in the matrix refer to the points in `nodes(msh)`
"""
connectivity(msh::LagrangeMesh, E::DataType) = msh.etype2mat[E]

function ent2nodetags(msh::LagrangeMesh, ent::EntityKey)
    tags = Int[]
    for (E, t) in msh.ent2etags[ent]
        append!(tags, view(msh.etype2mat[E], :, t))
    end
    return tags
end

entities(msh::LagrangeMesh) = keys(msh.ent2etags)

"""
    domain(msh::AbstractMesh)

Return a [`Domain`] containing of all entities covered by the mesh.
"""
domain(msh::AbstractMesh) = Domain(entities(msh))

"""
    dom2elt(m::LagrangeMesh,Ω,E)::Vector{Int}

Compute the element indices `idxs` of the elements of type `E` composing `Ω`.
"""
function dom2elt(m::AbstractMesh, Ω::Domain, E::DataType)
    idxs = Int[]
    for k in keys(Ω)
        tags = get(m.ent2etags[k], E, Int[])
        append!(idxs, tags)
    end
    return idxs
end

function Base.getindex(msh::LagrangeMesh{N,T}, Ω::Domain) where {N,T}
    new_msh = LagrangeMesh{N,T}()
    (; nodes, etype2mat, etype2els, ent2etags) = new_msh
    foreach(k -> ent2etags[k] = Dict{DataType,Vector{Int}}(), keys(Ω))
    glob2loc = Dict{Int,Int}()
    for E in element_types(msh)
        # create new element iterator
        etype2els[E] = ElementIterator(new_msh, E)
        # create new connectivity
        connect = msh.etype2mat[E]::Matrix{Int}
        np, _ = size(connect)
        mat = Int[]
        etag_loc = 0 # tag of the element in the new mesh
        for k in keys(Ω)
            # check if parent has elements of type E for the entity
            haskey(msh.ent2etags[k], E) || continue
            etags_glob = msh.ent2etags[k][E]
            etags_loc  = get!(ent2etags[k], E, Int[])
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
            end
        end
        isempty(mat) || (etype2mat[E] = reshape(mat, np, :))
    end
    return new_msh
end

"""
    meshgen(Ω::Domain,num_elements)
    meshgen(Ω::Domain;meshsize)

Generate a `LagrangeMesh` for the domain `Ω` with `num_elements` per entity. To
specify a different number of elements per entity, `num_elements` should be a
dictionary mapping entities `e ∈ Ω` to the desired `num_elements`.

Alternatively, a `meshsize` can be passed, either as a scalar or a dictionary.
In such cases, the number of elements is computed as so as to obtain an
*average* mesh size of `meshsize`; the actual mesh size may vary significantly
for each element if the parametrization is far from uniform.

Requires the entities forming `Ω` to have an explicit parametrization.

!!! warning The quality of the generated mesh created usign `meshgen` depends on
    the quality of the underlying parametrization. For complex surfaces, you are
    better off using a proper mesher such as `gmsh`.
"""
function meshgen(Ω::Domain, num_elements::Dict)
    # extract the ambient dimension for these entities (i.e. are we in 2d or
    # 3d). Only makes sense if all entities have the same ambient dimension.
    N = ambient_dimension(first(Ω))
    @assert all(p -> ambient_dimension(p) == N, entities(Ω)) "Entities must have the same ambient dimension"
    mesh = LagrangeMesh{N,Float64}()
    meshgen!(mesh, Ω, num_elements)
    return mesh
end
function meshgen(Ω::Domain, num_elements::NTuple{<:Any,Int})
    return meshgen(Ω, Dict(e => num_elements for e in keys(Ω)))
end

"""
    meshgen!(mesh,Ω,sz)

Similar to [`meshgen`](@ref), but append entries to `mesh`.
"""
function meshgen!(msh::LagrangeMesh, Ω::Domain, dict::Dict)
    N = ambient_dimension(msh)
    d = geometric_dimension(Ω)
    @assert N == 2 "meshgen! only supports 2d meshes for now"
    for k in keys(Ω)
        if d == 1
            haskey(dict, k) || error(msg1(k))
            sz = dict[k]
            _meshgen!(msh, k, sz)
        elseif d == 2
            # mesh the boundary first, then the interior
            ent = global_get_entity(k)
            @assert length(boundary(ent)) == 4
            b1, b2, b3, b4 = boundary(ent)
            @show b1, b2, b3, b4
            # check consistency between the size of a boundary curve and its
            # opposite side
            pairs = ((b1, b3), (b2, b4))
            n = map(pairs) do (l, op_l)
                if haskey(dict, l)
                    haskey(dict, op_l) &&
                        dict[l] != dict[op_l] &&
                        error("num_elements for $l and $op_l must be the same")
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
        end
    end
    _build_connectivity!(msh)
    return msh
end

# a simple mesh generation for GeometricEntity objects containing a
# parametrization
function _meshgen!(mesh::LagrangeMesh, key::EntityKey, sz)
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
        low  = lc + (Tuple(I) .- 1) .* Δx
        high = low .+ Δx
        return ParametricElement(f, HyperRectangle(low, high))
    end |> vec
end

function _build_connectivity!(msh::LagrangeMesh{N,T}, tol = 1e-8) where {N,T}
    nodes = msh.nodes
    connect_dict = msh.etype2mat
    # first build a naive connectivity matrix where duplicate points are present
    for E in keys(msh.etype2els)
        E <: ParametricElement || continue
        connect = Int[]
        E <: SVector && continue # skip points
        # map to equivalent Meshes type depending on the ReferenceShape
        x̂ = Inti.vertices(Inti.domain(E))
        nv = length(x̂)
        map(Inti.elements(msh, E)) do el
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
function Base.length(iter::ElementIterator{E,<:LagrangeMesh}) where {E}
    tags = iter.mesh.etype2mat[E]::Matrix{Int}
    _, Nel = size(tags)
    return Nel
end
Base.size(iter::ElementIterator{E,<:LagrangeMesh}) where {E} = (length(iter),)

function Base.getindex(
    iter::ElementIterator{E,<:LagrangeMesh},
    i::Int,
) where {E<:LagrangeElement}
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
function _convert_to_2d(mesh::LagrangeMesh{3,T}) where {T}
    msh2d = LagrangeMesh{2,T}()
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

function remove_duplicate_nodes!(msh::LagrangeMesh, tol)
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
    parent::LagrangeMesh{N,T}
    domain::Domain
    # etype2etags maps E => indices of elements in parent mesh contained in the
    # submesh
    etype2etags::Dict{DataType,Vector{Int}}
    function SubMesh(mesh::LagrangeMesh{N,T}, Ω::Domain) where {N,T}
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
Base.view(m::LagrangeMesh, Ω::Domain) = SubMesh(m, Ω)
Base.view(m::LagrangeMesh, ent::EntityKey) = SubMesh(m, Domain(ent))

Base.collect(msh::SubMesh) = msh.parent[msh.domain]

ambient_dimension(::SubMesh{N}) where {N} = N

geometric_dimension(msh::AbstractMesh) = geometric_dimension(domain(msh))

domain(msh::SubMesh) = msh.domain
entities(msh::SubMesh) = entities(domain(msh))

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
        append!(tags, ent2nodetags(msh, key(ent)))
    end
    return unique!(tags)
end
nodetags(msh::LagrangeMesh) = collect(1:length(msh.nodes))

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
