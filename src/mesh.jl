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
        iter = elements(msh,E)
        print(io, "\n\t $(length(iter)) elements of type ", E)
    end
    return io
end

"""
    struct ElementIterator{E,M}

Iterator for all elements of type `E` on a mesh of type `M`.

Besides the methods listed in the [iterator
iterface](https://docs.julialang.org/en/v1/manual/interfaces/) of `Julia`, some
functions also require the `getindex(iter,i::Int)` method for accessing the
`i`-th element directly.
"""
struct ElementIterator{E,M}
    mesh::M
end

"""
    elements(msh::AbstractMesh,E::DataType)

Return an iterator for all elements of type `E` on a mesh `msh`.
"""
elements(msh::AbstractMesh,E::DataType) = ElementIterator{E,typeof(msh)}(msh)

"""
    struct LagrangeMesh{N,T} <: AbstractMesh{N,T}

Data structure representing a generic mesh in an ambient space of dimension `N`,
with data of type `T`.

The `LagrangeMesh` can, in principle, store elements of any type. Those are given
as a key in the `elements` dictionary, and the value is a data structure which
is capable of reconstructing the elements. For example, for a Lagrange element
described by `p` nodes, the value is a `p×Nel` matrix of integer, where each
columns is a list of tags for the nodes of the element. The nodes are stored in
the `nodes` field.
"""
struct LagrangeMesh{N,T} <: AbstractMesh{N,T}
    nodes::Vector{SVector{N,T}}
    # for each element type (key), return the connectivity matrix
    etype2mat::Dict{DataType,Matrix{Int}}
    # mapping from entity to a dict containing (etype=>tags)
    ent2tags::Dict{AbstractEntity,Dict{DataType,Vector{Int}}}
end

# empty constructor
function LagrangeMesh{N,T}() where {N,T}
    return LagrangeMesh{N,T}(SVector{N,T}[], Dict{DataType,Matrix{Int}}(), Dict{AbstractEntity,Dict{DataType,Vector{Int}}}())
end

"""
    element_types(msh::AbstractMesh)

Return the element types present in the `msh`.
"""
element_types(msh::LagrangeMesh) = keys(msh.etype2mat)

"""
    dom2elt(m::LagrangeMesh,Ω,E)

Compute the element indices `idxs` of the elements of type `E` composing `Ω`, so
that `elements(m)[idxs]` gives all the elements of type `E` meshing `Ω`.
"""
function dom2elt(m::LagrangeMesh, Ω::Domain, E::DataType)
    idxs = Int[]
    for ent in entities(Ω)
        tags = get(m.ent2tags[ent], E, Int[])
        append!(idxs, tags)
    end
    return idxs
end

# implement the interface for ElementIterator of lagrange elements on a generic
# mesh. The elements are constructed on the flight based on the global nodes and
# the connectivity list stored
function Base.length(iter::ElementIterator{E,<:LagrangeMesh}) where {E<:LagrangeElement}
    tags = iter.mesh.etype2mat[E]::Matrix{Int}
    _, Nel = size(tags)
    return Nel
end

function Base.getindex(iter::ElementIterator{E,<:LagrangeMesh}, i::Int) where {E<:LagrangeElement}
    tags = iter.mesh.etype2mat[E]::Matrix{Int}
    node_tags = view(tags, :, i)
    vtx = view(iter.mesh.nodes, node_tags)
    el = E(vtx)
    return el
end

function Base.iterate(iter::ElementIterator{<:LagrangeElement,<:LagrangeMesh}, state=1)
    state > length(iter) && (return nothing)
    return iter[state], state + 1
end

"""
    struct SubMesh{N,T} <: AbstractMesh{N,T}

Create a view of a `parent` `LagrangeMesh` over a given `domain`.

A submesh implements the interface for `AbstractMesh`; therefore you can iterate
over elements of the submesh just like you would with a mesh.
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
Base.view(m::LagrangeMesh, ent::AbstractEntity) = SubMesh(m, Domain(ent))

ambient_dimension(::SubMesh{N}) where {N} = N

geometric_dimension(msh::SubMesh) = geometric_dimension(msh.domain)

element_types(msh::SubMesh) = keys(msh.etype2etags)

# ElementIterator for submesh
function Base.length(iter::ElementIterator{E,<:SubMesh}) where {E<:LagrangeElement}
    submesh = iter.mesh
    idxs    = submesh.etype2etags[E]::Vector{Int}
    return length(idxs)
end

function Base.getindex(iter::ElementIterator{E,<:SubMesh}, i::Int) where {E<:LagrangeElement}
    submsh = iter.mesh
    p_msh  = submsh.parent # parent mesh
    idxs   = submsh.etype2etags[E]::Vector{Int}
    iglob = idxs[i] # global index of element in parent mesh
    iter = elements(p_msh, E) # iterator over parent mesh
    return iter[iglob]
end

function Base.iterate(iter::ElementIterator{<:LagrangeElement,<:SubMesh}, state=1)
    state > length(iter) && (return nothing)
    return iter[state], state + 1
end

"""
    gmsh_import_mesh(Ω;[dim=3])

Create a `LagrangeMesh` for the entities in `Ω`. Passing `dim=2` will create a
two-dimensional mesh by projecting the original mesh onto the `x,y` plane.

!!! danger
    This function assumes that `gmsh` has been initialized, and does not handle its
    finalization.
"""
function gmsh_import_mesh end
