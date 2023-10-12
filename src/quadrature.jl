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
    coords(q::QuadratureNode)

Return the spatial coordinates of `q`.
"""
coords(q::QuadratureNode) = q.coords

# useful for using either a quadrature node or a just a simple point in
# `IntegralOperators`.
coords(x::Union{SVector,NTuple}) = SVector(x)

"""
    normal(q::QuadratureNode)

Return the normal vector of `q`, if it exists.
"""
normal(q::QuadratureNode) = q.normal

weight(q::QuadratureNode) = q.weight

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

ambient_dimension(quad::Quadrature) = ambient_dimension(quad.mesh)

function Base.show(io::IO, quad::Quadrature)
    return print(io, " Quadrature with $(length(quad.qnodes)) quadrature nodes")
end

"""
    Quadrature(msh::AbstractMesh,etype2qrule::Dict)
    Quadrature(msh::AbstractMesh;qorder)

Construct a `Quadrature` of `msh`, where for each element type `E` of `msh` the
reference quadrature `q = etype2qrule[E]` is used. If an `order` keyword is
passed, a default quadrature of the desired order is used for each element type
usig [`_qrule_for_reference_shape`](@ref).

For co-dimension 1 elements, the normal vector is computed and stored in the
`QuadratureNode`s.
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
