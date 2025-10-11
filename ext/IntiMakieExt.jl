module IntiMakieExt

using Meshes
using Makie
import Inti
using StaticArrays

function __init__()
    return @info "Loading Inti.jl Makie extension"
end

## Coversion to Meshes.jl equivalent formates for visualization

"""
    to_meshes(obj)

Convert an object from Inti.jl to an equivalent object in Meshes.jl for
visualization. High-order (curved) elements to their low-order (flat)
counterparts.
"""
function to_meshes end

to_point(x::SVector) = Meshes.Point(x...)
# LagrangeLine
to_meshes(el::Inti.LagrangeLine) = Segment(to_point.(Inti.vertices(el))...)
# LagrangeTriangle
to_meshes(el::Inti.LagrangeTriangle) = Triangle(to_point.(Inti.vertices(el))...)
# LagrangeSquare
to_meshes(el::Inti.LagrangeSquare) = Quadrangle(to_point.(Inti.vertices(el))...)
# LagrangeTetrahedron
to_meshes(el::Inti.LagrangeTetrahedron) = Tetrahedron(to_point.(Inti.vertices(el))...)
# LagrangeCube
to_meshes(el::Inti.LagrangeCube) = Hexahedron(to_point.(Inti.vertices(el))...)

# ParametricElement gets converted to its low-order equivalent
function to_meshes(el::Inti.ParametricElement{D}) where {D}
    x̂ = Inti.vertices(D())
    v = map(el, x̂)
    lag_el = Inti.LagrangeElement{D}(v)
    return to_meshes(lag_el)
end

# AbstractMesh
const ReferenceShapeToMeshes = Dict(
    Inti.ReferenceLine() => Segment,
    Inti.ReferenceTriangle() => Triangle,
    Inti.ReferenceSquare() => Quadrangle,
    Inti.ReferenceTetrahedron() => Tetrahedron,
    Inti.ReferenceCube() => Hexahedron,
)

function to_meshes(msh::Inti.AbstractMesh)
    pts = to_point.(Inti.nodes(msh))
    connec = Vector{Connectivity}()
    for E in Inti.element_types(msh)
        E <: SVector && continue # skip points
        # map to equivalent Meshes type depending on the ReferenceShape
        T = ReferenceShapeToMeshes[Inti.domain(E)]
        idxs = Inti.vertices_idxs(E)
        npts = length(idxs)
        mat = Inti.connectivity(msh, E)
        els = map(eachcol(mat)) do col
            return connect(ntuple(i -> col[idxs[i]], npts), T)
        end
        append!(connec, els)
    end
    return SimpleMesh(pts, connec)
end

# Overload the viz function to accept Inti elements and meshes
function Meshes.viz(el::Inti.ReferenceInterpolant, args...; kwargs...)
    return viz(to_meshes(el), args...; kwargs...)
end
function Meshes.viz!(el::Inti.ReferenceInterpolant, args...; kwargs...)
    return viz!(to_meshes(el), args...; kwargs...)
end

function Meshes.viz(els::AbstractVector{<:Inti.ReferenceInterpolant}, args...; kwargs...)
    return viz([to_meshes(el) for el in els])
end
function Meshes.viz!(els::AbstractVector{<:Inti.ReferenceInterpolant}, args...; kwargs...)
    return viz!([to_meshes(el) for el in els])
end

function Meshes.viz(msh::Inti.AbstractMesh, args...; kwargs...)
    return viz(to_meshes(msh), args...; kwargs...)
end
function Meshes.viz!(msh::Inti.AbstractMesh, args...; kwargs...)
    return viz!(to_meshes(msh), args...; kwargs...)
end

end # module
