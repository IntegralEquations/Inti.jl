module IntiMeshesExt

using Meshes
import Inti
using StaticArrays

function __init__()
    @info "Loading Inti.jl Meshes extension"
end

## Coversion to Meshes.jl equivalent formates for visualization

# LagrangeLine
to_meshes(el::Inti.LagrangeLine) = Segment(Point.(Inti.vertices(el))...)
# LagrangeTriangle
to_meshes(el::Inti.LagrangeTriangle) = Triangle(Point.(Inti.vertices(el))...)
# LagrangeSquare
to_meshes(el::Inti.LagrangeSquare) = Quadrangle(Point.(Inti.vertices(el))...)
# LagrangeTetrahedron
to_meshes(el::Inti.LagrangeTetrahedron) = Tetrahedron(Point.(Inti.vertices(el))...)
# LagrangeCube
to_meshes(el::Inti.LagrangeCube) = Hexahedron(Point.(Inti.vertices(el))...)
# AbstractMesh
function to_meshes(msh::Inti.AbstractMesh)
    pts = Point.(Inti.nodes(msh))
    connec = Vector{Connectivity}()
    for E in Inti.element_types(msh)
        E <: SVector && continue
        idxs = Inti.vertices_idxs(E)
        npts = length(idxs)
        mat = Inti.connectivity(msh, E)
        T = if E <: Inti.LagrangeLine
            Segment
        elseif E <: Inti.LagrangeTriangle
            Triangle
        elseif E <: Inti.LagrangeSquare
            Quadrangle
        elseif E <: Inti.LagrangeTetrahedron
            Tetrahedron
        elseif E <: Inti.LagrangeCube
            Hexahedron
        else
            error("Element type not supported")
        end
        els = map(eachcol(mat)) do col
            return connect(ntuple(i -> col[idxs[i]], npts), T)
        end
        append!(connec, els)
    end
    return SimpleMesh(pts, connec)
end

# Overload the viz function to accept Inti elements and meshes
function Meshes.viz(el::Inti.LagrangeElement, args...; kwargs...)
    return viz(to_meshes(el), args...; kwargs...)
end

function Meshes.viz(els::AbstractVector{<:Inti.LagrangeElement}, args...; kwargs...)
    return viz([to_meshes(el) for el in els])
end

function Meshes.viz(msh::Inti.AbstractMesh, args...; kwargs...)
    return viz(to_meshes(msh), args...; kwargs...)
end

end # module
