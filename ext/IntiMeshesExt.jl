module IntiMeshesExt

using Meshes
import GLMakie as Mke
import Inti
using StaticArrays

function __init__()
    @info "Loading Inti.jl Meshes extension"
end

## Coversion to Meshes.jl equivalent formates for visualization

"""
    to_meshes(obj)

Convert an object from Inti.jl to an equivalent object in Meshes.jl for
visualization. High-order (curved) elements to their low-order (flat)
counterparts.
"""
function to_meshes end

to_point(x::SVector) = Point(x...)
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
    return viz([to_meshes(el) for el in els], args...; kwargs...)
end
function Meshes.viz!(els::AbstractVector{<:Inti.ReferenceInterpolant}, args...; kwargs...)
    return viz!([to_meshes(el) for el in els], args...; kwargs...)
end

function Meshes.viz(msh::Inti.AbstractMesh, args...; kwargs...)
    return viz(to_meshes(msh), args...; kwargs...)
end
function Meshes.viz!(msh::Inti.AbstractMesh, args...; kwargs...)
    return viz!(to_meshes(msh), args...; kwargs...)
end


function Inti.viz_elements(els, msh)
    E = first(Inti.element_types(msh))
    Els = [Inti.elements(msh, E)[i] for (E, i) in els]
    fig, _, _ = viz(Els)
    viz!(msh; color = 0, showsegments = true,alpha=0.3)
    display(fig)
end


function Inti.viz_elements_bords(Ei, els, ell, bords, msh; quad = nothing)
    E = first(Inti.element_types(msh))
    #ell = collect(Ei[(E, 1)])[1]
    el = Inti.elements(msh, ell[1])[ell[2]]
    fig, _, _ = viz(msh; color = 0, showsegments = true,alpha=0.3)
    viz!(el; color = 0, showsegments = true,alpha=0.5)
    for (E, i) in els
        el = Inti.elements(msh, E)[i]
        viz!(el; showsegments = true, alpha=0.7)
    end
    viz!(bords;color=4,showsegments = false,segmentsize=5,segmentcolor=4)
    if !isnothing(quad)
        xs = [qnode.coords[1] for qnode in quad.qnodes]
        ys = [qnode.coords[2] for qnode in quad.qnodes]
        nx = [Inti.normal(qnode)[1] for qnode in quad.qnodes]
        ny = [Inti.normal(qnode)[2] for qnode in quad.qnodes]
        Mke.arrows!(xs, ys, nx, ny; lengthscale = 0.15)
    end
    display(fig)
end

end # module
