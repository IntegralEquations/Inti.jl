module IntiMakieExt

using Makie
import Inti
using StaticArrays

const GB = Makie.GeometryBasics

function __init__()
    return @debug "Loading Inti.jl Makie extension"
end

## Decomposition of Inti elements/meshes into triangles and segments
#
# For visualization we represent every element by its (low-order) vertices and
# decompose it into a set of triangular faces (for surface/volume elements) and a
# set of edges/segments (for the wireframe). At present high-order (curved) elements are
# rendered using their flat, low-order counterpart.

# Triangulation and edges of each reference shape, given as local vertex indices
# (1-based) into the element's `vertices`. The vertex ordering follows
# `Inti.vertices(::ReferenceShape)`.
_reference_triangles(::Inti.ReferenceLine) = NTuple{3, Int}[]
_reference_triangles(::Inti.ReferenceTriangle) = [(1, 2, 3)]
_reference_triangles(::Inti.ReferenceSquare) = [(1, 2, 3), (1, 3, 4)]
function _reference_triangles(::Inti.ReferenceTetrahedron)
    return [(1, 3, 2), (1, 2, 4), (2, 3, 4), (1, 4, 3)]
end
function _reference_triangles(::Inti.ReferenceCube)
    quads = (
        (1, 4, 3, 2), # bottom (z=0)
        (5, 6, 7, 8), # top    (z=1)
        (1, 2, 6, 5), # front  (y=0)
        (2, 3, 7, 6), # right  (x=1)
        (3, 4, 8, 7), # back   (y=1)
        (4, 1, 5, 8), # left   (x=0)
    )
    tris = NTuple{3, Int}[]
    for (a, b, c, d) in quads
        push!(tris, (a, b, c))
        push!(tris, (a, c, d))
    end
    return tris
end

_reference_segments(::Inti.ReferenceLine) = [(1, 2)]
_reference_segments(::Inti.ReferenceTriangle) = [(1, 2), (2, 3), (3, 1)]
_reference_segments(::Inti.ReferenceSquare) = [(1, 2), (2, 3), (3, 4), (4, 1)]
function _reference_segments(::Inti.ReferenceTetrahedron)
    return [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
end
function _reference_segments(::Inti.ReferenceCube)
    return [
        (1, 2), (2, 3), (3, 4), (4, 1), # bottom
        (5, 6), (6, 7), (7, 8), (8, 5), # top
        (1, 5), (2, 6), (3, 7), (4, 8), # verticals
    ]
end

_to_point(x::SVector) = GB.Point(x)

# Normalize the supported plotting inputs to a common form: a vector of node
# coordinates plus, for each element, the global indices of its vertices and its
# reference domain.
function _vertex_topology(msh::Inti.AbstractMesh)
    pts = _to_point.(Inti.nodes(msh))
    elt_vertices = Vector{Vector{Int}}()
    elt_domains = Vector{Inti.ReferenceShape}()
    for E in Inti.element_types(msh)
        E <: SVector && continue # skip point "elements"
        d = Inti.domain(E)
        idxs = Inti.vertices_idxs(E)
        mat = Inti.connectivity(msh, E)
        for col in eachcol(mat)
            push!(elt_vertices, [col[i] for i in idxs])
            push!(elt_domains, d)
        end
    end
    return pts, elt_vertices, elt_domains
end

function _vertex_topology(els::AbstractVector{<:Inti.ReferenceInterpolant})
    pts = GB.Point[]
    elt_vertices = Vector{Vector{Int}}()
    elt_domains = Vector{Inti.ReferenceShape}()
    for el in els
        v = Inti.vertices(el)
        offset = length(pts)
        append!(pts, _to_point.(v))
        push!(elt_vertices, collect((offset + 1):(offset + length(v))))
        push!(elt_domains, Inti.domain(el))
    end
    # call identity.(pts) to try an infer a more specific type for the vector of points
    return identity.(pts), elt_vertices, elt_domains
end

_vertex_topology(el::Inti.ReferenceInterpolant) = _vertex_topology([el])

# Build the (global) triangular faces and segment index pairs from the topology.
function _faces_and_segments(elt_vertices, elt_domains)
    tris = GB.GLTriangleFace[]
    tri2elt = Int[]
    segs = NTuple{2, Int}[]
    for (e, (gverts, d)) in enumerate(zip(elt_vertices, elt_domains))
        for (a, b, c) in _reference_triangles(d)
            push!(tris, GB.GLTriangleFace(gverts[a], gverts[b], gverts[c]))
            push!(tri2elt, e)
        end
        for (a, b) in _reference_segments(d)
            push!(segs, (gverts[a], gverts[b]))
        end
    end
    return tris, tri2elt, segs
end

# Every Inti object that can be plotted: a mesh, a single element, or a vector of
# elements.
const PlotInput = Union{
    Inti.AbstractMesh,
    Inti.ReferenceInterpolant,
    AbstractVector{<:Inti.ReferenceInterpolant},
}

# Ambient dimension of any plottable object.
_ambient_dimension(obj) = Inti.ambient_dimension(obj)
function _ambient_dimension(obj::AbstractVector)
    @assert allequal(Inti.ambient_dimension.(obj)) "all elements must have the same ambient dimension"
    return Inti.ambient_dimension(first(obj))
end

# A flat (low-order) triangular `GeometryBasics` mesh for a plottable object.
function _geometry_mesh(obj)
    pts, elt_vertices, elt_domains = _vertex_topology(obj)
    tris, _, _ = _faces_and_segments(elt_vertices, elt_domains)
    return GB.Mesh(pts, tris)
end

## convert_arguments: enables native Makie verbs (mesh, lines, scatter, ...)

# surface/volume objects and wireframes share the triangular-mesh conversion
Makie.convert_arguments(::Type{<:Makie.Mesh}, obj::PlotInput) = (_geometry_mesh(obj),)
Makie.convert_arguments(::Type{<:Makie.Wireframe}, obj::PlotInput) = (_geometry_mesh(obj),)

# point-based plots (scatter, lines on a single element, ...)
function Makie.convert_arguments(::Makie.PointBased, obj::Inti.AbstractMesh)
    return (_to_point.(Inti.nodes(obj)),)
end
function Makie.convert_arguments(::Makie.PointBased, el::Inti.ReferenceInterpolant)
    return (_to_point.(Inti.vertices(el)),)
end

## MeshPlot recipe

"""
    MeshPlot

`Makie` recipe for visualizing `Inti` elements and meshes. It is reached through
`Makie.plot`/`plot!` (registered via `Makie.plottype`), so the usual call is
`plot(msh; kwargs...)` / `plot!(msh; kwargs...)`. Native `Makie` methods (`mesh`,
`wireframe`, `scatter`, `lines`) also accept `Inti` objects through
`Makie.convert_arguments`.

Any attribute understood by `Makie.mesh` (`colormap`, `colorrange`, `alpha`,
`shading`, `transparency`, `nan_color`, ...) is forwarded to the underlying
plot, so standard `Makie` keywords work as usual. The recipe-specific attributes
are:

  - `color`: a single color, or a vector of values. A vector matching the number
    of nodes is treated as nodal data; one matching the number of elements as
    elementwise data.
  - `interpolate = true`: for nodal data, interpolate the color across each
    element (Gouraud) when `true`, or use a single (flat) color per element when
    `false`.
  - `strokewidth = 0`: width of the element edges drawn on surface/volume meshes
    (`0` hides them), following the convention of `Makie.poly`.
  - `strokecolor = :black`: color of those edges.
  - `shading = Makie.automatic`: by default, lighting is enabled (`shading = true`)
    for surfaces in 3D ambient space and disabled (`shading = false`) for coplanar
    2D meshes (where lighting conveys no information and renders the faces black in
    a 2D `Axis`). Pass an explicit `shading` value to override.

For curve (1D) meshes the standard `Makie` attribute `linewidth` controls the
line width (there are no faces, so `strokewidth` does not apply).

!!! note
    High-order (curved) elements are rendered using their low-order (flat)
    counterpart.
"""
Makie.@recipe(MeshPlot, obj) do scene
    # Declare, in addition to the recipe-specific attributes, every styling
    # attribute understood by the underlying `Mesh`/`LineSegments` plots. This is
    # required because attributes the user passes that are *not* declared here are
    # only attached to the plot *after* `plot!` runs, so they would otherwise be
    # invisible to `shared_attributes` (and hence never forwarded to the children).
    # `cycle` and the structural attributes (`transformation`, `model`, ...) are
    # excluded: they are handled specially by Makie's plot machinery and must not
    # be declared as plain attributes here.
    exclude = (:transformation, :model, :transform_func, :dim_conversions, :cycle)
    attr = Makie.Attributes()
    for T in (Makie.Mesh, Makie.LineSegments)
        for (k, v) in Makie.default_theme(scene, T)
            (k in exclude) && continue
            attr[k] = Makie.to_value(v)
        end
    end
    # recipe-specific attributes (these override any merged-in defaults)
    attr[:color] = :lightgray
    attr[:interpolate] = true
    attr[:strokewidth] = 0
    attr[:strokecolor] = :black
    # `automatic` is resolved per ambient dimension in `plot!`: lighting helps
    # convey the form of genuine 3D surfaces, but is useless (and renders flat 2D
    # meshes black in a 2D `Axis`) for coplanar data. An explicit value passes
    # through untouched.
    attr[:shading] = Makie.automatic
    return attr
end

# Given the topology and a `color`, build the GeometryBasics mesh and the color
# vector to be handed to `mesh!`, honouring the `interpolate` flag for nodal
# data and supporting elementwise data.
function _colored_mesh(pts, tris, tri2elt, nnodes, nelts, color, interpolate)
    is_nodal = color isa AbstractVector && length(color) == nnodes
    is_elementwise = color isa AbstractVector && length(color) == nelts
    if is_nodal && interpolate
        # share vertices and let Makie interpolate the nodal data (Gouraud)
        return GB.Mesh(pts, tris), color
    elseif is_nodal || is_elementwise
        # flat shading: duplicate vertices per triangle and assign a single value
        newpts = similar(pts, 3 * length(tris))
        newtris = similar(tris, length(tris))
        newcol = Vector{eltype(color)}(undef, 3 * length(tris))
        for (t, f) in enumerate(tris)
            a, b, c = GB.value.((f[1], f[2], f[3]))
            i = 3t - 2
            newpts[i], newpts[i + 1], newpts[i + 2] = pts[a], pts[b], pts[c]
            newtris[t] = GB.GLTriangleFace(i, i + 1, i + 2)
            val = is_elementwise ? color[tri2elt[t]] : (color[a] + color[b] + color[c]) / 3
            newcol[i] = newcol[i + 1] = newcol[i + 2] = val
        end
        return GB.Mesh(newpts, newtris), newcol
    else
        # uniform color
        return GB.Mesh(pts, tris), color
    end
end

# linesegments expects a flat vector of points (consecutive pairs are segments)
function _segment_points(pts, segs)
    out = similar(pts, 2 * length(segs))
    for (k, (a, b)) in enumerate(segs)
        out[2k - 1], out[2k] = pts[a], pts[b]
    end
    return out
end

# Expand a nodal/elementwise `color` to one value per segment endpoint (matching
# `_segment_points`). A scalar color is returned unchanged.
function _segment_color(segs, color, nnodes, nelts)
    if color isa AbstractVector && length(color) == nnodes
        return reduce(vcat, ([color[a], color[b]] for (a, b) in segs); init = eltype(color)[])
    elseif color isa AbstractVector && length(color) == nelts
        # one value per element, shared by both endpoints
        return reduce(vcat, ([c, c] for c in color); init = eltype(color)[])
    else
        return color
    end
end

function Makie.plot!(p::MeshPlot)
    topo = Makie.lift(p.obj) do obj
        pts, elt_vertices, elt_domains = _vertex_topology(obj)
        tris, tri2elt, segs = _faces_and_segments(elt_vertices, elt_domains)
        return (; pts, tris, tri2elt, segs, nnodes = length(pts), nelts = length(elt_vertices))
    end

    has_faces = !isempty(topo[].tris)

    if has_faces
        meshobs = Makie.lift(topo, p.color, p.interpolate) do t, color, interpolate
            return _colored_mesh(t.pts, t.tris, t.tri2elt, t.nnodes, t.nelts, color, interpolate)
        end
        # resolve `automatic` shading per ambient dimension: lit for 3D surfaces,
        # unlit for coplanar 2D meshes (an explicit user value is kept).
        shadingobs = Makie.lift(p.obj, p.shading) do obj, shading
            shading === Makie.automatic || return shading
            return _ambient_dimension(obj) == 3
        end
        # forward all Makie `mesh` attributes (colormap, alpha, shading, ...),
        # overriding `color` with the (possibly per-node) computed color vector.
        # `shared_attributes` collects every attribute the user passed (including
        # ones not declared by the recipe) and keeps those valid for `Mesh`.
        Makie.mesh!(
            p,
            Makie.shared_attributes(p, Makie.Mesh),
            Makie.lift(first, meshobs);
            color = Makie.lift(last, meshobs),
            shading = shadingobs,
        )
        # element edges (poly-style): drawn when `strokewidth > 0`. These keep
        # their own `strokecolor`/`strokewidth` (independent of the fill's `color`
        # and `alpha`), but follow the plot's `visible` toggle.
        Makie.linesegments!(
            p,
            Makie.lift(t -> _segment_points(t.pts, t.segs), topo);
            color = p.strokecolor,
            linewidth = p.strokewidth,
            visible = p.visible,
        )
    else
        # no faces (e.g. a mesh of line elements): draw the segments themselves.
        # Forward all attributes valid for `LineSegments` (so `linewidth`,
        # `colormap`, ... work) and override `color` so nodal/elementwise data
        # can still be shown.
        segpts = Makie.lift(t -> _segment_points(t.pts, t.segs), topo)
        segcolor = Makie.lift((t, color) -> _segment_color(t.segs, color, t.nnodes, t.nelts), topo, p.color)
        Makie.linesegments!(
            p,
            Makie.shared_attributes(p, Makie.LineSegments),
            segpts;
            color = segcolor,
        )
    end
    return p
end

# Prefer a 3D axis when the data lives in 3D ambient space.
function Makie.preferred_axis_type(p::MeshPlot)
    return _ambient_dimension(Makie.to_value(p.obj)) == 3 ? Makie.LScene : Makie.Axis
end

## Route `Makie.plot`/`plot!` to the `MeshPlot` recipe for Inti objects. With this
## in place, `plot(msh)`/`plot!(msh)` (which dispatch to the recipe-generated
## `meshplot`/`meshplot!`) just work, so no symbol needs to be exported from `Inti`.
Makie.plottype(::PlotInput) = MeshPlot

end # module
