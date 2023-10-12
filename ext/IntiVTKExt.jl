module IntiVTKExt

using WriteVTK
import Inti
using StaticArrays

function __init__()
    @info "Loading Inti.jl VTK extension"
end

"""
    mesh_file(mesh::LagrangeMesh[, Ω::Domain], name::String)

Creates a `VTK` file (.vtu) with name `name` containing the mesh information.
It is possible to export only a `Domain` (i.e. only a part of the mesh).

Note that all the heavy lifting is done by the package `WriteVTK`.
We refer to its documentation for more information.

Warning: the output is not an actual file (on disk).
To save it, simply write:

    mesh_file(mesh, "my_mesh") |> vtk_save

(Noting that you will need to be `using WriteVTK` to do so)

To add some data (scalar or vector values at the mesh nodes or mesh cells)
for visualization purposes, you can do for instance:

    vtkfile = mesh_file(mesh, name)
    vtkfile["my_point_data", VTKPointData()] = pdata
    vtkfile["my_cell_data", VTKCellData()] = cdata
    vtk_save(vtkfile)

It is possible also to export a partition `Ωs::Vector{Domain}` using
`Multiblock` files (.vtm), for instance like so

    vtmfile = vtk_multiblock(name)
    for (Ω, pdata) in zip(Ωs, pdatas)
        vtkfile = mesh_file(mesh, Ω, name)
        vtkfile["my_point_data", VTKPointData()] = pdata
    end
    vtk_save(vtmfile)

To save a sequence of solutions (time steps, iterations), simply append
the number of the element to the file name.
Paraview will recognize the sequence automatically.
"""
function mesh_file(mesh::Inti.LagrangeMesh, name::String)
    points = _vtk_points(mesh)
    cells = _vtk_cells(mesh)
    return vtk_grid(name * ".vtu", points, cells)
end
function mesh_file(mesh::Inti.LagrangeMesh, Ω::Inti.Domain, name::String)
    points = _vtk_points(mesh)
    cells = _vtk_cells(mesh, Ω)
    return vtk_grid(name * ".vtu", points, cells)
end

"""
    _vtk_points(mesh::LagrangeMesh)

Creates the matrix of nodes in the format required by `WriteVTK`.
"""
function _vtk_points(mesh::Inti.LagrangeMesh)
    vtx = zeros(Float64, 3, length(Inti.nodes(mesh)))
    for (i, nd) in enumerate(Inti.nodes(mesh))
        vtx[1:Inti.ambient_dimension(mesh), i] = nd
    end
    return vtx
end

"""
    _vtk_cells(mesh::LagrangeMesh, E::DataType)
    _vtk_cells(mesh::LagrangeMesh, Ω::Domain)

Creates the vector of all elements contained in the mesh in the format
required by `WriteVTK` for a particular element type `E<:AbstractElement`
or a `Domain` instance.
"""
function _vtk_cells end

function _vtk_cells(tags, E::DataType)
    vtk_cell_type, ind = etype_to_vtk_cell_type[E]
    return [MeshCell(vtk_cell_type, tags[ind, i]) for i in axes(tags, 2)]
end
function _vtk_cells(mesh::Inti.LagrangeMesh, Ω::Inti.Domain)
    cells = MeshCell[]
    # Loop on `ElementaryEntity`
    for ω in Ω
        # Loop on `AbstractElement`
        for (E, ind) in Inti.ent2tags(mesh)[ω]
            # Subset corresponding to the `ElementaryEntity`
            tags = Inti.elements(mesh)[E][:, ind]
            append!(cells, _vtk_cells(tags, E))
        end
    end
    return cells
end
function _vtk_cells(mesh::Inti.LagrangeMesh)
    cells = MeshCell[]
    # Loop on `AbstractElement`
    for (E, tags) in Inti.elements(mesh)
        # Export only the cells of the largest geometrical dimension
        if Inti.domain_dimension(E) == Inti.ambient_dimension(mesh)
            append!(cells, _vtk_cells(tags, E))
        end
    end
    return cells
end

"""
    const etype_to_vtk_cell_type

Dictionary mapping internal element types to a tuple containing:
    - the corresponding `WriteVTK` cell types (following the convention
    chosen by `VTK`, see below);
    - the indices in the `elements` column that defines the element.
    This is because we want to support the export of more than just the flat
    elements available in the `VTK` specification, hence which may require
    a conversion of some sort.

See VTK specification [Fig. 2] on
[http://www.vtk.org/VTK/img/file-formats.pdf](http://www.vtk.org/VTK/img/file-formats.pdf)

- VTK_VERTEX (=1)
- VTK_POLY_VERTEX (=2)
- VTK_LINE (=3)
- VTK_POLY_LINE (=4)
- VTK_TRIANGLE (=5)
- VTK_TRIANGLE_STRIP (=6)
- VTK_POLYGON (=7)
- VTK_PIXEL (=8)
- VTK_QUAD (=9)
- VTK_TETRA (=10)
- VTK_VOXEL (=11)
- VTK_HEXAHEDRON (=12)
- VTK_WEDGE (=13)
- VTK_PYRAMID (=14)
"""
const etype_to_vtk_cell_type = Dict(
    SVector{3,Float64} => (VTKCellTypes.VTK_VERTEX, collect(1:1)),
    Inti.LagrangeLine{2,SVector{3,Float64}} => (VTKCellTypes.VTK_LINE, collect(1:2)),
    Inti.LagrangeTriangle{3,SVector{2,Float64}} =>
        (VTKCellTypes.VTK_TRIANGLE, collect(1:3)),
    Inti.LagrangeTriangle{3,SVector{3,Float64}} =>
        (VTKCellTypes.VTK_TRIANGLE, collect(1:3)),
    Inti.LagrangeTetrahedron{4,SVector{3,Float64}} =>
        (VTKCellTypes.VTK_TETRA, collect(1:4)),
)

end # module IntiVTKExt
