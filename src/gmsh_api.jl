"""
   gmsh_import_domain([model;dim=3,verbosity=2])

Construct a [`Domain`](@ref Inti.Domain) from the `gmsh` `model` with all entities of
dimension `dim`; by defaul the current `gmsh` model is used.

!!! warning
    This function assumes that `gmsh` has been initialized, and does not handle
    its finalization.
"""
function gmsh_import_domain end

"""
    gmsh_import_mesh(Ω;[dim=3,verbosity=2])

Create a [`LagrangeMesh`](@ref) for the entities in `Ω`. Passing `dim=2` will
create a two-dimensional mesh by projecting the original mesh onto the `x,y`
plane.

!!! warning
    This function assumes that `gmsh` has been initialized, and does not handle
    its finalization.
"""
function gmsh_import_mesh end

"""
    gmsh_read_geo(fname::String;dim=3)

Read a `.geo` file and generate a [`Domain`](@ref Inti.Domain) with all entities
of dimension `dim`.

!!! warning
    This function assumes that `gmsh` has been initialized, and does not handle
    its finalization.
"""
function gmsh_read_geo end

"""
    gmsh_read_msh(fname::String; dim=3)

Read `fname` and create a `Domain` and a `GenericMesh` structure with all
entities in `Ω` of dimension `dim`.

!!! warning
    This function assumes that `gmsh` has been initialized, and does not handle its
    finalization.
"""
function gmsh_read_msh end
