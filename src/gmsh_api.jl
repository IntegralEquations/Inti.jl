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

Create a `LagrangeMesh` for the entities in `Ω`. Passing `dim=2` will create a
two-dimensional mesh by projecting the original mesh onto the `x,y` plane.

!!! warning
    This function assumes that `gmsh` has been initialized, and does not handle
    its finalization.
"""
function gmsh_import_mesh end
