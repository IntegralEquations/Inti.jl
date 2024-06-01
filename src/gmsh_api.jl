"""
    import_mesh_from_gmsh_model(;[dim=3]) --> Ω, msh

Create a [`Domain`](@ref) and a [`LagrangeMesh`](@ref) from the current `gmsh`
model. Passing `dim=2` will create a two-dimensional mesh by projecting the
original mesh onto the `x,y` plane.

This function assumes that the *Gmsh* API has been initialized through
`gmsh.initialize`.
"""
function import_mesh_from_gmsh_model end

"""
    import_mesh(fname::String; dim=3)

Open `fname` and create a [`LagrangeMesh`](@ref) from the `gmsh` model in it.

See also: [`import_mesh_from_gmsh_model`](@ref).
"""
function import_mesh end

"""
    gmsh_curve(f::Function, a, b; npts=100, tag=-1)

Create a curve in the current `gmsh` model given by `{f(t) : t ∈
(a,b) }` where `f` is a function from `ℝ` to `ℝ^3`. The curve is approximated
by C² b-splines passing through `npts` equispaced in parameter space.
"""
function gmsh_curve end

function write_gmsh_model end

function write_gmsh_view! end

function write_gmsh_view end
