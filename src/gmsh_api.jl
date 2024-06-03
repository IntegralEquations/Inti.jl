"""
    import_mesh(filename = nothing; dim=3)

Open `filename` and create a [`LagrangeMesh`](@ref) from the `gmsh` model in it.

If `filename` is `nothing`, the current `gmsh` model is used. Note that this
assumes that the *Gmsh* API has been initialized through `gmsh.initialize`.

Passing `dim=2` will create a two-dimensional mesh by projecting the original
mesh onto the `x,y` plane.
"""
function import_mesh(args...; kwargs...)
    return error("Inti.import_mesh not found. Did you forget to load Gmsh.jl?")
end

"""
    gmsh_curve(f::Function, a, b; npts=100, tag=-1)

Create a curve in the current `gmsh` model given by `{f(t) : t ∈
(a,b) }` where `f` is a function from `ℝ` to `ℝ^3`. The curve is approximated
by C² b-splines passing through `npts` equispaced in parameter space.
"""
function gmsh_curve(args...; kwargs...)
    return error("Inti.gmsh_curve not found. Did you forget to load Gmsh.jl?")
end

function write_gmsh_model end

function write_gmsh_view! end

function write_gmsh_view end
