#=

Utility functions that have nowhere obvious to go.

=#

"""
    svector(f,n)

Create an `SVector` of length n, computing each element as f(i), where i is the
index of the element.
"""
@inline svector(f::F, n) where {F} = ntuple(f, n) |> SVector

"""
    interface_method(x)

A method of an `abstract type` for which concrete subtypes are expected to
provide an implementation.
"""
function interface_method(T::DataType)
    return error("this method needs to be implemented by the concrete subtype $T.")
end
interface_method(x) = interface_method(typeof(x))

"""
    standard_basis_vector(k, ::Val{N})

Create an `SVector` of length N with a 1 in the kth position and zeros elsewhere.
"""
standard_basis_vector(k, n::Val{N}) where {N} = svector(i -> i == k ? 1 : 0, n)

"""
    ambient_dimension(x)

Dimension of the ambient space where `x` lives. For geometrical objects this can
differ from its [`geometric_dimension`](@ref); for example a triangle in `ℝ³`
has ambient dimension `3` but geometric dimension `2`, while a curve in `ℝ³` has
ambient dimension 3 but geometric dimension 1.
"""
function ambient_dimension end

"""
    geometric_dimension(x)

NNumber of degrees of freedom necessary to locally represent the geometrical
object. For example, lines have geometric dimension of 1 (whether in `ℝ²` or in
`ℝ³`), while surfaces have geometric dimension of 2.
"""
function geometric_dimension end

"""
    return_type(f[,args...])

The type returned by `f(args...)`, where `args` is a tuple of types. Falls back
to `Base.promote_op` by default.

A functors of type `T` with a knonw return type should extend
`return_type(::T,args...)` to avoid relying on `promote_op`.
"""
function return_type(f, args...)
    return Base.promote_op(f, args...)
end

"""
    domain(f)

Given a function-like object `f: Ω → R`, return `Ω`.
"""
function domain end

"""
    image(f)

Given a function-like object `f: Ω → R`, return `f(Ω)`.
"""
function image end

"""
    integration_measure(f, x̂)

Given the Jacobian matrix `J` of a transformation `f : ℝᴹ → ℝᴺ` compute the
integration measure `√det(JᵀJ)` at the parametric coordinate `x̂`
"""
function integration_measure(f, x)
    jac = jacobian(f, x)
    return _integration_measure(jac)
end

function _integration_measure(jac::AbstractMatrix)
    M, N = size(jac)
    if M == N
        abs(det(jac)) # cheaper when `M=N`
    else
        g = det(transpose(jac) * jac)
        g < -sqrt(eps()) && (@warn "negative integration measure g=$g")
        g = max(g, 0)
        sqrt(g)
    end
end

"""
    normal(el, x̂)

Return the normal vector of `el` at the parametric coordinate `x̂`.
"""
function normal(el, x)
    jac = jacobian(el, x)
    return _normal(jac)
end

"""
    _normal(jac::SMatrix{M,N})

Given a an `M` by `N` matrix representing the jacobian of a codimension one
object, compute the normal vector.
"""
function _normal(jac::SMatrix{N,M}) where {N,M}
    msg = "computing the normal vector requires the element to be of co-dimension one."
    @assert (N - M == 1) msg
    if M == 1 # a line in 2d
        t = jac[:, 1] # tangent vector
        n = SVector(t[2], -t[1]) |> normalize
        return n
    elseif M == 2 # a surface in 3d
        t₁ = jac[:, 1]
        t₂ = jac[:, 2]
        n = cross(t₁, t₂) |> normalize
        return n
    else
        notimplemented()
    end
end

"""
    uniform_points_circle(N,r,c)

Return `N` points uniformly distributed on a circle of radius `r` centered at `c`.
"""
function uniform_points_circle(N, r, c)
    pts = SVector{2,Float64}[]
    for i in 0:(N-1)
        x = r * SVector(cos(2π * i / N), sin(2π * i / N)) + c
        push!(pts, x)
    end
    return pts
end

# https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
"""
    fibonnaci_points_sphere(N,r,c)

Return `N` points distributed (roughly) in a uniform manner on the sphere of
radius `r` centered at `c`.
"""
function fibonnaci_points_sphere(N, r, center)
    pts = Vector{SVector{3,Float64}}(undef, N)
    phi = π * (3 - sqrt(5)) # golden angle in radians
    for i in 1:N
        ytmp = 1 - ((i - 1) / (N - 1)) * 2
        radius = sqrt(1 - ytmp^2)
        theta = phi * i
        x = cos(theta) * radius * r + center[1]
        y = ytmp * r + center[2]
        z = sin(theta) * radius * r + center[3]
        pts[i] = SVector(x, y, z)
    end
    return pts
end

"""
    _copyto!(target,source)

Defaults to `Base.copyto!`, but includes some specialized methods to copy from a
`Matrix` of `SMatrix` to a `Matrix` of `Number`s and viceversa.
"""
function _copyto!(dest::AbstractMatrix{<:Number}, src::AbstractMatrix{<:SMatrix})
    S = eltype(src)
    sblock = size(S)
    ss = size(src) .* sblock # matrix size when viewed as matrix over T
    @assert size(dest) == ss
    for i in 1:ss[1], j in 1:ss[2]
        bi, ind_i = divrem(i - 1, sblock[1]) .+ (1, 1)
        bj, ind_j = divrem(j - 1, sblock[2]) .+ (1, 1)
        dest[i, j] = src[bi, bj][ind_i, ind_j]
    end
    return dest
end
function _copyto!(dest::AbstractMatrix{<:SMatrix}, src::AbstractMatrix{<:Number})
    T = eltype(dest)
    sblock = size(T)
    nblock = div.(size(src), sblock)
    for i in 1:nblock[1]
        istart = (i - 1) * sblock[1] + 1
        iend = i * sblock[1]
        for j in 1:nblock[2]
            jstart = (j - 1) * sblock[2] + 1
            jend = j * sblock[2]
            dest[i, j] = T(view(src, istart:iend, jstart:jend))
        end
    end
    return dest
end

# defaults to Base.copyto!
function _copyto!(dest, src)
    return copyto!(dest, src)
end

# https://discourse.julialang.org/t/putting-threads-threads-or-any-macro-in-an-if-statement/41406/7
macro usethreads(multithreaded, expr::Expr)
    ex = quote
        if $multithreaded
            Threads.@threads $expr
        else
            $expr
        end
    end
    return esc(ex)
end

# some useful type aliases
const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}

function _normalize_compression(compression, target, source)
    methods = (:hmatrix, :fmm, :none)
    # check that method is valid
    compression.method ∈ (:hmatrix, :fmm, :none) || error(
        "Unknown compression.method $(compression.method). Available options: $methods",
    )
    # set default tolerance if not provided
    compression = merge((tol = 1e-8,), compression)
    return compression
end

function _normalize_correction(correction, target, source)
    methods = (:dim, :adaptive, :none)
    # check that method is valid
    correction.method ∈ methods ||
        error("Unknown correction.method $(correction.method). Available options: $methods")
    # set default values if absent
    if correction.method == :dim
        haskey(correction, :target_location) &&
            target === source &&
            correction.target_location != :on &&
            @warn("ignoring target_location field in correction since target === source")
        # target location required unless target === source
        haskey(correction, :target_location) ||
            target === source ||
            error("missing target_location field in correction")
        haskey(correction, :maxdist) ||
            target === source ||
            @warn("missing maxdist field in correction: setting to Inf")
        correction = merge(
            (maxdist = Inf, interpolation_order = nothing, center = nothing),
            correction,
        )
    elseif correction.method == :adaptive
        haskey(correction, :tol) || error("missing tol field in correction")
        correction = merge((maxsplit = 10_000, maxdist = nothing), correction)
    end
    return correction
end

"""
    SAME_POINTS_TOLERANCE

Two points `x` and `y` are considerd the same if `norm(x-y) ≤ SAME_POINT_TOLERANCE`.
"""
const SAME_POINT_TOLERANCE = 1e-14

"""
    notimplemented()

Things which should probably be implemented at some point.
"""
function notimplemented()
    return error("not (yet) implemented")
end

"""
    MultiIndex{N}

Wrapper around `NTuple{N,Int}` mimicking a multi-index in `ℤ₀ᴺ`.
"""
struct MultiIndex{N}
    indices::NTuple{N,Int}
end

Base.:+(a::MultiIndex, b::MultiIndex) = MultiIndex(a.indices .+ b.indices)
Base.:-(a::MultiIndex, b::MultiIndex) = MultiIndex(a.indices .- b.indices)

Base.factorial(n::MultiIndex) = prod(factorial, n.indices)
Base.abs(n::MultiIndex) = sum(n.indices)

function Base.binomial(n::MultiIndex, k::MultiIndex)
    prod(zip(n.indices, k.indices)) do (n, k)
        return binomial(n, k)
    end
end

Base.:<(a::MultiIndex, b::MultiIndex) = all(a.indices .< b.indices)
Base.:<=(a::MultiIndex, b::MultiIndex) = all(a.indices .<= b.indices)

const WEAKDEPS_PROJ = let
    deps = TOML.parse(read(joinpath(PROJECT_ROOT, "Project.toml"), String))["weakdeps"]
    compat = Dict{String,Any}()
    for (pkg, bound) in
        TOML.parse(read(joinpath(PROJECT_ROOT, "Project.toml"), String))["compat"]
        haskey(deps, pkg) || continue
        compat[pkg] = bound
    end
    Dict("deps" => deps, "compat" => compat)
end

# adapted from DataFlowTasks.jl (code by François Févotte)
"""
    stack_weakdeps_env!(; verbose = false, update = false)

Push to the load stack an environment providing the weak dependencies of
Inti.jl. This allows benefiting from additional functionalities of Inti.jl which
are powered by weak dependencies without having to manually install them in your
environment.

Set `update=true` if you want to update the `weakdeps` environment.

!!! warning
    Calling this function can take quite some time, especially the first time
    around, if packages have to be installed or precompiled. Run in `verbose`
    mode to see what is happening.

## Examples:
```example
Inti.stack_weakdeps_env!()
using HMatrices
```
"""
function stack_weakdeps_env!(; verbose = false, update = false)
    weakdeps_env = Scratch.@get_scratch!("weakdeps-$(VERSION.major).$(VERSION.minor)")
    open(joinpath(weakdeps_env, "Project.toml"), "w") do f
        return TOML.print(f, WEAKDEPS_PROJ)
    end

    cpp = Pkg.project().path
    io = verbose ? stderr : devnull

    try
        Pkg.activate(weakdeps_env; io)
        update && Pkg.update(; io)
        Pkg.resolve(; io)
        Pkg.instantiate(; io)
        Pkg.status()
    finally
        Pkg.activate(cpp; io)
    end

    push!(LOAD_PATH, weakdeps_env)
    return nothing
end

"""
    cart2sph(x,y,z)

Map cartesian coordinates `x,y,z` to spherical ones `r, θ, φ` representing the
radius, elevation, and azimuthal angle respectively. The convention followed is
that `0 ≤ θ ≤ π` and ` -π < φ ≤ π`. Same as the `cart2sph` function in MATLAB.
"""
function cart2sph(x, y, z)
    azimuth = atan(y, x)
    a = x^2 + y^2
    elevation = atan(z, sqrt(a))
    r = sqrt(a + z^2)
    return azimuth, elevation, r
end

"""
    rotation_matrix(rot)

Constructs a rotation matrix given the rotation angles around the x, y, and z
axes.

# Arguments
- `rot`: A tuple or vector containing the rotation angles in radians for each
  axis.

# Returns
- `R::SMatrix`: The resulting rotation matrix.
"""
function rotation_matrix(rot)
    dim = length(rot)
    dim == 1 ||
        dim == 3 ||
        throw(
            ArgumentError(
                "rot must have 1 or 3 elements for a 2D or 3D rotation, respectively.",
            ),
        )
    return dim == 1 ? _rotation_matrix_2d(rot) : _rotation_matrix_3d(rot)
end
function _rotation_matrix_2d(rot)
    R = @SMatrix [cos(rot[1]) -sin(rot[1]); sin(rot) cos(rot[1])]
    return R
end
function _rotation_matrix_3d(rot)
    Rx = @SMatrix [1 0 0; 0 cos(rot[1]) sin(rot[1]); 0 -sin(rot[1]) cos(rot[1])]
    Ry = @SMatrix [cos(rot[2]) 0 -sin(rot[2]); 0 1 0; sin(rot[2]) 0 cos(rot[2])]
    Rz = @SMatrix [cos(rot[3]) sin(rot[3]) 0; -sin(rot[3]) cos(rot[3]) 0; 0 0 1]
    return Rz * Ry * Rx
end

"""
    kress_change_of_variables(P)

Return a change of variables mapping `[0,1]` to `[0,1]` with the property that
the first `P-1` derivatives of the transformation vanish at `x=0`.
"""
function kress_change_of_variables(P)
    v = x -> (1 / P - 1 / 2) * ((1 - x))^3 + 1 / P * ((x - 1)) + 1 / 2
    return x -> 2v(x)^P / (v(x)^P + v(2 - x)^P)
end

"""
    kress_change_of_variables_periodic(P)

Like [`kress_change_of_variables`](@ref), this change of variables maps the interval `[0,1]` onto
itself, but the first `P` derivatives of the transformation vanish at **both**
endpoints (thus making it a periodic function).

This change of variables can be used to *periodize* integrals over the interval
`[0,1]` by mapping the integrand into a new integrand that vanishes (to order P)
at both endpoints.
"""
function kress_change_of_variables_periodic(P)
    v = (x) -> (1 / P - 1 / 2) * ((1 - 2x))^3 + 1 / P * ((2x - 1)) + 1 / 2
    return x -> v(x)^P / (v(x)^P + v(1 - x)^P)
end

"""
    connected_components(nodes, neighbors)

Given a list of `nodes` and a dictionary `neighbors` mapping each node to its neighbors,
return partition of `nodes` into connected components.

The partition is given by a `Vector{typeof(nodes)}`, where the `k`-th entry gives a list of
nodes in the `k`-th connected component.
"""
function connected_components(nodes, neighbors)
    visited    = empty(nodes)    # Set to keep track of visited nodes
    components = typeof(nodes)[] # Vector of connected components

    function dfs(node, component)
        # Helper function for DFS
        if node in visited || node ∉ nodes
            return
        end
        push!(visited, node)
        push!(component, node)
        for neighbor in get(neighbors, node, [])
            dfs(neighbor, component)
        end
    end

    for node in nodes
        if node ∉ visited
            component = empty(nodes) # Temporary storage for the current component
            dfs(node, component)
            push!(components, component)
        end
    end

    return components
end
