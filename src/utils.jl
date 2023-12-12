#=

Utility functions that have nowhere obvious to go.

=#

"""
    svector(f,n)

Create an `SVector` of length n, computing each element as f(i), where i is the
index of the element.
"""
svector(f, n) = ntuple(f, n) |> SVector

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
    @debug "using `Base.promote_op` to infer return type. Consider defining `return_type(::typeof($f),args...)`."
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
    _integration_measure(J::AbstractMatrix)

Given the Jacobian matrix `J` of a transformation `f : ℝᴹ → ℝᴺ`, compute the
integration measure `√det(JᵀJ)`.
"""
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
