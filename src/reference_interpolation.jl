"""
    abstract type ReferenceInterpolant{D,T}

Interpolating function mapping points on the domain `D<:ReferenceShape`
(of singleton type) to a value of type `T`.

Instances `el` of `ReferenceInterpolant` are expected to implement:
- `el(x̂)`: evaluate the interpolation scheme at the (reference) coordinate `x̂
  ∈ D`.
- `jacobian(el,x̂)` : evaluate the jacobian matrix of the interpolation at the
  (reference) coordinate `x ∈ D`.

!!! note
    For performance reasons, both `el(x̂)` and `jacobian(el,x̂)` should
    take as input a `StaticVector` and output a static vector or static array.
"""
abstract type ReferenceInterpolant{D,T} end

function (el::ReferenceInterpolant)(x)
    return interface_method(el)
end

geometric_dimension(::ReferenceInterpolant{D,T}) where {D,T} = geometric_dimension(D)
ambient_dimension(el::ReferenceInterpolant{D,T}) where {D,T} = length(T)

"""
    jacobian(f,x)

Given a (possibly vector-valued) functor `f : 𝐑ᵐ → 𝐅ⁿ`, return the `n × m`
matrix `Aᵢⱼ = ∂fᵢ/∂xⱼ`. By default `ForwardDiff` is used to compute the
jacobian, but you should overload this method for specific `f` if better
performance and/or precision is required.

Note: both `x` and `f(x)` are expected to be of `SVector` type.
"""
function jacobian(f, s)
    return ForwardDiff.jacobian(f, s)
end
jacobian(f, s::Real) = jacobian(f, SVector(s))

"""
    hessian(el,x)

Given a (possibly vector-valued) functor `f : 𝐑ᵐ → 𝐅ⁿ`, return the `n × m × m`
matrix `Aᵢⱼⱼ = ∂²fᵢ/∂xⱼ∂xⱼ`. By default `ForwardDiff` is used to compute the
hessian, but you should overload this method for specific `f` if better
performance and/or precision is required.

Note: both `x` and `f(x)` are expected to be of `SVector` type.
"""
function hessian(el::ReferenceInterpolant, s)
    N = ambient_dimension(el)
    M = geometric_dimension(el)
    S = Tuple{N,M,M}
    return SArray{S}(stack(i -> ForwardDiff.hessian(x -> el(x)[i], s), 1:N; dims = 1))
end

function first_fundamental_form(el::ReferenceInterpolant, x̂)
    jac = jacobian(el, x̂)
    # first fundamental form
    E = dot(jac[:, 1], jac[:, 1])
    F = dot(jac[:, 1], jac[:, 2])
    G = dot(jac[:, 2], jac[:, 2])
    return E, F, G
end

function second_fundamental_form(el::ReferenceInterpolant, x̂)
    jac = jacobian(el, x̂)
    ν = _normal(jac)
    # second fundamental form
    hess = hessian(el, x̂)
    L = dot(hess[:, 1, 1], ν)
    M = dot(hess[:, 1, 2], ν)
    N = dot(hess[:, 2, 2], ν)

    return L, M, N
end

"""
    mean_curvature(τ, x̂)

Calculate the [mean curvature](https://en.wikipedia.org/wiki/Mean_curvature) of
the element `τ` at the parametric coordinate `x̂`.
"""
function mean_curvature(el::ReferenceInterpolant, x̂)
    E, F, G = first_fundamental_form(el, x̂)
    L, M, N = second_fundamental_form(el, x̂)
    # mean curvature
    κ = (L * G - 2 * F * M + E * N) / (2 * (E * G - F^2))
    return κ
end

"""
    gauss_curvature(τ, x̂)

Calculate the [Gaussian
curvature](https://en.wikipedia.org/wiki/Gaussian_curvature) of the element `τ`
at the parametric coordinate `x̂`.
"""
function gauss_curvature(el::ReferenceInterpolant, x̂)
    E, F, G = first_fundamental_form(el, x̂)
    L, M, N = second_fundamental_form(el, x̂)
    # Guassian curvature
    κ = (L * N - M^2) / (E * G - F^2)
    return κ
end

domain(::ReferenceInterpolant{D,T}) where {D,T} = D()
domain(::Type{<:ReferenceInterpolant{D,T}}) where {D,T} = D()

# TODO: deprecate `domain` in favor of `reference_domain` for clarity
reference_domain(el) = domain(el)

return_type(::ReferenceInterpolant{D,T}) where {D,T} = T
return_type(::Type{<:ReferenceInterpolant{D,T}}) where {D,T} = T
domain_dimension(t::ReferenceInterpolant{D,T}) where {D,T} = domain(t) |> center |> length
function domain_dimension(t::Type{<:ReferenceInterpolant{D,T}}) where {D,T}
    return domain(t) |> center |> length
end
function range_dimension(el::ReferenceInterpolant{R,T}) where {R,T}
    return domain(el) |> center |> el |> length
end
function range_dimension(el::Type{<:ReferenceInterpolant{R,T}}) where {R,T}
    return domain(el) |> center |> el |> length
end

center(el::ReferenceInterpolant{D}) where {D} = el(center(D()))

# FIXME: need a practical definition of an approximate "radius" of an element.
# Does not need to be very sharp, since we mostly need to put elements inside a
# bounding ball. The method below is more a of a hack, but it is valid for
# convex polygons.
function radius(el::ReferenceInterpolant{D}) where {D}
    xc = center(el)
    return maximum(x -> norm(x - xc), vertices(el))
end

vertices(el::ReferenceInterpolant{D}) where {D} = el.(vertices(D()))

"""
    struct HyperRectangle{N,T} <: ReferenceInterpolant{ReferenceHyperCube{N},T}

Axis-aligned hyperrectangle in `N` dimensions given by
`low_corner::SVector{N,T}` and `high_corner::SVector{N,T}`.
"""
struct HyperRectangle{N,T} <: ReferenceInterpolant{ReferenceHyperCube{N},T}
    low_corner::SVector{N,T}
    high_corner::SVector{N,T}
    # check that low_corner <= high_corner
    function HyperRectangle(low_corner::SVector{N,T}, high_corner::SVector{N,T}) where {N,T}
        @assert all(low_corner .<= high_corner) "low_corner must be less than high_corner"
        return new{N,T}(low_corner, high_corner)
    end
end

low_corner(el::HyperRectangle) = el.low_corner
high_corner(el::HyperRectangle) = el.high_corner
geometric_dimension(::HyperRectangle{N,T}) where {N,T} = N
ambient_dimension(::HyperRectangle{N,T}) where {N,T} = N

function (el::HyperRectangle)(u)
    lc = low_corner(el)
    hc = high_corner(el)
    v = @. lc + (hc - lc) * u
    return v
end

"""
    ParametricElement{D,T,F} <: ReferenceInterpolant{D,T}

An element represented through a explicit function `f` mapping `D` into the
element. For performance reasons, `f` should take as input a `StaticVector` and
return a `StaticVector` or `StaticArray`.

See also: [`ReferenceInterpolant`](@ref), [`LagrangeElement`](@ref)
"""
struct ParametricElement{D<:ReferenceShape,T,F} <: ReferenceInterpolant{D,T}
    parametrization::F
    function ParametricElement{D,T}(f::F) where {F,D,T}
        return new{D,T,F}(f)
    end
end

parametrization(el::ParametricElement) = el.parametrization
domain(::ParametricElement{D,T,F}) where {D,T,F} = D()
return_type(::ParametricElement{D,T,F}) where {D,T,F} = T

ambient_dimension(p::ParametricElement) = length(return_type(p))

function (el::ParametricElement)(u)
    @assert u ∈ domain(el)
    f = parametrization(el)
    return f(u)
end

vertices_idxs(::Type{<:ParametricElement{ReferenceLine}}) = 1:2
vertices_idxs(::Type{<:ParametricElement{ReferenceTriangle}}) = 1:3
vertices_idxs(::Type{<:ParametricElement{ReferenceSquare}}) = 1:4
vertices_idxs(::Type{<:ParametricElement{ReferenceTetrahedron}}) = 1:4
vertices_idxs(::Type{<:ParametricElement{ReferenceCube}}) = 1:8
vertices_idxs(el::ParametricElement) = vertices_idxs(typeof(el))

"""
    ParametricElement(f, d::HyperRectangle)

Construct the element defined as the image of `f` over `d`.
"""
function ParametricElement(f, d::HyperRectangle{N,T}) where {N,T}
    V = return_type(f, SVector{N,T})
    D = ReferenceHyperCube{N}
    return ParametricElement{D,V}((x) -> f(d(x)))
end

"""
    struct LagrangeElement{D,Np,T} <: ReferenceInterpolant{D,T}

A polynomial `p : D → T` uniquely defined by its `Np` values on the `Np` reference nodes
of `D`.

The return type `T` should be a vector space (i.e. support addition and
multiplication by scalars). For istance, `T` could be a number or a vector, but
not a `Tuple`.
"""
struct LagrangeElement{D<:ReferenceShape,Np,T} <: ReferenceInterpolant{D,T}
    vals::SVector{Np,T}
end

vals(el::LagrangeElement) = el.vals

"""
    reference_nodes(el::LagrangeElement)
    reference_nodes(::Type{<:LagrangeElement})

Return the reference nodes on `domain(el)` used for the polynomial
interpolation. The function values on these nodes completely determines the
interpolating polynomial.
"""
function reference_nodes(el::LagrangeElement)
    return interface_method(el)
end

# infer missig information from type of vals
function LagrangeElement{D}(vals::SVector{Np,T}) where {D,Np,T}
    return LagrangeElement{D,Np,T}(vals)
end

# a more convenient syntax
LagrangeElement{D}(x1, xs...) where {D} = LagrangeElement{D}(SVector(x1, xs...))

"""
    order(el::LagrangeElement)

The order of the element's interpolating polynomial (e.g. a `LagrangeLine` with
`2` nodes defines a linear polynomial, and thus has order `1`).
"""
@generated function order(::Type{<:LagrangeElement{D,Np}})::Int where {D,Np}
    if D <: ReferenceHyperCube
        N = geometric_dimension(D)
        K = findfirst(i -> i^N == Np, 1:100) - 1
        isnothing(K) && error("Np must be a perfect $N-th root")
    elseif D <: ReferenceSimplex
        N = geometric_dimension(D)
        K = findfirst(i -> binomial(i + N, N) == Np, 0:100) - 1
        isnothing(K) && error("Np must be an $N-triangular number")
    else
        notimplemented()
    end
    return :($K)
end

"""
    const LagrangeLine = LagrangeElement{ReferenceLine}
"""
const LagrangeLine = LagrangeElement{ReferenceLine}

const Line1D{T} = LagrangeElement{ReferenceLine,2,SVector{1,T}}
const Line2D{T} = LagrangeElement{ReferenceLine,2,SVector{2,T}}
const Line3D{T} = LagrangeElement{ReferenceLine,2,SVector{3,T}}
Line1D(args...) = Line1D{Float64}(args...)
Line2D(args...) = Line2D{Float64}(args...)
Line3D(args...) = Line3D{Float64}(args...)

integration_measure(l::Line1D) = norm(vals(l)[2] - vals(l)[1])

"""
    const LagrangeTriangle = LagrangeElement{ReferenceTriangle}
"""
const LagrangeTriangle = LagrangeElement{ReferenceTriangle}

const Triangle2D{T} = LagrangeElement{ReferenceTriangle,3,SVector{2,T}}
const Triangle3D{T} = LagrangeElement{ReferenceTriangle,3,SVector{3,T}}
Triangle2D(args...) = Triangle2D{Float64}(args...)
Triangle3D(args...) = Triangle3D{Float64}(args...)

"""
    const LagrangeTetrahedron = LagrangeElement{ReferenceTetrahedron}
"""
const LagrangeTetrahedron = LagrangeElement{ReferenceTetrahedron}

"""
    const LagrangeSquare = LagrangeElement{ReferenceSquare}
"""
const LagrangeSquare = LagrangeElement{ReferenceSquare}

const Quadrangle2D{T} = LagrangeElement{ReferenceSquare,4,SVector{2,T}}
const Quadrangle3D{T} = LagrangeElement{ReferenceSquare,4,SVector{3,T}}
Quadrangle2D(args...) = Quadrangle2D{Float64}(args...)
Quadrangle3D(args...) = Quadrangle3D{Float64}(args...)

"""
    const LagrangeCube = LagrangeElement{ReferenceCube}
"""
const LagrangeCube = LagrangeElement{ReferenceCube}

"""
    vertices_idxs(el::LagrangeElement)
    vertices_idxs(::Type{LagrangeElement})

The indices of the nodes in `el` that define the vertices of the element.
"""
vertices_idxs(::Type{<:LagrangeLine{N}}) where {N} = SVector(1, N)

function vertices_idxs(::Type{<:LagrangeSquare{N2}}) where {N2}
    N = order(LagrangeSquare{N2}) + 1
    return SVector(1, N, N2, N2 - N + 1)
end

function vertices_idxs(::Type{<:LagrangeCube{N3}}) where {N3}
    N = order(LagrangeCube{N3}) + 1
    N2 = N * N
    low_face = SVector(1, N, N2 - N + 1, N2)
    up_face = (N3 - N2) .+ low_face
    return SVector(low_face..., up_face...)
end

function vertices_idxs(::Type{<:LagrangeTriangle{Np}}) where {Np}
    N = order(LagrangeTriangle{Np}) + 1
    return SVector(1, N, Np)
end

function vertices_idxs(::Type{<:LagrangeTetrahedron{Np}}) where {Np}
    N = order(LagrangeTetrahedron{Np}) + 1
    return SVector(1, N, N * (N + 1) ÷ 2, Np)
end

vertices_idxs(el::LagrangeElement) = vertices_idxs(typeof(el))

"""
    vertices(el::LagrangeElement)

Coordinates of the vertices of `el`.
"""
vertices(el::LagrangeElement) = view(vals(el), vertices_idxs(el))

"""
    boundary_idxs(el::LagrangeElement)

The indices of the nodes in `el` that define the boundary of the element.
"""

function boundary_idxs(::Type{<:LagrangeLine})
    return 1, 2
end

function boundary_idxs(T::Type{<:LagrangeTriangle})
    I = vertices_idxs(T)
    return (I[1], I[2]), (I[2], I[3]), (I[3], I[1])
end

function boundary_idxs(T::Type{<:LagrangeSquare})
    I = vertices_idxs(T)
    return (I[1], I[2]), (I[2], I[3]), (I[3], I[4]), (I[4], I[1])
end

# generic ℚₖ elements for ReferenceHyperCube
function reference_nodes(T::Type{<:LagrangeElement{ReferenceHyperCube{D},Np}}) where {D,Np}
    n = order(T) + 1
    @assert abs(n - Np^(1 / D)) < 1e-8 "Np must be a perfect power of D"
    nodes1d = ntuple(i -> n == 1 ? 0.5 : range(0, 1, n), D)
    nodes = map(Iterators.product(nodes1d...)) do x
        return SVector(x...)
    end
    return SVector{Np}(nodes)
end

@generated function (el::LagrangeElement{ReferenceHyperCube{D},Np})(u) where {D,Np}
    n    = order(el) + 1
    dims = ntuple(i -> n, D)
    # fetch references nodes on format expected by `lagrange_interp`
    nodes1d   = n == 1 ? [0.5] : collect(range(0, 1, n))
    weights1d = barycentric_lagrange_weights(nodes1d)
    nodes     = ntuple(i -> nodes1d, D)
    weights   = ntuple(i -> weights1d, D)
    return quote
        v = reshape(vals(el), $dims)
        return tensor_lagrange_interp(SVector(u), v, $nodes, $weights, Val(D), 1, Np)
    end
end

"""
    tensor_lagrange_interp(
        x::SVector{N,Td},
        vals::AbstractArray{<:Any,N},
        nodes::NTuple{N},
        weights::NTuple{N},
        ::Val{dim},
        i1,
        len,
        ::Val{SKIP} = Val(false)
    ) where {N,Td,dim,SKIP}

Low-level function performing tensor-product Lagrange interpolation of an N-dimensional
function at the point `x`.

# Arguments
- `x::SVector{N,Td}`: The point at which to interpolate, given as a static vector of length `N`.
- `vals::AbstractArray{<:Any,N}`: The array of function values at the interpolation nodes, with `N` dimensions.
- `nodes::NTuple{N}`: A tuple containing the interpolation nodes for each dimension.
- `weights::NTuple{N}`: A tuple containing the barycentric weights for each dimension.
- `::Val{dim}`: A type-level value indicating the current dimension for recursion.
- `i1`: The starting index for the current slice of `vals`.
- `len`: The stride length for the current dimension.

# Returns
- The interpolated value at the point `x`, of the same type as the elements of `vals`.
"""
@inline function tensor_lagrange_interp(
    x::SVector{N,Td},
    vals::AbstractArray{<:Any,N},
    nodes::NTuple{N},
    weights::NTuple{N},
    ::Val{dim},
    i1,
    len,
    ::Val{SKIP} = Val(false),
) where {N,Td,dim,SKIP}
    T = eltype(vals)
    n = size(vals, dim)
    @inbounds xd = x[dim]
    @inbounds W = weights[dim]
    @inbounds X = nodes[dim]
    num = zero(T)
    l = one(Td)
    res = zero(T)
    δ = one(Td)
    # although the modified lagrange formula below is backward stable, autodiffing through
    # it is not if we are close to an interpolation node. The fix here is to switch to a
    # slightly different representation, which is more stable, if `x` is ever close to an
    # interpolation node. The `thres` variable below was chosen empirically for `Float64`
    # types.
    thres = 1e-3
    if dim == 1
        for i in 1:n
            @inbounds ci = vals[i1+(i-1)]
            @inbounds wi = W[i]
            @inbounds x_m_xi = xd - X[i]
            if SKIP || abs(x_m_xi) > thres || (!iszero(res))
                l *= x_m_xi
                num += (wi / x_m_xi) * ci
            else
                δ = x_m_xi
                res = ci * wi
            end
        end
        return res * l + l * δ * num
    else
        Δi = len ÷ n # column-major stride of current dimension
        # recurse down on dimension
        dim′ = Val{dim - 1}()
        @inbounds for i in 1:n
            ci =
                tensor_lagrange_interp(x, vals, nodes, weights, dim′, i1 + (i - 1) * Δi, Δi)
            wi = W[i]
            x_m_xi = xd - X[i]
            if SKIP || abs(x_m_xi) > thres || (!iszero(res))
                l *= x_m_xi
                num += (wi / x_m_xi) * ci
            else
                δ = x_m_xi
                res = ci * wi
            end
        end
        return res * l + l * δ * num
    end
end

function barycentric_lagrange_weights(x::AbstractVector)
    n = length(x)
    w = map(1:n) do i
        xᵢ = x[i]
        prod(Iterators.filter(j -> j ≠ i, 1:n); init = one(eltype(x))) do j
            xⱼ = x[j]
            return xᵢ - xⱼ
        end
    end
    w .= 1 ./ w
    return w
end

# generic ℙₖ elements for ReferenceSimplex
function reference_nodes(T::Type{<:LagrangeElement{ReferenceSimplex{D},Np}}) where {D,Np}
    k = order(T)
    k == 0 && return SVector{1}((svector(i -> 1 / (D + 1), D),))
    nodes = SVector{D,Float64}[]
    for I in Iterators.product(ntuple(i -> 0:k, D)...)
        sum(I) > k && continue # skip if sum of indices exceeds n
        x = svector(i -> I[i] / k, D)
        push!(nodes, x)
    end
    return SVector{Np}(nodes)
end

# Based on a formula found in
# ``On a class of finite elements generated by Lagrange Interpolation''
# Nicolaides. SINUM 1972.
function (el::LagrangeElement{ReferenceSimplex{D},Np})(u) where {D,Np}
    T = eltype(u)
    k = order(typeof(el))::Int
    iszero(k) && return vals(el)[1] # constant element
    u = SVector{D}(u)
    x = push(u, 1 - sum(u)) # add the last coordinate
    lags1d = MMatrix{k + 1,D + 1,T}(undef)
    @inbounds for dim in 1:(D+1)
        xk = k * x[dim] # scaled coordinate
        lags1d[1, dim] = 1.0 # constant term
        for j in 1:k
            lags1d[j+1, dim] = lags1d[j, dim] * (xk - (j - 1)) / j
        end
    end
    v = vals(el)
    acc = zero(eltype(v))
    @inbounds for (n, I) in enumerate(_barycentric_iterator(el))
        l = v[n] # value at the current node
        for d in 1:(D+1)
            i = I[d] + 1
            l *= lags1d[i, d]
        end
        acc += l
    end
    return acc
end

@generated function _barycentric_iterator(
    el::LagrangeElement{ReferenceSimplex{D},Np},
) where {D,Np}
    k = order(el)
    idxs = MVector{Np,NTuple{D + 1,Int}}(undef)
    cc = 0
    for I in Iterators.product(ntuple(i -> 0:k, D)...)
        sum(I) > k && continue # skip if sum of indices exceeds n
        cc += 1
        idxs[cc] = (I..., k - sum(I)) # add the last coordinate
    end
    return :($idxs)
end

"""
    lagrange_basis(E::Type{<:LagrangeElement})

Return the Lagrange basis `B` for the element `E`. Evaluating `B(x)` yields the
value of each basis function at `x`.
"""
function lagrange_basis(::Type{LagrangeElement{D,N,T}}) where {D,N,T}
    vals = svector(i -> svector(j -> i == j, N), N)
    return LagrangeElement{D}(vals)
end

function boundarynd(::Type{T}, els, msh) where {T}
    bdi = Inti.boundary_idxs(T)
    nedges = length(els) * length(bdi)
    edgelist = Vector{SVector{length(bdi[1]),Int64}}(undef, nedges)
    edgelist_unsrt = Vector{SVector{length(bdi[1]),Int64}}(undef, nedges)
    bords = Vector{MVector{length(bdi[1]),Int64}}(undef, length(bdi))
    for i in 1:length(bdi)
        bords[i] = MVector{length(bdi[1]),Int64}(undef)
    end
    j = 1
    for ii in els
        for k in 1:length(bdi)
            for jjj in 1:length(bdi[k])
                bords[k][jjj] = Inti.connectivity(msh, T)[bdi[k][jjj], ii]
            end
        end
        for q in bords
            edgelist_unsrt[j] = q[:]
            edgelist[j] = sort!(q)
            j += 1
        end
    end
    I = sortperm(edgelist)
    uniqlist = Int64[]
    sizehint!(uniqlist, length(els))
    i = 1
    while i <= length(edgelist) - 1
        if isequal(edgelist[I[i]], edgelist[I[i+1]])
            i += 1
        else
            push!(uniqlist, I[i])
        end
        i += 1
    end
    if !isequal(edgelist[I[end-1]], edgelist[I[end]])
        push!(uniqlist, I[end])
    end
    return edgelist_unsrt[uniqlist]
end
