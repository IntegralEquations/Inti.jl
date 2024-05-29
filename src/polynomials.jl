"""
    struct PolynomialSpace{D,K}

The space of all polynomials of degree `≤K`, commonly referred to as `ℙₖ`.

The type parameter `D`, of singleton type, is used to determine the reference
domain of the polynomial basis. In particular, when `D` is a hypercube in `d`
dimensions, the precise definition is `ℙₖ = span{𝐱ᶿ : 0≤max(θ)≤ K}`; when `D`
is a `d`-dimensional simplex, the space is `ℙₖ = span{𝐱ᶿ : 0≤sum(θ)≤ K}`, where
`θ ∈ 𝐍ᵈ` is a multi-index.

# See also: [`monomial_basis`](@ref), [`lagrange_basis`](@ref)
"""
struct PolynomialSpace{D,K} end

PolynomialSpace(d::ReferenceShape, k::Int) = PolynomialSpace{typeof(d),k}()

function Base.show(io::IO, ::PolynomialSpace{D,K}) where {D,K}
    return print(io, "ℙ$K : space of all polynomials over $D of degree ≤ $K")
end

domain(sp::PolynomialSpace{D}) where {D} = D()

"""
    dimension(space)

The length of a basis for `space`; i.e. the number of linearly independent
elements required to span `space`.
"""
function dimension(::Type{PolynomialSpace{D,K}}) where {D,K}
    if D == ReferenceLine
        return K + 1
    elseif D == ReferenceTriangle
        return (K + 1) * (K + 2) ÷ 2
    elseif D == ReferenceTetrahedron
        return (K + 1) * (K + 2) * (K + 3) ÷ 6
    elseif D == ReferenceSquare
        return (K + 1)^2
    elseif D == ReferenceCube
        return (K + 1)^3
    else
        notimplemented()
    end
end
dimension(sp::PolynomialSpace) = dimension(typeof(sp))

"""
    monomial_basis(sp::PolynomialSpace)

Return a function `f : ℝᴺ → ℝᵈ`, where `N` is the dimension of the domain of `sp`
    containing a basis of monomials `𝐱ᶿ` spanning the polynomial
space [`PolynomialSpace`](@ref).
"""
function monomial_basis end

function monomial_basis(::PolynomialSpace{ReferenceLine,K}) where {K}
    b = x -> let
        svector(K + 1) do i
            return prod(x .^ (i - 1))
        end
    end
    return b
end

# function monomial_basis(::PolynomialSpace{ReferenceSquare,K}) where {K}
#     # the K+1 monomials x^(0,0), x^(0,1),x^(1,0), ..., x^(K,K)
#     I = CartesianIndices((K + 1, K + 1)) .- CartesianIndex(1, 1)
#     N = length(I)
#     b = ntuple(N) do i
#         θ = Tuple(I[i]) # map linear to cartesian index
#         return x -> prod(x .^ θ)
#     end
#     return b
# end

function monomial_basis(::PolynomialSpace{ReferenceTriangle,K}) where {K}
    # the (K+1)*(K+2)/2 monomials x^(a,b) with a+b ≤ K
    # construct first the indices for the square, then filter only those for
    # which the sum is less than K.
    Isq = CartesianIndices((K + 1, K + 1)) .- CartesianIndex(1, 1)
    I = filter(Isq) do idx
        return sum(Tuple(idx)) ≤ K
    end
    N = Val((K + 1) * (K + 2) ÷ 2)
    b = x -> begin
        svector(N) do i
            return x[1]^I[i][1] * x[2]^I[i][2]
        end
    end
    return b
end

"""
    lagrange_basis(nodes,[sp::AbstractPolynomialSpace])

Return the set of `n` polynomials in `sp` taking the value of `1` on node `i`
and `0` on nodes `j ≂̸ i` for `1 ≤ i ≤ n`.
"""
function lagrange_basis(nodes, sp::PolynomialSpace)
    N = dimension(sp)
    @assert length(nodes) == N
    basis = monomial_basis(sp)
    # compute the matrix of coefficients of the lagrange polynomials over the
    # monomomial basis
    V = hcat([basis(x) for x in nodes]...)
    C = SArray(MArray(V) \ I)
    lag_basis = x -> C * basis(x)
    return lag_basis
end
