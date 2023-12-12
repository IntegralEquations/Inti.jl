"""
    struct BlockMatrix{T<:SMatrix,S} <: AbstractMatrix{T}

A struct which behaves identically to a `Matrix{T}`, but with the underlying
`data` stored as a `Matrix{S}`, where `S::Number = eltype(T)` is the scalar type
associated with `T`. This allows for the use of `blas` routines under-the-hood,
while providing a convenient interface for handling matrices over tensors.

Use `BlockMatrix(T::DataType, A::Matrix{S})` to construct a `BlockMatrix{T,S}`
with underlying data given by `A`.
"""
struct BlockMatrix{T<:SMatrix,S} <: AbstractMatrix{T}
    data::Matrix{S}
    function BlockMatrix(::Type{T}, data::Matrix{S}) where {T<:Union{SMatrix},S<:Number}
        @assert S == eltype(T)
        @assert sum(rem.(size(data), size(T))) == 0
        return new{T,S}(data)
    end
end
# function BlockMatrix(::Type{T},args...) where {T <: SVector}
#     p,q = length(T),1
#     M = SMatrix{p,q,eltype(T),p*q}
#     BlockMatrix(M,args...)
# end

"""
    BlockMatrix(T::DataType, undef, m, n)

Construct a `BlockMatrix{T}` of size m×n, initialized with missing entries.
"""
function BlockMatrix(::Type{T}, ::UndefInitializer, m, n) where {T<:SMatrix}
    S = eltype(T)
    p, q = size(T)
    data = Matrix{S}(undef, m * p, n * q)
    return BlockMatrix(T, data)
end

blocksize(::BlockMatrix{T}) where {T} = size(T)

function Base.size(A::BlockMatrix)
    return size(A.data) .÷ blocksize(A)
end

function Base.getindex(A::BlockMatrix{T}, i::Int, j::Int) where {T}
    error()
    @warn "using slow getindex for BlockMatrix" maxlog=1
    p, q = size(T)
    I = ((i-1)*p+1):(i*p)
    J = ((j-1)*q+1):(j*q)
    return T(view(A.data, I, J))
end

function Base.setindex!(A::BlockMatrix{T}, v::T, i::Integer, j::Integer) where {T}
    p, q = blocksize(A)
    I = ((i-1)*p+1):(i*p)
    J = ((j-1)*q+1):(j*q)
    return A.data[I, J] = v
end

# Base.similar methods for BlockMatrix
Base.similar(A::BlockMatrix) = BlockMatrix(eltype(A), similar(A.data))
function Base.similar(A::BlockMatrix, ::Type{T}) where {T}
    return BlockMatrix(eltype(A), similar(A.data, T))
end
function Base.similar(A::BlockMatrix, ::Type{T}, dims::Dims) where {T<:SMatrix}
    bsz = size(T)
    @assert bsz == blocksize(A)
    return BlockMatrix(T, similar(A.data, eltype(T), dims .* bsz))
end

Base.BroadcastStyle(::Type{<:BlockMatrix}) = Broadcast.ArrayStyle{BlockMatrix}()

function LinearAlgebra.mul!(C::T, A::T, B::T, a::Number, b::Number) where {T<:BlockMatrix}
    mul!(C.data, A.data, B.data, a, b)
    return C
end

function LinearAlgebra.mul!(
    C::AbstractVector{S},
    A::BlockMatrix{T},
    B::AbstractVector{S},
    a::Number,
    b::Number,
) where {S<:SVector,T}
    Cr = reinterpret(eltype(S), C)
    Br = reinterpret(eltype(S), B)
    mul!(Cr, A.data, Br, a, b)
    return C
end

# https://stackoverflow.com/questions/52603561/how-to-iterate-over-the-non-zero-values-of-a-sparse-array
function LinearAlgebra.axpy!(a::Number, X::SparseMatrixCSC, Y::BlockMatrix)
    size(X) == size(Y) || throw(DimensionMismatch())
    for col in 1:size(X, 2)
        for r in nzrange(X, col)
            row = rowvals(X)[r]
            val = nonzeros(X)[r]
            Y[row, col] += a * val
        end
    end
    return Y
end
Base.:+(X::SparseMatrixCSC, Y::BlockMatrix) = LinearAlgebra.axpy!(true, X, copy(Y))
Base.:+(X::BlockMatrix, Y::SparseMatrixCSC) = Y + X
