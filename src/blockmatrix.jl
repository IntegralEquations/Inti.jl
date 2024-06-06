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
    function BlockMatrix{T}(data::Matrix{S}) where {T<:Union{SMatrix},S<:Number}
        @assert S == eltype(T)
        @assert sum(rem.(size(data), size(T))) == 0
        return new{T,S}(data)
    end
end

Base.parent(A::BlockMatrix) = A.data

"""
    BlockMatrix(T::DataType, undef, m, n)

Construct a `BlockMatrix{T}` of size m×n, initialized with missing entries.
"""
function BlockMatrix{T}(::UndefInitializer, m, n) where {T<:SMatrix}
    S = eltype(T)
    p, q = size(T)
    data = Matrix{S}(undef, m * p, n * q)
    return BlockMatrix{T}(data)
end

blocksize(::BlockMatrix{T}) where {T} = size(T)

function Base.size(A::BlockMatrix)
    return size(A.data) .÷ blocksize(A)
end

function Base.getindex(A::BlockMatrix{T}, i::Int, j::Int) where {T}
    # @warn "calling getindex on BlockMatrix is slow"
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
    return BlockMatrix{T}(similar(A.data, T))
end
function Base.similar(A::BlockMatrix, ::Type{T}, dims::Dims) where {T<:SMatrix}
    bsz = size(T) # block size
    sz  = ntuple(length(bsz)) do i
        if i ≤ length(dims)
            bsz[i] * dims[i]
        else
            bsz[i]
        end
    end
    @assert bsz == blocksize(A)
    @info dims, bsz, sz
    return BlockMatrix{T}(similar(A.data, eltype(T), sz))
end

Base.BroadcastStyle(::Type{<:BlockMatrix}) = Broadcast.ArrayStyle{BlockMatrix}()

function LinearAlgebra.mul!(C::T, A::T, B::T, a::Number, b::Number) where {T<:BlockMatrix}
    mul!(C.data, A.data, B.data, a, b)
    return C
end

function LinearAlgebra.mul!(
    C::Vector{<:SVector},
    A::BlockMatrix,
    B::Vector{<:SVector},
    a::Number,
    b::Number,
)
    C_ = reinterpret(eltype(eltype(C)), C)
    B_ = reinterpret(eltype(eltype(B)), B)
    mul!(C_, A.data, B_, a, b)
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
