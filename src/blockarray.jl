"""
    struct BlockArray{T<:StaticArray,N,S} <: AbstractMatrix{T,N}

A struct which behaves like an  `Array{T,N}`, but with the underlying `data`
stored as a `Matrix{S}`, where `S::Number = eltype(T)` is the scalar type
associated with `T`. This allows for the use of `blas` routines under-the-hood,
while providing a convenient interface for handling matrices over
`StaticArray`s.

```jldoctest
using StaticArrays
T = SMatrix{2,2,Int,4}
B = Inti.BlockArray{T}([i*j for i in 1:4, j in 1:4])

# output

2×2 Inti.BlockArray{SMatrix{2, 2, Int64, 4}, 2, Int64}:
 [1 2; 2 4]  [3 4; 6 8]
 [3 6; 4 8]  [9 12; 12 16]

```

"""
struct BlockArray{T<:SArray,N,S} <: AbstractArray{T,N}
    data::Array{S,N}
    function BlockArray{T,N}(data::Array{S,N}) where {T<:Union{SArray},N,S<:Number}
        @assert S == eltype(T) "eltype of data must match eltype of T: $S != $(eltype(T))"
        bsz = size(T)
        sz = size(data)
        # check compatibility of block size
        for i in 1:length(bsz)
            sz[i] % bsz[i] == 0 || throw(
                ArgumentError(
                    "size(data, $i) = $(sz[i]) is not a multiple of block size $(bsz[i])",
                ),
            )
        end
        return new{T,N,S}(data)
    end
end
BlockArray{T}(data::Array{<:Any,N}) where {T,N} = BlockArray{T,N}(data)

BlockVector{T,S} = BlockArray{T,1,S}
BlockMatrix{T,S} = BlockArray{T,2,S}

Base.parent(A::BlockArray) = A.data
blocksize(::BlockArray{T}) where {T} = size(T)

function blocksize_normalized(::BlockArray{T,N}) where {T,N}
    bsz = size(T)
    ntuple(N) do i
        if i ≤ length(bsz)
            bsz[i]
        else
            1
        end
    end
end

# Array interface:
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

function Base.size(A::BlockArray{T,N}) where {T,N}
    return size(A.data) .÷ blocksize_normalized(A)
end

function Base.getindex(A::BlockArray{T,N}, I::Vararg{Int,N}) where {T,N}
    data = parent(A)
    idxs = _blockindex(A, I...)
    return T(view(data, idxs...))
end

function _blockindex(A::BlockArray{T,N}, I::Vararg{Int,N}) where {T,N}
    bsz = blocksize_normalized(A)
    idxs = ntuple(N) do dim
        return ((I[dim]-1)*bsz[dim]+1):(I[dim]*bsz[dim])
    end
    return idxs
end

function Base.setindex!(A::BlockArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
    data = parent(A)
    idxs = _blockindex(A, I...)
    setindex!(data, v, idxs...)
    return v
end

function BlockArray{T}(::UndefInitializer, dims::Dims) where {T<:SArray}
    N = length(dims)
    bsz = size(T)
    S = eltype(T)
    sz = ntuple(N) do dim
        if dim ≤ length(bsz)
            bsz[dim] * dims[dim]
        else
            dims[dim]
        end
    end
    data = Array{S}(undef, sz)
    return BlockArray{T}(data)
end
function BlockArray{T}(::UndefInitializer, m::Integer, n::Integer) where {T<:SArray}
    return BlockArray{T}(undef, (m, n))
end

function Base.similar(::BlockArray, ::Type{T}, dims::Dims) where {T<:SArray}
    return BlockArray{T}(undef, dims)
end

# Broadcast interface:
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting
Base.BroadcastStyle(::Type{<:BlockArray}) = Broadcast.ArrayStyle{BlockArray}()

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{BlockArray}},
    ::Type{ElType},
) where {ElType}
    return similar(BlockArray{ElType}, axes(bc))
end

# Overload some BLAS operations for performance

function LinearAlgebra.mul!(
    C::BlockArray,
    A::BlockArray,
    B::BlockArray,
    a::Number,
    b::Number,
)
    mul!(C.data, A.data, B.data, a, b)
    return C
end

function LinearAlgebra.mul!(
    C::Vector{<:SVector},
    A::BlockArray,
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
function LinearAlgebra.axpy!(a::Number, X::SparseMatrixCSC, Y::BlockArray)
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
Base.:+(X::SparseMatrixCSC, Y::BlockArray) = LinearAlgebra.axpy!(true, X, copy(Y))
Base.:+(X::BlockArray, Y::SparseMatrixCSC) = Y + X

function LinearAlgebra.axpy!(a::Number, X::BlockArray, Y::BlockArray)
    axpy!(a, parent(X), parent(Y))
    return Y
end

LinearAlgebra.rmul!(A::BlockArray, a::Number) = (rmul!(A.data, a); A)
LinearAlgebra.rdiv!(A::BlockArray, a::Number) = (rdiv!(A.data, a); A)

Base.:(*)(a::Number, X::BlockArray) = LinearAlgebra.rmul!(copy(X), a)
Base.:(*)(X::BlockArray, a::Number) = LinearAlgebra.rmul!(copy(X), a)
Base.:(+)(X::BlockArray, Y::BlockArray) = LinearAlgebra.axpy!(true, X, copy(Y))
Base.:(-)(X::BlockArray, Y::BlockArray) = LinearAlgebra.axpy!(-1, X, copy(Y))
Base.:(-)(X::BlockArray) = -1 * X
Base.:(/)(X::BlockArray, a::Number) = rdiv!(copy(X), a)
