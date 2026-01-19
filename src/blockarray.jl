"""
    struct BlockArray{T<:StaticArray,N,S} <: AbstractMatrix{T,N}

A struct which behaves like an  `Array{T,N}`, but with the underlying `data` stored as a
`Matrix{S}`, where `S::Number = eltype(T)` is the scalar type associated with `T`. This
allows for the use of many `blas` routines under-the-hood, while providing a convenient
interface for handling arrays over `StaticArray`s.

```jldoctest
using StaticArrays
T = SMatrix{2,2,Int,4}
B = Inti.BlockArray{T}([i*j for i in 1:4, j in 1:4])

# output

2×2 Inti.BlockArray{SMatrix{2, 2, Int64, 4}, 2, Int64, 2}:
 [1 2; 2 4]  [3 4; 6 8]
 [3 6; 4 8]  [9 12; 12 16]

```
"""
struct BlockArray{T <: SArray, N, S, M} <: AbstractArray{T, N}
    data::Array{S, M}
    function BlockArray{T, N}(data::Array{S, M}) where {T <: SArray, N, S <: Number, M}
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
        return new{T, N, S, M}(data)
    end
end
BlockArray{T}(data::Array{<:Any, N}) where {T, N} = BlockArray{T, N}(data)

BlockVector{T, S} = BlockArray{T, 1, S}
BlockMatrix{T, S} = BlockArray{T, 2, S}

Base.parent(A::BlockArray) = A.data

"""
    blocksize(A::BlockArray)

The size of an individual entry of `A`.
"""
blocksize(::BlockArray{T}) where {T} = size(T)

"""
    _blocksize_normalized(A::BlockArray)

Like [`blocksize`](@ref), but appends `1`s if `A` is a higher-dimensional.

For example, a `BlockArray{SVector{3,Float64}, 2}` has a `blocksize` of `(3,)`, but a
`normalized_blocksize` of `(3, 1)`.
"""
function _blocksize_normalized(::BlockArray{T, N}) where {T, N}
    bsz = size(T)
    M = length(bsz)
    if M < N
        return ntuple(i -> i ≤ M ? bsz[i] : 1, N)
    else
        return bsz
    end
end

function one_padding(tp1::Dims{N}, tp2::Dims{M}) where {N, M}
    if N < M
        tp1_pad = ntuple(i -> i ≤ N ? tp1[i] : 1, M)
        return tp1_pad, tp2
    else # N >= M
        tp2_pad = ntuple(i -> i ≤ M ? tp2[i] : 1, N)
        return tp1, tp2_pad
    end
end

function BlockArray(src::AbstractArray{T}) where {T <: SArray}
    dest = BlockArray{T}(undef, size(src))
    dest .= src
    return dest
end

function Base.Array(src::BlockArray{T, N}) where {T, N}
    dest = Array{T, N}(undef, size(src))
    dest .= src
    return dest
end

# Array interface:
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

function Base.size(A::BlockArray{T, N}) where {T, N}
    bsz = _blocksize_normalized(A)
    return ntuple(N) do dim
        return size(A.data, dim) ÷ bsz[dim]
    end
end

function Base.getindex(A::BlockArray{T, N}, I::Vararg{Int, N}) where {T, N}
    data = parent(A)
    idxs = _blockindex(A, I...)
    return T(view(data, idxs...))
end

function _blockindex(A::BlockArray{T, N}, I_::Vararg{Int, N}) where {T, N}
    bsz_ = blocksize(A)
    bsz, I = one_padding(bsz_, I_)
    idxs = ntuple(length(I)) do dim
        return ((I[dim] - 1) * bsz[dim] + 1):(I[dim] * bsz[dim])
    end
    return idxs
end

function Base.setindex!(A::BlockArray{T, N}, v, I::Vararg{Int, N}) where {T, N}
    data = parent(A)
    idxs = _blockindex(A, I...)
    setindex!(data, v, idxs...)
    return v
end

function BlockArray{T}(::UndefInitializer, dims_::Dims) where {T <: SArray}
    N = length(dims_)
    bsz_ = size(T)
    bsz, dims = one_padding(bsz_, dims_)
    S = eltype(T)
    sz = bsz .* dims
    data = Array{S}(undef, sz)
    B = BlockArray{T, N}(data)
    return B
end
function BlockArray{T}(::UndefInitializer, I::Vararg{Int}) where {T}
    return BlockArray{T}(undef, I)
end

function Base.similar(::BlockArray, ::Type{T}, dims::Dims) where {T <: SArray}
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
