using StaticArrays
using LinearAlgebra
using Inti
using Test

T = SVector{3,Float64}
data = rand(30, 10)
B = Inti.BlockArray{T}(data)
@test Inti.blocksize(B) == (3,)
@test Inti.blocksize_normalized(B) == (3, 1)
@test size(B) == (10, 10)
@test @inferred(B[1, 1]) == data[1:3, 1]
@test @inferred(B[1, 2]) == data[1:3, 2]
@test B[2, 1] == data[4:6, 1]
@test B[2, 2] == data[4:6, 2]
@test eltype(B) == T
v = rand(T)
B[1, 1] = v
@test B[1, 1] == v

B = Inti.BlockArray{T}(undef, (3, 3))
@test B[1:2, 1] isa Inti.BlockVector
@test B[1:2, 1:2] isa Inti.BlockMatrix
@test 2 * B isa Inti.BlockArray

T = SMatrix{3,3,Float64,9}
B = Inti.BlockArray{T}(undef, (3, 3))
x = rand(SVector{3,Float64}, 3)
B * x
