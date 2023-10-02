#=

Utility functions that have nowhere obvious to go.

=#

"""
    svector(f,n)

Create an `SVector` of length n, computing each element as f(i), where i is the
index of the element.
"""
svector(f,n)=  ntuple(f,n) |> SVector

"""
    standard_basis_vector(k, ::Val{N})

Create an `SVector` of length N with a 1 in the kth position and zeros elsewhere.
"""
standard_basis_vector(k, n::Val{N}) where {N} = svector(i-> i == k ? 1 : 0, n)
