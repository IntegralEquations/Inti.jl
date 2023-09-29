#=

Utility functions that have nowhere obvious to go.

=#

"""
    svector(f,n)

Create an `SVector` of length n, computing each element as f(i), where i is the
index of the element.
"""
svector(f,n)=  ntuple(f,n) |> SVector
