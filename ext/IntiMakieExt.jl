module IntiMakieExt

import Makie
import Inti
using StaticArrays

function __init__()
    @info "Loading Inti.jl Makie extension"
end

function tomakie_dim1(msh::Inti.AbstractMesh{N,T}) where {N,T}
    coords = Makie.Point{N,T}[]
    NAN    = Inti.svector(i -> NaN * one(T), N)
    for E in Inti.element_types(msh)
        iter = Inti.elements(msh, E)
        D    = Inti.domain(E)
        @assert D isa Inti.ReferenceLine
        vtxs = Inti.vertices(D)
        for el in iter
            for vtx in vtxs
                push!(coords, el(vtx))
            end
            # trick from Meshes.jl. Interleave with NaNs to plot segments
            push!(coords, NAN)
        end
    end
    return coords
end

function Makie.convert_arguments(::Type{<:Makie.Lines}, msh::Inti.AbstractMesh)
    @assert Inti.geometric_dimension(msh) == 1 "Lines only supported for meshes of geometric dimension 1"
    coords = tomakie_dim1(msh)
    return (coords,)
end

function Makie.convert_arguments(P::Type{<:Makie.Poly}, msh::Inti.AbstractMesh)
    connec = Inti.triangle_connectivity(msh)
    return Makie.convert_arguments(P, Inti.nodes(msh), connec)
end

end # module
