module IntiMakieExt

import Makie
import Inti
using StaticArrays

function __init__()
    @info "Loading Inti.jl Makie extension"
end

function tomakie_dim1(msh::Inti.AbstractMesh{N,T}) where {N,T}
    coords = Makie.Point{N,T}[]
    NAN    = svector(i -> NaN * one(T), N)
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

function tomakie_dim2(msh::Inti.AbstractMesh{N,T}) where {N,T}
    coords = Makie.Point{N,T}[]
    connec = Int[]
    for E in Inti.element_types(msh)
        iter = Inti.elements(msh, E)
        D = Inti.domain(E)
        if D isa Inti.ReferenceTriangle
            vtxs = Inti.vertices(D)
            for el in iter
                for vtx in vtxs
                    push!(coords, el(vtx))
                    push!(connec, length(coords))
                end
            end
        elseif D isa ReferenceSquare
            # split square in two triangles for visualization
            vtxs_down = Inti.vertices(Inti.ReferenceTriangle())
            vtxs_up   = map(v -> -v .+ 1, vtx_down)
            for el in iter
                for vtxs in (vtxs_down, vtxs_up) # the two triangles
                    for vtx in vtxs
                        push!(coords, el(vtx))
                        push!(connec, length(coords))
                    end
                end
            end
        end
    end
    return coords, connec
end

function tomakie_dim3(msh::Inti.AbstractMesh{N,T}) where {N,T}
    coords = Makie.Point{N,T}[]
    connec = Int[]
    for E in Inti.element_types(msh)
        iter = Inti.elements(msh, E)
        D = Inti.domain(E)
        @assert D isa Inti.ReferenceTetrahedron
        vtxs = Inti.vertices(D)
        for el in iter
            for nf in 1:4 # four faces
                for (i, vtx) in enumerate(vtxs)
                    i == nf && continue # i-th face exclude the i-th vertex
                    push!(coords, el(vtx))
                    push!(connec, length(coords))
                end
            end
        end
    end
    return coords, connec
end

function Makie.convert_arguments(P::Type{<:Makie.Lines}, msh::Inti.AbstractMesh)
    @assert Inti.geometric_dimension(msh) == 1 "Lines only supported for meshes of geometric dimension 1"
    coords = tomakie_dim1(msh)
    return (coords,)
end

function Makie.convert_arguments(P::Type{<:Makie.Poly}, msh::Inti.AbstractMesh)
    gdim = Inti.geometric_dimension(msh)
    if gdim == 2
        coords, connec = tomakie_dim2(msh)
    elseif gdim == 3
        coords, connec = tomakie_dim3(msh)
    else
        error("Poly only supported for meshes of geometric dimension 2 or 3")
    end
    return Makie.convert_arguments(P, coords, connec)
end

function Makie.convert_arguments(
    P::Type{<:Makie.Arrows},
    msh::Inti.AbstractMesh{N,T},
) where {N,T}
    gdim = Inti.geometric_dimension(msh)
    adim = Inti.ambient_dimension(msh)
    codim = adim - gdim
    @assert codim == 1 "Arrows only supported for meshes of codimension 1"
    coords  = Makie.Point{N,T}[]
    normals = Makie.Point{N,T}[]
    for E in Inti.element_types(msh)
        iter = Inti.elements(msh, E)
        dom = Inti.domain(E)
        xc = Inti.center(dom)
        for el in iter
            push!(coords, el(xc))
            push!(normals, normal(el, xc))
        end
    end
    return Makie.convert_arguments(P, coords, normals)
end

end # module
