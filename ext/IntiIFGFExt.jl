module IntiIFGFExt

import Inti
import IFGF
import LinearMaps
using LinearAlgebra
using StaticArrays

function __init__()
    @info "Loading Inti.jl IFGF extension"
end

function Inti._assemble_ifgf(iop::Inti.IntegralOperator; atol = sqrt(eps()))
    N = Inti.ambient_dimension(iop.source)

    # unpack the necessary fields in the appropriate format
    m, n = size(iop)
    targets = Vector{SVector{N,Float64}}(undef, m)
    for i in 1:m
        targets[i] = Inti.coords(iop.target[i])
    end
    sources = Vector{SVector{N,Float64}}(undef, n)
    for j in 1:n
        sources[j] = Inti.coords(iop.source[j])
    end
    weights = [q.weight for q in iop.source]
    K = iop.kernel

    # Map the Inti kernel to the IFGF namespace counterpart
    # IFGF doesn't handle the constant factors, so prepare to pre-multiply
    if K.pde isa Inti.Laplace{N}
        ifgf_pde = IFGF.Laplace(; dim = N)
        prefac = N == 2 ? -1 / (2π) : 1 / (4π)
    elseif K.pde isa Inti.Helmholtz{N}
        ifgf_pde = IFGF.Helmholtz(; dim = N, k = K.pde.k)
        prefac = N == 2 ? im / 4 : 1 / (4π)
    end

    if K isa Inti.SingleLayerKernel
        T = eltype(iop)
        L = IFGF.plan_forward_map(ifgf_pde, targets, sources; tol = atol, charges = true)
        charges = Vector{T}(undef, n)
        return LinearMaps.LinearMap{T}(m, n) do y, x
            # multiply by weights and constant
            @. charges = prefac * weights * x
            return copyto!(y, IFGF.forward_map(L; charges = charges))
        end
    elseif K isa Inti.DoubleLayerKernel
        T = eltype(iop)
        L = IFGF.plan_forward_map(ifgf_pde, targets, sources; tol = atol, dipvecs = true)
        normals = Vector{SVector{N,T}}(undef, n)
        for j in 1:n
            normals[j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals)
        return LinearMaps.LinearMap{T}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[j] = prefac * normals[j] * weights[j] * x[j]
            end
            return copyto!(y, IFGF.forward_map(L; dipvecs = dipvecs))
        end
    elseif K isa Inti.AdjointDoubleLayerKernel
        T = eltype(iop)
        L = IFGF.plan_forward_map(
            ifgf_pde,
            targets,
            sources;
            tol = atol,
            charges = true,
            grad = true,
        )
        xnormals = Vector{SVector{N,T}}(undef, m)
        for j in 1:m
            xnormals[j] = Inti.normal(iop.target[j])
        end
        charges = Vector{T}(undef, n)
        return LinearMaps.LinearMap{T}(m, n) do y, x
            # multiply by weights and constant
            @. charges = prefac * weights * x
            return copyto!(y, dot.(xnormals, IFGF.forward_map(L; charges = charges)))
        end
    elseif K isa Inti.HyperSingularKernel
        T = eltype(iop)
        L = IFGF.plan_forward_map(
            ifgf_pde,
            targets,
            sources;
            tol = atol,
            dipvecs = true,
            grad = true,
        )
        xnormals = Vector{SVector{N,T}}(undef, m)
        ynormals = Vector{SVector{N,T}}(undef, n)
        for j in 1:m
            xnormals[j] = Inti.normal(iop.target[j])
        end
        for j in 1:n
            ynormals[j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(ynormals)
        return LinearMaps.LinearMap{T}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[j] = prefac * ynormals[j] * weights[j] * x[j]
            end
            return copyto!(y, dot.(xnormals, IFGF.forward_map(L; dipvecs = dipvecs)))
        end
    else
        error("integral operator not supported by Inti's IFGF wrapper")
    end
end

end # module
