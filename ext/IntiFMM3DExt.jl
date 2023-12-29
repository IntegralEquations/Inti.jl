module IntiFMM3DExt

import Inti
import FMM3D
import LinearMaps

function __init__()
    @info "Loading Inti.jl FMM3D extension"
end

function Inti.assemble_fmm(iop::Inti.IntegralOperator; atol=sqrt(eps()))
    # unpack the necessary fields in the appropriate format
    m,n = size(iop)
    targets = Matrix{Float64}(undef, 3, m)
    for i in 1:m
        targets[:,i] = Inti.coords(iop.target[i])
    end
    sources = Matrix{Float64}(undef, 3, n)
    for j in 1:n
        sources[:,j] = Inti.coords(iop.source[j])
    end
    weights = [q.weight for q in iop.source]
    K = iop.kernel
    # Laplace
    if K isa Inti.SingleLayerKernel{Float64,Inti.Laplace{3}}
        charges = Vector{Float64}(undef,n)
        return LinearMaps.LinearMap{Float64}(m, n) do y,x
            # multiply by weights and constant
            @. charges = 1/(4*π) * weights * x
            out = FMM3D.lfmm3d(atol, sources; charges, targets, pgt=1)
            copyto!(y, out.pottarg)
        end
    elseif K isa Inti.DoubleLayerKernel{Float64, Inti.Laplace{3}}
        normals = Matrix{Float64}(undef, 3, n)
        for j in 1:n
            normals[:,j] = Inti.normal(iop.source[j])
        end
        dipvecs  = similar(normals)
        return LinearMaps.LinearMap{Float64}(m, n) do y,x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:,j] = 1/(4*π) * view(normals,:,j) * x[j] * weights[j]
            end
            out = FMM3D.lfmm3d(atol, sources; dipvecs, targets, pgt=1)
            copyto!(y, out.pottarg)
        end
    # Helmholtz
    elseif K isa Inti.SingleLayerKernel{ComplexF64,<:Inti.Helmholtz{3}}
        charges = Vector{ComplexF64}(undef,n)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y,x
            # multiply by weights and constant
            @. charges = 1/(4*π) * weights * x
            out = FMM3D.hfmm3d(atol, zk, sources; charges, targets, pgt=1)
            copyto!(y, out.pottarg)
        end
    elseif K isa Inti.DoubleLayerKernel{ComplexF64, <:Inti.Helmholtz{3}}
        normals = Matrix{Float64}(undef, 3, n)
        for j in 1:n
            normals[:,j] = Inti.normal(iop.source[j])
        end
        dipvecs  = similar(normals, ComplexF64)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y,x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:,j] = 1/(4*π) * view(normals,:,j) * x[j] * weights[j]
            end
            out = FMM3D.hfmm3d(atol, zk, sources; dipvecs, targets, pgt=1)
            copyto!(y, out.pottarg)
        end
    else
        error("integral operator not supported by Inti's FMM3D wrapper")
    end
end

end # module
