module IntiFMM3DExt

import Inti
import FMM3D
import LinearMaps

function __init__()
    @info "Loading Inti.jl FMM3D extension"
end

function Inti._assemble_fmm3d(iop::Inti.IntegralOperator; rtol = sqrt(eps()))
    # unpack the necessary fields in the appropriate format
    m, n = size(iop)
    targets = Matrix{Float64}(undef, 3, m)
    for i in 1:m
        targets[:, i] = Inti.coords(iop.target[i])
    end
    sources = Matrix{Float64}(undef, 3, n)
    for j in 1:n
        sources[:, j] = Inti.coords(iop.source[j])
    end
    weights = [q.weight for q in iop.source]
    K = iop.kernel
    # Laplace
    if K isa Inti.SingleLayerKernel{Float64,<:Inti.Laplace{3}}
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = 1 / (4 * π) * weights * x
            out = FMM3D.lfmm3d(rtol, sources; charges, targets, pgt = 1)
            return copyto!(y, out.pottarg)
        end
    elseif K isa Inti.DoubleLayerKernel{Float64,<:Inti.Laplace{3}}
        normals = Matrix{Float64}(undef, 3, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = 1 / (4 * π) * view(normals, :, j) * x[j] * weights[j]
            end
            out = FMM3D.lfmm3d(rtol, sources; dipvecs, targets, pgt = 1)
            return copyto!(y, out.pottarg)
        end
    elseif K isa Inti.AdjointDoubleLayerKernel{Float64,<:Inti.Laplace{3}}
        xnormals = Matrix{Float64}(undef, 3, m)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = 1 / (4 * π) * weights * x
            out = FMM3D.lfmm3d(rtol, sources; charges, targets, pgt = 2)
            return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
        end
    elseif K isa Inti.HyperSingularKernel{Float64,<:Inti.Laplace{3}}
        xnormals = Matrix{Float64}(undef, 3, m)
        ynormals = Matrix{Float64}(undef, 3, n)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        for j in 1:n
            ynormals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(ynormals, Float64)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = 1 / (4 * π) * view(ynormals, :, j) * x[j] * weights[j]
            end
            out = FMM3D.lfmm3d(rtol, sources; dipvecs, targets, pgt = 2)
            return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
        end
        # Helmholtz
    elseif K isa Inti.SingleLayerKernel{ComplexF64,<:Inti.Helmholtz{3}}
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = 1 / (4 * π) * weights * x
            out = FMM3D.hfmm3d(rtol, zk, sources; charges, targets, pgt = 1)
            return copyto!(y, out.pottarg)
        end
    elseif K isa Inti.DoubleLayerKernel{ComplexF64,<:Inti.Helmholtz{3}}
        normals = Matrix{Float64}(undef, 3, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals, ComplexF64)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = 1 / (4 * π) * view(normals, :, j) * x[j] * weights[j]
            end
            out = FMM3D.hfmm3d(rtol, zk, sources; dipvecs, targets, pgt = 1)
            return copyto!(y, out.pottarg)
        end
    elseif K isa Inti.AdjointDoubleLayerKernel{ComplexF64,<:Inti.Helmholtz{3}}
        xnormals = Matrix{Float64}(undef, 3, m)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = 1 / (4 * π) * weights * x
            out = FMM3D.hfmm3d(rtol, zk, sources; charges, targets, pgt = 2)
            return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
        end
    elseif K isa Inti.HyperSingularKernel{ComplexF64,<:Inti.Helmholtz{3}}
        xnormals = Matrix{Float64}(undef, 3, m)
        ynormals = Matrix{Float64}(undef, 3, n)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        for j in 1:n
            ynormals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(ynormals, ComplexF64)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = 1 / (4 * π) * view(ynormals, :, j) * x[j] * weights[j]
            end
            out = FMM3D.hfmm3d(rtol, zk, sources; dipvecs, targets, pgt = 2)
            return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
        end
    else
        error("integral operator not supported by Inti's FMM3D wrapper")
    end
end

end # module
