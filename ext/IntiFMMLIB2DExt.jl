module IntiFMMLIB2DExt

import Inti
import FMMLIB2D
import LinearMaps

function __init__()
    @info "Loading Inti.jl FMMLIB2D extension"
end

function Inti.assemble_fmm(
    iop::Inti.IntegralOperator{VT,KT,TT,ST};
    atol = sqrt(eps()),
) where {VT,KT,TT,ST<:Inti.Quadrature{2,Float64}}
    # unpack the necessary fields in the appropriate format
    m, n = size(iop)
    sources = Matrix{Float64}(undef, 2, n)
    for j in 1:n
        sources[:, j] = Inti.coords(iop.source[j])
    end
    if iop.source != iop.target
        targets = Matrix{Float64}(undef, 2, m)
        for i in 1:m
            targets[:, i] = Inti.coords(iop.target[i])
        end
    end
    weights = [q.weight for q in iop.source]
    K = iop.kernel
    # Laplace
    if K isa Inti.SingleLayerKernel{Float64,Inti.Laplace{2}}
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = -1 / (2 * π) * weights * x
            # FMMLIB2D does no checking for if targets are also sources
            if iop.source == iop.target
                out = FMMLIB2D.rfmm2d(; source = sources, charge = charges, tol = atol)
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    charge = charges,
                    target = targets,
                    tol = atol,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.DoubleLayerKernel{Float64,Inti.Laplace{2}}
        normals = Matrix{Float64}(undef, 2, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = -1 / (2 * π) * view(normals, :, j) * weights[j]
            end
            dipstr = Vector{Float64}(undef, n)
            for j in 1:n
                dipstr[j] = x[j]
            end
            # FMMLIB2D does no checking for if targets are also sources
            if iop.source == iop.target
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    dipstr = dipstr,
                    dipvec = dipvecs,
                    tol = atol,
                )
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    target = targets,
                    dipstr = dipstr,
                    dipvec = dipvecs,
                    tol = atol,
                )
                return copyto!(y, out.pottarg)
            end
        end
        # Helmholtz
    elseif K isa Inti.SingleLayerKernel{ComplexF64,<:Inti.Helmholtz{2}}
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            # FMMLIB2D does no checking for if targets are also sources
            if iop.source == iop.target
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    charge = charges,
                    tol = atol,
                )
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    charge = charges,
                    target = targets,
                    tol = atol,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.DoubleLayerKernel{ComplexF64,<:Inti.Helmholtz{2}}
        normals = Matrix{Float64}(undef, 2, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals, Float64)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = view(normals, :, j) * weights[j]
            end
            # FMMLIB2D does no checking for if targets are also sources
            if iop.source == iop.target
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    dipstr = x,
                    dipvec = dipvecs,
                    tol = atol,
                )
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    target = targets,
                    dipstr = x,
                    dipvec = dipvecs,
                    tol = atol,
                )
                return copyto!(y, out.pottarg)
            end
        end
    else
        error("integral operator not supported by Inti's FMMLIB2D wrapper")
    end
end

end # module
