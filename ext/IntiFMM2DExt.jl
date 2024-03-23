module IntiFMM2DExt

import Inti
import FMM2D
import LinearMaps

function __init__()
    @info "Loading Inti.jl FMM2D extension"
end

function Inti._assemble_fmm2d(iop::Inti.IntegralOperator; atol = sqrt(eps()))
    # unpack the necessary fields in the appropriate format
    m, n = size(iop)
    sources = Matrix{Float64}(undef, 2, n)
    for j in 1:n
        sources[:, j] = Inti.coords(iop.source[j])
    end
    targets = Matrix{Float64}(undef, 2, m)
    for i in 1:m
        targets[:, i] = Inti.coords(iop.target[i])
    end
    weights = [q.weight for q in iop.source]
    # This is really a hack to check if the entire set of targets and sources coincide.
    # If only some of the targets and coincide overlap, garbage will be returned.
    same_surface =
        m == n ? isapprox(targets, sources; atol = Inti.SAME_POINT_TOLERANCE) : false
    K = iop.kernel
    # Laplace
    if K isa Inti.SingleLayerKernel{Float64,<:Inti.Laplace{2}}
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = -1 / (2 * π) * weights * x
            if same_surface
                out =
                    FMM2D.rfmm2d(; sources = sources, charges = charges, eps = atol, pg = 1)
                return copyto!(y, out.pot)
            else
                out = FMM2D.rfmm2d(;
                    sources = sources,
                    charges = charges,
                    targets = targets,
                    eps     = atol,
                    pgt     = 1,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.DoubleLayerKernel{Float64,<:Inti.Laplace{2}}
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
            if same_surface
                out = FMM2D.rfmm2d(;
                    sources = sources,
                    dipstr  = dipstr,
                    dipvecs = dipvecs,
                    eps     = atol,
                    pg      = 1,
                )
                return copyto!(y, out.pot)
            else
                out = FMM2D.rfmm2d(;
                    sources = sources,
                    targets = targets,
                    dipstr  = dipstr,
                    dipvecs = dipvecs,
                    eps     = atol,
                    pgt     = 1,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.AdjointDoubleLayerKernel{Float64,<:Inti.Laplace{2}}
        xnormals = Matrix{Float64}(undef, 2, m)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = -1 / (2 * π) * weights * x
            if same_surface
                out =
                    FMM2D.rfmm2d(; charges = charges, sources = sources, eps     = atol, pg      = 2)
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMM2D.rfmm2d(;
                    charges = charges,
                    sources = sources,
                    targets = targets,
                    eps     = atol,
                    pgt     = 2,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    elseif K isa Inti.HyperSingularKernel{Float64,<:Inti.Laplace{2}}
        xnormals = Matrix{Float64}(undef, 2, m)
        ynormals = Matrix{Float64}(undef, 2, n)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        for j in 1:n
            ynormals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(ynormals, Float64)
        dipstrs = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = -1 / (2 * π) * view(ynormals, :, j) * weights[j]
            end
            for j in 1:n
                dipstrs[j] = x[j]
            end
            if same_surface
                out = FMM2D.rfmm2d(;
                    dipvecs = dipvecs,
                    dipstr  = dipstrs,
                    sources = sources,
                    eps     = atol,
                    pg      = 2,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMM2D.rfmm2d(;
                    dipvecs = dipvecs,
                    dipstr  = dipstrs,
                    sources = sources,
                    targets = targets,
                    eps     = atol,
                    pgt     = 2,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
        # Helmholtz
    elseif K isa Inti.SingleLayerKernel{ComplexF64,<:Inti.Helmholtz{2}}
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            if same_surface
                out = FMM2D.hfmm2d(;
                    zk      = zk,
                    sources = sources,
                    charges = charges,
                    eps     = atol,
                    pg      = 1,
                )
                return copyto!(y, out.pot)
            else
                out = FMM2D.hfmm2d(;
                    zk      = zk,
                    sources = sources,
                    charges = charges,
                    targets = targets,
                    eps     = atol,
                    pgt     = 1,
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
        dipstrs = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = view(normals, :, j) * weights[j]
            end
            for j in 1:n
                dipstrs[j] = x[j]
            end
            if same_surface
                out = FMM2D.hfmm2d(;
                    zk      = zk,
                    sources = sources,
                    dipstr  = dipstrs,
                    dipvecs = dipvecs,
                    eps     = atol,
                    pg      = 1,
                )
                return copyto!(y, out.pot)
            else
                out = FMM2D.hfmm2d(;
                    zk      = zk,
                    sources = sources,
                    targets = targets,
                    dipstr  = dipstrs,
                    dipvecs = dipvecs,
                    eps     = atol,
                    pgt     = 1,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.AdjointDoubleLayerKernel{ComplexF64,<:Inti.Helmholtz{2}}
        xnormals = Matrix{Float64}(undef, 2, m)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights
            @. charges = x * weights
            if same_surface
                out = FMM2D.hfmm2d(;
                    zk      = zk,
                    charges = charges,
                    sources = sources,
                    eps     = atol,
                    pg      = 2,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMM2D.hfmm2d(;
                    zk      = zk,
                    charges = charges,
                    sources = sources,
                    targets = targets,
                    eps     = atol,
                    pgt     = 2,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    elseif K isa Inti.HyperSingularKernel{ComplexF64,<:Inti.Helmholtz{2}}
        xnormals = Matrix{Float64}(undef, 2, m)
        ynormals = Matrix{Float64}(undef, 2, n)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        for j in 1:n
            ynormals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(ynormals, Float64)
        dipstrs = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.pde.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = view(ynormals, :, j) * weights[j]
            end
            for j in 1:n
                dipstrs[j] = x[j]
            end
            if same_surface
                out = FMM2D.hfmm2d(;
                    zk      = zk,
                    dipvecs = dipvecs,
                    dipstr  = dipstrs,
                    sources = sources,
                    eps     = atol,
                    pg      = 2,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMM2D.hfmm2d(;
                    zk      = zk,
                    dipvecs = dipvecs,
                    dipstr  = dipstrs,
                    sources = sources,
                    targets = targets,
                    eps     = atol,
                    pgt     = 2,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    else
        error("integral operator not supported by Inti's FMM2D wrapper")
    end
end

end # module
