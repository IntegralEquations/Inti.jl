module IntiFMMLIB2DExt

import Inti
import FMMLIB2D
import LinearMaps

function __init__()
    @info "Loading Inti.jl FMMLIB2D extension"
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
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
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
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
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
    elseif K isa Inti.AdjointDoubleLayerKernel{Float64,<:Inti.Laplace{2}}
        xnormals = Matrix{Float64}(undef, 2, m)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = -1 / (2 * π) * weights * x
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.rfmm2d(;
                    charge = charges,
                    source = sources,
                    ifgrad = true,
                    tol = atol,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMMLIB2D.rfmm2d(;
                    charge = charges,
                    source = sources,
                    target = targets,
                    ifgradtarg = true,
                    tol = atol,
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
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.rfmm2d(;
                    dipvec = dipvecs,
                    dipstr = dipstrs,
                    source = sources,
                    ifgrad = true,
                    tol = atol,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMMLIB2D.rfmm2d(;
                    dipvec = dipvecs,
                    dipstr = dipstrs,
                    source = sources,
                    target = targets,
                    ifgradtarg = true,
                    tol = atol,
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
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
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
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    dipstr = dipstrs,
                    dipvec = dipvecs,
                    tol = atol,
                )
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    target = targets,
                    dipstr = dipstrs,
                    dipvec = dipvecs,
                    tol = atol,
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
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    charge = charges,
                    source = sources,
                    ifgrad = true,
                    tol = atol,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    charge = charges,
                    source = sources,
                    target = targets,
                    ifgradtarg = true,
                    tol = atol,
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
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    dipvec = dipvecs,
                    dipstr = dipstrs,
                    source = sources,
                    ifgrad = true,
                    tol = atol,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    dipvec = dipvecs,
                    dipstr = dipstrs,
                    source = sources,
                    target = targets,
                    ifgradtarg = true,
                    tol = atol,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    else
        error("integral operator not supported by Inti's FMMLIB2D wrapper")
    end
end

end # module
