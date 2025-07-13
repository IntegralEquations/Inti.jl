module IntiFMM2DExt

import Inti
import FMM2D
import LinearMaps

function __init__()
    @info "Loading Inti.jl FMM2D extension"
end

function Inti._assemble_fmm2d(iop::Inti.IntegralOperator; rtol = sqrt(eps()))
    m, n = size(iop)
    targets = Matrix{Float64}(undef, 2, m)
    for i in 1:m
        targets[:, i] = Inti.coords(iop.target[i])
    end
    sources = Matrix{Float64}(undef, 2, n)
    for j in 1:n
        sources[:, j] = Inti.coords(iop.source[j])
    end
    weights = [q.weight for q in iop.source]
    same_surface =
        m == n ? isapprox(targets, sources; atol = Inti.SAME_POINT_TOLERANCE) : false
    return _fmm2d_map(iop.kernel, sources, targets, weights, iop, rtol, Val(same_surface))
end

# Laplace
function _fmm2d_map(
    K::Inti.SingleLayerKernel{Float64,<:Inti.Laplace{2}},
    sources,
    targets,
    weights,
    iop,
    rtol,
    same_surface,
)
    m, n = size(iop)
    charges = Vector{Float64}(undef, n)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        @. charges = -1 / (2 * π) * weights * x
        if same_surface == Val(true)
            out = FMM2D.rfmm2d(; sources = sources, charges = charges, eps = rtol, pg = 1)
            return copyto!(y, out.pot)
        else
            out = FMM2D.rfmm2d(;
                sources = sources,
                charges = charges,
                targets = targets,
                eps = rtol,
                pgt = 1,
            )
            return copyto!(y, out.pottarg)
        end
    end
end

function _fmm2d_map(
    K::Inti.DoubleLayerKernel{Float64,<:Inti.Laplace{2}},
    sources,
    targets,
    weights,
    iop,
    rtol,
    same_surface,
)
    m, n = size(iop)
    normals = Matrix{Float64}(undef, 2, n)
    for i in 1:n
        normals[:, i] = Inti.normal(iop.source[i])
    end
    dipvecs = similar(normals)
    dipstr = Vector{Float64}(undef, n)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = -1 / (2 * π) * view(normals, :, j) * weights[j]
        end
        @. dipstr = x
        if same_surface == Val(true)
            out = FMM2D.rfmm2d(;
                sources = sources,
                dipstr = dipstr,
                dipvecs = dipvecs,
                eps = rtol,
                pg = 1,
            )
            return copyto!(y, out.pot)
        else
            out = FMM2D.rfmm2d(;
                sources = sources,
                targets = targets,
                dipstr = dipstr,
                dipvecs = dipvecs,
                eps = rtol,
                pgt = 1,
            )
            return copyto!(y, out.pottarg)
        end
    end
end

function _fmm2d_map(
    K::Inti.AdjointDoubleLayerKernel{Float64,<:Inti.Laplace{2}},
    sources,
    targets,
    weights,
    iop,
    rtol,
    same_surface,
)
    m, n = size(iop)
    xnormals = Matrix{Float64}(undef, 2, m)
    for i in 1:m
        xnormals[:, i] = Inti.normal(iop.target[i])
    end
    charges = Vector{Float64}(undef, n)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        @. charges = -1 / (2 * π) * weights * x
        if same_surface == Val(true)
            out = FMM2D.rfmm2d(; sources = sources, charges = charges, eps = rtol, pg = 2)
            grad = out.grad
        else
            out = FMM2D.rfmm2d(;
                sources = sources,
                charges = charges,
                targets = targets,
                eps = rtol,
                pgt = 2,
            )
            grad = out.gradtarg
        end
        return copyto!(y, sum(xnormals .* grad; dims = 1) |> vec)
    end
end

function _fmm2d_map(
    K::Inti.HyperSingularKernel{Float64,<:Inti.Laplace{2}},
    sources,
    targets,
    weights,
    iop,
    rtol,
    same_surface,
)
    m, n = size(iop)
    xnormals = Matrix{Float64}(undef, 2, m)
    for i in 1:m
        xnormals[:, i] = Inti.normal(iop.target[i])
    end
    ynormals = Matrix{Float64}(undef, 2, n)
    for i in 1:n
        ynormals[:, i] = Inti.normal(iop.source[i])
    end
    dipvecs = similar(ynormals, Float64)
    dipstrs = Vector{Float64}(undef, n)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = -1 / (2 * π) * view(ynormals, :, j) * weights[j]
        end
        @. dipstrs = x
        if same_surface == Val(true)
            out = FMM2D.rfmm2d(;
                sources = sources,
                dipstr = dipstrs,
                dipvecs = dipvecs,
                eps = rtol,
                pg = 2,
            )
            grad = out.grad
        else
            out = FMM2D.rfmm2d(;
                sources = sources,
                targets = targets,
                dipstr = dipstrs,
                dipvecs = dipvecs,
                eps = rtol,
                pgt = 2,
            )
            grad = out.gradtarg
        end
        return copyto!(y, sum(xnormals .* grad; dims = 1) |> vec)
    end
end

# Helmholtz
function _fmm2d_map(
    K::Inti.SingleLayerKernel{ComplexF64,<:Inti.Helmholtz{2}},
    sources,
    targets,
    weights,
    iop,
    rtol,
    same_surface,
)
    m, n = size(iop)
    charges = Vector{ComplexF64}(undef, n)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        @. charges = weights * x
        if same_surface == Val(true)
            out =
                FMM2D.hfmm2d(; zk = zk, sources = sources, charges = charges, eps = rtol, pg = 1)
            return copyto!(y, out.pot)
        else
            out = FMM2D.hfmm2d(;
                zk = zk,
                sources = sources,
                charges = charges,
                targets = targets,
                eps = rtol,
                pgt = 1,
            )
            return copyto!(y, out.pottarg)
        end
    end
end

function _fmm2d_map(
    K::Inti.DoubleLayerKernel{ComplexF64,<:Inti.Helmholtz{2}},
    sources,
    targets,
    weights,
    iop,
    rtol,
    same_surface,
)
    m, n = size(iop)
    normals = Matrix{Float64}(undef, 2, n)
    for i in 1:n
        normals[:, i] = Inti.normal(iop.source[i])
    end
    dipvecs = similar(normals, Float64)
    dipstrs = Vector{ComplexF64}(undef, n)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = view(normals, :, j) * weights[j]
        end
        @. dipstrs = x
        if same_surface == Val(true)
            out = FMM2D.hfmm2d(;
                zk = zk,
                sources = sources,
                dipstr = dipstrs,
                dipvecs = dipvecs,
                eps = rtol,
                pg = 1,
            )
            return copyto!(y, out.pot)
        else
            out = FMM2D.hfmm2d(;
                zk = zk,
                sources = sources,
                targets = targets,
                dipstr = dipstrs,
                dipvecs = dipvecs,
                eps = rtol,
                pgt = 1,
            )
            return copyto!(y, out.pottarg)
        end
    end
end

function _fmm2d_map(
    K::Inti.AdjointDoubleLayerKernel{ComplexF64,<:Inti.Helmholtz{2}},
    sources,
    targets,
    weights,
    iop,
    rtol,
    same_surface,
)
    m, n = size(iop)
    xnormals = Matrix{Float64}(undef, 2, m)
    for i in 1:m
        xnormals[:, i] = Inti.normal(iop.target[i])
    end
    charges = Vector{ComplexF64}(undef, n)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        @. charges = x * weights
        if same_surface == Val(true)
            out =
                FMM2D.hfmm2d(; zk = zk, sources = sources, charges = charges, eps = rtol, pg = 2)
            grad = out.grad
        else
            out = FMM2D.hfmm2d(;
                zk = zk,
                sources = sources,
                charges = charges,
                targets = targets,
                eps = rtol,
                pgt = 2,
            )
            grad = out.gradtarg
        end
        return copyto!(y, sum(xnormals .* grad; dims = 1) |> vec)
    end
end

function _fmm2d_map(
    K::Inti.HyperSingularKernel{ComplexF64,<:Inti.Helmholtz{2}},
    sources,
    targets,
    weights,
    iop,
    rtol,
    same_surface,
)
    m, n = size(iop)
    xnormals = Matrix{Float64}(undef, 2, m)
    for i in 1:m
        xnormals[:, i] = Inti.normal(iop.target[i])
    end
    ynormals = Matrix{Float64}(undef, 2, n)
    for i in 1:n
        ynormals[:, i] = Inti.normal(iop.source[i])
    end
    dipvecs = similar(ynormals, Float64)
    dipstrs = Vector{ComplexF64}(undef, n)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = view(ynormals, :, j) * weights[j]
        end
        @. dipstrs = x
        if same_surface == Val(true)
            out = FMM2D.hfmm2d(;
                zk = zk,
                sources = sources,
                dipstr = dipstrs,
                dipvecs = dipvecs,
                eps = rtol,
                pg = 2,
            )
            grad = out.grad
        else
            out = FMM2D.hfmm2d(;
                zk = zk,
                sources = sources,
                targets = targets,
                dipstr = dipstrs,
                dipvecs = dipvecs,
                eps = rtol,
                pgt = 2,
            )
            grad = out.gradtarg
        end
        return copyto!(y, sum(xnormals .* grad; dims = 1) |> vec)
    end
end

function _fmm2d_map(K, sources, targets, weights, iop, rtol, same_surface)
    return error("integral operator not supported by Inti's FMM2D wrapper")
end

end # module