module IntiFMM3DExt

import Inti
import FMM3D
import LinearMaps
using StaticArrays

function __init__()
    @info "Loading Inti.jl FMM3D extension"
end

function Inti._assemble_fmm3d(iop::Inti.IntegralOperator; rtol = sqrt(eps()))
    sources, targets = _fmm3d_get_sources_and_targets(iop)
    weights = _fmm3d_get_weights(iop)
    return _fmm3d_map(iop.kernel, sources, targets, weights, iop, rtol)
end

# Helper functions
function _fmm3d_get_sources_and_targets(iop)
    m, n = size(iop)
    targets = Matrix{Float64}(undef, 3, m)
    for i in 1:m
        targets[:, i] = Inti.coords(iop.target[i])
    end
    sources = Matrix{Float64}(undef, 3, n)
    for j in 1:n
        sources[:, j] = Inti.coords(iop.source[j])
    end
    return sources, targets
end

function _fmm3d_get_weights(iop)
    return [q.weight for q in iop.source]
end

function _fmm3d_get_target_normals(iop)
    m, _ = size(iop)
    normals = Matrix{Float64}(undef, 3, m)
    for i in 1:m
        normals[:, i] = Inti.normal(iop.target[i])
    end
    return normals
end

function _fmm3d_get_source_normals(iop)
    _, n = size(iop)
    normals = Matrix{Float64}(undef, 3, n)
    for i in 1:n
        normals[:, i] = Inti.normal(iop.source[i])
    end
    return normals
end

# Laplace
function _fmm3d_map(
    K::Inti.SingleLayerKernel{Float64,<:Inti.Laplace{3}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    charges = Vector{Float64}(undef, n)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        @. charges = 1 / (4 * π) * weights * x
        out = FMM3D.lfmm3d(rtol, sources; charges, targets, pgt = 1)
        return copyto!(y, out.pottarg)
    end
end

function _fmm3d_map(
    K::Inti.DoubleLayerKernel{Float64,<:Inti.Laplace{3}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    normals = _fmm3d_get_source_normals(iop)
    dipvecs = similar(normals)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = 1 / (4 * π) * view(normals, :, j) * x[j] * weights[j]
        end
        out = FMM3D.lfmm3d(rtol, sources; dipvecs, targets, pgt = 1)
        return copyto!(y, out.pottarg)
    end
end

function _fmm3d_map(
    K::Inti.AdjointDoubleLayerKernel{Float64,<:Inti.Laplace{3}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    xnormals = _fmm3d_get_target_normals(iop)
    charges = Vector{Float64}(undef, n)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        @. charges = 1 / (4 * π) * weights * x
        out = FMM3D.lfmm3d(rtol, sources; charges, targets, pgt = 2)
        return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
    end
end

function _fmm3d_map(
    K::Inti.HyperSingularKernel{Float64,<:Inti.Laplace{3}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    xnormals = _fmm3d_get_target_normals(iop)
    ynormals = _fmm3d_get_source_normals(iop)
    dipvecs = similar(ynormals, Float64)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = 1 / (4 * π) * view(ynormals, :, j) * x[j] * weights[j]
        end
        out = FMM3D.lfmm3d(rtol, sources; dipvecs, targets, pgt = 2)
        return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
    end
end

# Helmholtz
function _fmm3d_map(
    K::Inti.SingleLayerKernel{ComplexF64,<:Inti.Helmholtz{3}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    charges = Vector{ComplexF64}(undef, n)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        @. charges = 1 / (4 * π) * weights * x
        out = FMM3D.hfmm3d(rtol, zk, sources; charges, targets, pgt = 1)
        return copyto!(y, out.pottarg)
    end
end

function _fmm3d_map(
    K::Inti.DoubleLayerKernel{ComplexF64,<:Inti.Helmholtz{3}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    normals = _fmm3d_get_source_normals(iop)
    dipvecs = similar(normals, ComplexF64)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = 1 / (4 * π) * view(normals, :, j) * x[j] * weights[j]
        end
        out = FMM3D.hfmm3d(rtol, zk, sources; dipvecs, targets, pgt = 1)
        return copyto!(y, out.pottarg)
    end
end

function _fmm3d_map(
    K::Inti.AdjointDoubleLayerKernel{ComplexF64,<:Inti.Helmholtz{3}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    xnormals = _fmm3d_get_target_normals(iop)
    charges = Vector{ComplexF64}(undef, n)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        @. charges = 1 / (4 * π) * weights * x
        out = FMM3D.hfmm3d(rtol, zk, sources; charges, targets, pgt = 2)
        return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
    end
end

function _fmm3d_map(
    K::Inti.HyperSingularKernel{ComplexF64,<:Inti.Helmholtz{3}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    xnormals = _fmm3d_get_target_normals(iop)
    ynormals = _fmm3d_get_source_normals(iop)
    dipvecs = similar(ynormals, ComplexF64)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = 1 / (4 * π) * view(ynormals, :, j) * x[j] * weights[j]
        end
        out = FMM3D.hfmm3d(rtol, zk, sources; dipvecs, targets, pgt = 2)
        return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
    end
end

# Stokes
function _fmm3d_map(
    K::Inti.SingleLayerKernel{SMatrix{3,3,Float64,9},<:Inti.Stokes{3,Float64}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    T = SVector{3,Float64}
    stoklet = Matrix{Float64}(undef, 3, n)
    return LinearMaps.LinearMap{SMatrix{3,3,Float64,9}}(m, n) do y, x
        stoklet[:] = 1 / (4 * π * K.op.μ) .* reinterpret(Float64, weights .* x)
        out = FMM3D.stfmm3d(rtol, sources; stoklet, targets, ppregt = 1)
        return copyto!(y, reinterpret(T, out.pottarg))
    end
end

function _fmm3d_map(
    K::Inti.DoubleLayerKernel{SMatrix{3,3,Float64,9},<:Inti.Stokes{3,Float64}},
    sources,
    targets,
    weights,
    iop,
    rtol,
)
    m, n = size(iop)
    T = SVector{3,Float64}
    normals = _fmm3d_get_source_normals(iop)
    strsvec = similar(normals, Float64)
    strslet = similar(normals, Float64)
    for j in 1:n
        strsvec[:, j] = -1 / (4 * π) * view(normals, :, j) .* weights[j]
    end
    return LinearMaps.LinearMap{SMatrix{3,3,Float64,9}}(m, n) do y, x
        strslet[:] = reinterpret(Float64, x)
        out = FMM3D.stfmm3d(rtol, sources; strslet, strsvec, targets, ppregt = 1)
        return copyto!(y, reinterpret(T, out.pottarg))
    end
end

function _fmm3d_map(K, sources, targets, weights, iop, rtol)
    return error("integral operator not supported by Inti's FMM3D wrapper")
end

end # module
