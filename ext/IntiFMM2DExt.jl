
module IntiFMM2DExt

import Inti
import FMM2D
import LinearMaps

function __init__()
    @info "Loading Inti.jl FMM2D extension"
end

function Inti._assemble_fmm2d(iop::Inti.IntegralOperator; rtol = sqrt(eps()))
    sources, targets = _fmm2d_get_sources_and_targets(iop)
    weights = _fmm2d_get_weights(iop)
    m, n = size(iop)
    same_surface =
        m == n ? isapprox(targets, sources; atol = Inti.SAME_POINT_TOLERANCE) : false
    return _fmm2d_map(iop.kernel, sources, targets, weights, iop, rtol, Val(same_surface))
end

# Helper functions
function _fmm2d_get_sources_and_targets(iop)
    m, n = size(iop)
    targets = Matrix{Float64}(undef, 2, m)
    for i in 1:m
        targets[:, i] = Inti.coords(iop.target[i])
    end
    sources = Matrix{Float64}(undef, 2, n)
    for j in 1:n
        sources[:, j] = Inti.coords(iop.source[j])
    end
    return sources, targets
end

function _fmm2d_get_weights(iop)
    return [q.weight for q in iop.source]
end

function _fmm2d_get_target_normals(iop)
    m, _ = size(iop)
    normals = Matrix{Float64}(undef, 2, m)
    for i in 1:m
        normals[:, i] = Inti.normal(iop.target[i])
    end
    return normals
end

function _fmm2d_get_source_normals(iop)
    _, n = size(iop)
    normals = Matrix{Float64}(undef, 2, n)
    for i in 1:n
        normals[:, i] = Inti.normal(iop.source[i])
    end
    return normals
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
        out = _rfmm2d(sources, targets, charges, rtol, same_surface, Val(1))
        return copyto!(y, _get_potential(out, same_surface))
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
    normals = _fmm2d_get_source_normals(iop)
    dipvecs = similar(normals)
    dipstr = Vector{Float64}(undef, n)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = -1 / (2 * π) * view(normals, :, j) * weights[j]
        end
        @. dipstr = x
        out = _rfmm2d(sources, targets, dipstr, dipvecs, rtol, same_surface, Val(1))
        return copyto!(y, _get_potential(out, same_surface))
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
    xnormals = _fmm2d_get_target_normals(iop)
    charges = Vector{Float64}(undef, n)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        @. charges = -1 / (2 * π) * weights * x
        out = _rfmm2d(sources, targets, charges, rtol, same_surface, Val(2))
        return copyto!(
            y,
            sum(xnormals .* _get_gradient(out, same_surface); dims = 1) |> vec,
        )
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
    xnormals = _fmm2d_get_target_normals(iop)
    ynormals = _fmm2d_get_source_normals(iop)
    dipvecs = similar(ynormals, Float64)
    dipstrs = Vector{Float64}(undef, n)
    return LinearMaps.LinearMap{Float64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = -1 / (2 * π) * view(ynormals, :, j) * weights[j]
        end
        @. dipstrs = x
        out = _rfmm2d(sources, targets, dipstrs, dipvecs, rtol, same_surface, Val(2))
        return copyto!(
            y,
            sum(xnormals .* _get_gradient(out, same_surface); dims = 1) |> vec,
        )
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
        out = _hfmm2d(zk, sources, targets, charges, rtol, same_surface, Val(1))
        return copyto!(y, _get_potential(out, same_surface))
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
    normals = _fmm2d_get_source_normals(iop)
    dipvecs = similar(normals, Float64)
    dipstrs = Vector{ComplexF64}(undef, n)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = view(normals, :, j) * weights[j]
        end
        @. dipstrs = x
        out = _hfmm2d(zk, sources, targets, dipstrs, dipvecs, rtol, same_surface, Val(1))
        return copyto!(y, _get_potential(out, same_surface))
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
    xnormals = _fmm2d_get_target_normals(iop)
    charges = Vector{ComplexF64}(undef, n)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        @. charges = x * weights
        out = _hfmm2d(zk, sources, targets, charges, rtol, same_surface, Val(2))
        return copyto!(
            y,
            sum(xnormals .* _get_gradient(out, same_surface); dims = 1) |> vec,
        )
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
    xnormals = _fmm2d_get_target_normals(iop)
    ynormals = _fmm2d_get_source_normals(iop)
    dipvecs = similar(ynormals, Float64)
    dipstrs = Vector{ComplexF64}(undef, n)
    zk = ComplexF64(K.op.k)
    return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
        for j in 1:n
            dipvecs[:, j] = view(ynormals, :, j) * weights[j]
        end
        @. dipstrs = x
        out = _hfmm2d(zk, sources, targets, dipstrs, dipvecs, rtol, same_surface, Val(2))
        return copyto!(
            y,
            sum(xnormals .* _get_gradient(out, same_surface); dims = 1) |> vec,
        )
    end
end

function _fmm2d_map(K, sources, targets, weights, iop, rtol, same_surface)
    return error("integral operator not supported by Inti's FMM2D wrapper")
end

# FMM2D wrappers
_get_potential(out, ::Val{true}) = out.pot
_get_potential(out, ::Val{false}) = out.pottarg
_get_gradient(out, ::Val{true}) = out.grad
_get_gradient(out, ::Val{false}) = out.gradtarg

function _rfmm2d(sources, targets, charges, rtol, ::Val{true}, pg::Val{V}) where {V}
    return FMM2D.rfmm2d(; sources = sources, charges = charges, eps = rtol, pg = V)
end

function _rfmm2d(sources, targets, charges, rtol, ::Val{false}, pg::Val{V}) where {V}
    return FMM2D.rfmm2d(;
        sources = sources,
        charges = charges,
        targets = targets,
        eps = rtol,
        pgt = V,
    )
end

function _rfmm2d(sources, targets, dipstr, dipvecs, rtol, ::Val{true}, pg::Val{V}) where {V}
    return FMM2D.rfmm2d(;
        sources = sources,
        dipstr = dipstr,
        dipvecs = dipvecs,
        eps = rtol,
        pg = V,
    )
end

function _rfmm2d(
    sources,
    targets,
    dipstr,
    dipvecs,
    rtol,
    ::Val{false},
    pg::Val{V},
) where {V}
    return FMM2D.rfmm2d(;
        sources = sources,
        targets = targets,
        dipstr = dipstr,
        dipvecs = dipvecs,
        eps = rtol,
        pgt = V,
    )
end

function _hfmm2d(zk, sources, targets, charges, rtol, ::Val{true}, pg::Val{V}) where {V}
    return FMM2D.hfmm2d(; zk = zk, sources = sources, charges = charges, eps = rtol, pg = V)
end

function _hfmm2d(zk, sources, targets, charges, rtol, ::Val{false}, pg::Val{V}) where {V}
    return FMM2D.hfmm2d(;
        zk = zk,
        sources = sources,
        charges = charges,
        targets = targets,
        eps = rtol,
        pgt = V,
    )
end

function _hfmm2d(
    zk,
    sources,
    targets,
    dipstr,
    dipvecs,
    rtol,
    ::Val{true},
    pg::Val{V},
) where {V}
    return FMM2D.hfmm2d(;
        zk = zk,
        sources = sources,
        dipstr = dipstr,
        dipvecs = dipvecs,
        eps = rtol,
        pg = V,
    )
end

function _hfmm2d(
    zk,
    sources,
    targets,
    dipstr,
    dipvecs,
    rtol,
    ::Val{false},
    pg::Val{V},
) where {V}
    return FMM2D.hfmm2d(;
        zk = zk,
        sources = sources,
        targets = targets,
        dipstr = dipstr,
        dipvecs = dipvecs,
        eps = rtol,
        pgt = V,
    )
end

end # module
