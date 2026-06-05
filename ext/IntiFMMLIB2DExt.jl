module IntiFMMLIB2DExt

import Inti
import FMMLIB2D
import LinearMaps
using StaticArrays

function __init__()
    return @info "Loading Inti.jl FMMLIB2D extension"
end

function Inti._assemble_fmm2d(iop::Inti.IntegralOperator; rtol = sqrt(eps()))
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
    if K isa Inti.SingleLayerKernel{Float64, <:Inti.Laplace{2}}
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = -1 / (2 * π) * weights * x
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.rfmm2d(; source = sources, charge = charges, tol = rtol)
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    charge = charges,
                    target = targets,
                    tol = rtol,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.DoubleLayerKernel{Float64, <:Inti.Laplace{2}}
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
                    tol = rtol,
                )
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    target = targets,
                    dipstr = dipstr,
                    dipvec = dipvecs,
                    tol = rtol,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.AdjointDoubleLayerKernel{Float64, <:Inti.Laplace{2}}
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
                    tol = rtol,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMMLIB2D.rfmm2d(;
                    charge = charges,
                    source = sources,
                    target = targets,
                    ifgradtarg = true,
                    tol = rtol,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    elseif K isa Inti.HyperSingularKernel{Float64, <:Inti.Laplace{2}}
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
                    tol = rtol,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMMLIB2D.rfmm2d(;
                    dipvec = dipvecs,
                    dipstr = dipstrs,
                    source = sources,
                    target = targets,
                    ifgradtarg = true,
                    tol = rtol,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    elseif K isa Inti.GradientSingleLayerKernel{<:SVector{2}, <:Inti.Laplace{2}}
        # ∇ₓG : charges with scalar strengths (SVector output = ∇ₓ of the single layer).
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{SVector{2, Float64}}(m, n) do y, x
            @. charges = -1 / (2 * π) * weights * x
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    charge = charges,
                    ifgrad = true,
                    tol = rtol,
                )
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.grad)))
            else
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    charge = charges,
                    target = targets,
                    ifgradtarg = true,
                    tol = rtol,
                )
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.gradtarg)))
            end
        end
    elseif K isa Inti.GradientDoubleLayerKernel{<:SVector{2}, <:Inti.Laplace{2}}
        # ∇ₓ∂_{n_y}G : dipoles (vec = source normal, str = density), SVector output.
        normals = Matrix{Float64}(undef, 2, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals)
        dipstr = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{SVector{2, Float64}}(m, n) do y, x
            for j in 1:n
                dipvecs[:, j] = -1 / (2 * π) * view(normals, :, j) * weights[j]
            end
            for j in 1:n
                dipstr[j] = x[j]
            end
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    dipstr = dipstr,
                    dipvec = dipvecs,
                    ifgrad = true,
                    tol = rtol,
                )
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.grad)))
            else
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    target = targets,
                    dipstr = dipstr,
                    dipvec = dipvecs,
                    ifgradtarg = true,
                    tol = rtol,
                )
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.gradtarg)))
            end
        end
    elseif K isa Inti.SourceGradientSingleLayerKernel{<:Any, <:Inti.Laplace{2}}
        # ∇yG(x,y)⋅g : dipoles with vector strengths g (scalar output). The W operator
        # W[g] = -∫∇yG⋅g applies the leading minus when this map is assembled.
        dipvecs = Matrix{Float64}(undef, 2, n)
        dipstr = ones(Float64, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            for j in 1:n
                dipvecs[:, j] = -(1 / (2 * π)) * x[j] * weights[j]
            end
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    dipstr = dipstr,
                    dipvec = dipvecs,
                    tol = rtol,
                )
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    target = targets,
                    dipstr = dipstr,
                    dipvec = dipvecs,
                    tol = rtol,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.HessianSingleLayerKernel{<:Any, <:Inti.Laplace{2}}
        # X forward (PV part) = +∫∇ₓ∇ₓG⋅g = -∇ₓ(∫∇yG⋅g): dipoles with vector strengths g,
        # whose target-gradient is ∫∇ₓ∇yG⋅g = -X_forward, so the strengths are negated
        # (the +1/(2π) Laplace 2D prefactor folds in) and `grad`/`gradtarg` is the
        # `SVector` output directly.
        dipvecs = Matrix{Float64}(undef, 2, n)
        dipstrs = ones(Float64, n)
        return LinearMaps.LinearMap{SVector{2, Float64}}(m, n) do y, x
            for j in 1:n
                dipvecs[:, j] = (1 / (2 * π)) * x[j] * weights[j]
            end
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    dipstr = dipstrs,
                    dipvec = dipvecs,
                    ifgrad = true,
                    tol = rtol,
                )
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.grad)))
            else
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    target = targets,
                    dipstr = dipstrs,
                    dipvec = dipvecs,
                    ifgradtarg = true,
                    tol = rtol,
                )
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.gradtarg)))
            end
        end
        # Helmholtz
    elseif K isa Inti.SingleLayerKernel{ComplexF64, <:Inti.Helmholtz{2}}
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    charge = charges,
                    tol = rtol,
                )
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    charge = charges,
                    target = targets,
                    tol = rtol,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.DoubleLayerKernel{ComplexF64, <:Inti.Helmholtz{2}}
        normals = Matrix{Float64}(undef, 2, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals, Float64)
        dipstrs = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
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
                    tol = rtol,
                )
                return copyto!(y, out.pot)
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    target = targets,
                    dipstr = dipstrs,
                    dipvec = dipvecs,
                    tol = rtol,
                )
                return copyto!(y, out.pottarg)
            end
        end
    elseif K isa Inti.AdjointDoubleLayerKernel{ComplexF64, <:Inti.Helmholtz{2}}
        xnormals = Matrix{Float64}(undef, 2, m)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
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
                    tol = rtol,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    charge = charges,
                    source = sources,
                    target = targets,
                    ifgradtarg = true,
                    tol = rtol,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    elseif K isa Inti.HyperSingularKernel{ComplexF64, <:Inti.Helmholtz{2}}
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
        zk = ComplexF64(K.op.k)
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
                    tol = rtol,
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
                    tol = rtol,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    else
        error("integral operator not supported by Inti's FMMLIB2D wrapper")
    end
end

# Charge→Hessian realization of the Hessian single-layer *volume* operator used by the
# `X = ∇W` VDIM correction in 2D: a scalar density `ρ` maps to `∫∇ₓ∇ₓG(x,y)ρ(y)dy`
# (a 2×2 `SMatrix` per target), i.e. the Hessian of the single-layer potential of
# charges `ρ`. `rfmm2d` returns the 3 unique second derivatives per point as `(3,·)` in
# the order ∂xx, ∂xy, ∂yy; Inti's Laplace 2D kernel carries the −1/(2π) prefactor, folded
# into the charges exactly as for the single layer.
function Inti._assemble_fmm2d_chargehessian(iop::Inti.IntegralOperator; rtol = sqrt(eps()))
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
    same_surface =
        m == n ? isapprox(targets, sources; atol = Inti.SAME_POINT_TOLERANCE) : false
    K = iop.kernel
    if K isa Inti.HessianSingleLayerKernel{<:Any, <:Inti.Laplace{2}}
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{SMatrix{2, 2, Float64, 4}}(m, n) do y, x
            @. charges = -1 / (2 * π) * weights * x
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.rfmm2d(; source = sources, charge = charges, ifhess = true, tol = rtol)
                H = out.hess        # (3, n): ∂xx, ∂xy, ∂yy
            else
                out = FMMLIB2D.rfmm2d(;
                    source = sources,
                    charge = charges,
                    target = targets,
                    ifhesstarg = true,
                    tol = rtol,
                )
                H = out.hesstarg
            end
            @inbounds for i in 1:m
                y[i] = SMatrix{2, 2, Float64, 4}(H[1, i], H[2, i], H[2, i], H[3, i])
            end
            return y
        end
    else
        error("Inti's FMMLIB2D charge→Hessian wrapper only supports Laplace 2D")
    end
end

end # module
