module IntiFMM2DExt

import Inti
import FMM2D
import LinearMaps
using StaticArrays

function __init__()
    return @info "Loading Inti.jl FMM2D extension"
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
            if same_surface
                out =
                    FMM2D.rfmm2d(; sources = sources, charges = charges, eps = rtol, pg = 1)
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
            if same_surface
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
    elseif K isa Inti.AdjointDoubleLayerKernel{Float64, <:Inti.Laplace{2}}
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
                    FMM2D.rfmm2d(; charges = charges, sources = sources, eps = rtol, pg = 2)
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMM2D.rfmm2d(;
                    charges = charges,
                    sources = sources,
                    targets = targets,
                    eps = rtol,
                    pgt = 2,
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
            if same_surface
                out = FMM2D.rfmm2d(;
                    dipvecs = dipvecs,
                    dipstr = dipstrs,
                    sources = sources,
                    eps = rtol,
                    pg = 2,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMM2D.rfmm2d(;
                    dipvecs = dipvecs,
                    dipstr = dipstrs,
                    sources = sources,
                    targets = targets,
                    eps = rtol,
                    pgt = 2,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    elseif K isa Inti.GradientSingleLayerKernel{<:SVector{2}, <:Inti.Laplace{2}}
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{SVector{2, Float64}}(m, n) do y, x
            # multiply by weights and constant
            @. charges = -1 / (2 * π) * weights * x
            if same_surface
                out =
                    FMM2D.rfmm2d(; sources = sources, charges = charges, eps = rtol, pg = 2)
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.grad)))
            else
                out = FMM2D.rfmm2d(;
                    sources = sources,
                    charges = charges,
                    targets = targets,
                    eps = rtol,
                    pgt = 2,
                )
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.gradtarg)))
            end
        end
    elseif K isa Inti.GradientDoubleLayerKernel{<:SVector{2}, <:Inti.Laplace{2}}
        normals = Matrix{Float64}(undef, 2, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals)
        return LinearMaps.LinearMap{SVector{2, Float64}}(m, n) do y, x
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
                    dipstr = dipstr,
                    dipvecs = dipvecs,
                    eps = rtol,
                    pg = 2,
                )
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.grad)))
            else
                out = FMM2D.rfmm2d(;
                    sources = sources,
                    targets = targets,
                    dipstr = dipstr,
                    dipvecs = dipvecs,
                    eps = rtol,
                    pgt = 2,
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
            if same_surface
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
    elseif K isa Inti.HessianSingleLayerKernel{<:Any, <:Inti.Laplace{2}}
        # X forward (PV part) = +∫∇ₓ∇ₓG⋅g = -∇ₓ(∫∇yG⋅g). The bracket is the
        # `SourceGradientSingleLayer` dipole field above; its target-gradient (`gradtarg`,
        # pgt = 2) is ∫∇ₓ∇yG⋅g = -X_forward. Negating the dipole strengths relative to the
        # W branch folds in the leading minus, so `gradtarg` is the `SVector` output.
        dipvecs = Matrix{Float64}(undef, 2, n)
        dipstr = ones(Float64, n)
        return LinearMaps.LinearMap{SVector{2, Float64}}(m, n) do y, x
            for j in 1:n
                dipvecs[:, j] = (1 / (2 * π)) * x[j] * weights[j]
            end
            if same_surface
                out = FMM2D.rfmm2d(;
                    sources = sources,
                    dipstr = dipstr,
                    dipvecs = dipvecs,
                    eps = rtol,
                    pg = 2,
                )
                return copyto!(y, reinterpret(SVector{2, Float64}, vec(out.grad)))
            else
                out = FMM2D.rfmm2d(;
                    sources = sources,
                    targets = targets,
                    dipstr = dipstr,
                    dipvecs = dipvecs,
                    eps = rtol,
                    pgt = 2,
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
            if same_surface
                out = FMM2D.hfmm2d(;
                    zk = zk,
                    sources = sources,
                    charges = charges,
                    eps = rtol,
                    pg = 1,
                )
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
            if same_surface
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
            if same_surface
                out = FMM2D.hfmm2d(;
                    zk = zk,
                    charges = charges,
                    sources = sources,
                    eps = rtol,
                    pg = 2,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMM2D.hfmm2d(;
                    zk = zk,
                    charges = charges,
                    sources = sources,
                    targets = targets,
                    eps = rtol,
                    pgt = 2,
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
            if same_surface
                out = FMM2D.hfmm2d(;
                    zk = zk,
                    dipvecs = dipvecs,
                    dipstr = dipstrs,
                    sources = sources,
                    eps = rtol,
                    pg = 2,
                )
                return copyto!(y, sum(xnormals .* out.grad; dims = 1) |> vec)
            else
                out = FMM2D.hfmm2d(;
                    zk = zk,
                    dipvecs = dipvecs,
                    dipstr = dipstrs,
                    sources = sources,
                    targets = targets,
                    eps = rtol,
                    pgt = 2,
                )
                return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
            end
        end
    elseif K isa Inti.GradientSingleLayerKernel{<:SVector{2}, <:Inti.Helmholtz{2}}
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{SVector{2, ComplexF64}}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            if same_surface
                out = FMM2D.hfmm2d(;
                    zk = zk,
                    sources = sources,
                    charges = charges,
                    eps = rtol,
                    pg = 2,
                )
                return copyto!(y, reinterpret(SVector{2, ComplexF64}, vec(out.grad)))
            else
                out = FMM2D.hfmm2d(;
                    zk = zk,
                    sources = sources,
                    charges = charges,
                    targets = targets,
                    eps = rtol,
                    pgt = 2,
                )
                return copyto!(y, reinterpret(SVector{2, ComplexF64}, vec(out.gradtarg)))
            end
        end
    elseif K isa Inti.GradientDoubleLayerKernel{<:SVector{2}, <:Inti.Helmholtz{2}}
        normals = Matrix{Float64}(undef, 2, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals, Float64)
        dipstrs = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{SVector{2, ComplexF64}}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = view(normals, :, j) * weights[j]
            end
            for j in 1:n
                dipstrs[j] = x[j]
            end
            if same_surface
                out = FMM2D.hfmm2d(;
                    zk = zk,
                    sources = sources,
                    dipstr = dipstrs,
                    dipvecs = dipvecs,
                    eps = rtol,
                    pg = 2,
                )
                return copyto!(y, reinterpret(SVector{2, ComplexF64}, vec(out.grad)))
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
                return copyto!(y, reinterpret(SVector{2, ComplexF64}, vec(out.gradtarg)))
            end
        end

    elseif K isa Inti.SourceGradientSingleLayerKernel{<:Any, <:Inti.Helmholtz{2}}
        # ∫∇yG⋅g = Σ wⱼ gⱼ⋅∇yG : dipoles with vector strengths g. The W operator
        # W[g] = -∫∇yG⋅g applies the leading minus when this map is assembled.
        # FMM2D's `hfmm2d` requires real dipole directions (with a complex
        # strength), so the complex density is split into real/imaginary parts:
        #   Re(g): dipvec=Re(gⱼ), dipstr=wⱼ ;  Im(g): dipvec=Im(gⱼ), dipstr=i wⱼ.
        dipvecs = Matrix{Float64}(undef, 2, n)
        dipstr = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            fill!(y, 0)
            for (getpart, strconst) in ((real, one(ComplexF64)), (imag, 1im))
                for j in 1:n
                    dipvecs[:, j] = getpart.(x[j])
                    dipstr[j] = strconst * weights[j]
                end
                if same_surface
                    out = FMM2D.hfmm2d(;
                        zk = zk,
                        sources = sources,
                        dipstr = dipstr,
                        dipvecs = dipvecs,
                        eps = rtol,
                        pg = 1,
                    )
                    y .+= out.pot
                else
                    out = FMM2D.hfmm2d(;
                        zk = zk,
                        sources = sources,
                        targets = targets,
                        dipstr = dipstr,
                        dipvecs = dipvecs,
                        eps = rtol,
                        pgt = 1,
                    )
                    y .+= out.pottarg
                end
            end
            return y
        end
    elseif K isa Inti.HessianSingleLayerKernel{<:Any, <:Inti.Helmholtz{2}}
        # X forward = +∫∇ₓ∇ₓG⋅g = -∇ₓ(∫∇yG⋅g). The bracket is the ∇yG-dipole field (W
        # forward); its target-gradient is ∫∇ₓ∇yG⋅g = -X_forward, so the dipole strengths
        # are negated and `grad`/`gradtarg` is the output. As for the W forward, hfmm2d
        # needs real dipvecs with complex strengths, so the complex density is split into
        # real/imag parts (dipstr = wⱼ resp. i·wⱼ).
        dipvecs = Matrix{Float64}(undef, 2, n)
        dipstr = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{SVector{2, ComplexF64}}(m, n) do y, x
            fill!(y, zero(SVector{2, ComplexF64}))
            for (getpart, strconst) in ((real, one(ComplexF64)), (imag, 1im))
                for j in 1:n
                    dipvecs[:, j] = -getpart.(x[j])
                    dipstr[j] = strconst * weights[j]
                end
                if same_surface
                    out = FMM2D.hfmm2d(;
                        zk = zk, sources = sources, dipstr = dipstr, dipvecs = dipvecs,
                        eps = rtol, pg = 2,
                    )
                    y .+= reinterpret(SVector{2, ComplexF64}, vec(out.grad))
                else
                    out = FMM2D.hfmm2d(;
                        zk = zk, sources = sources, targets = targets, dipstr = dipstr,
                        dipvecs = dipvecs, eps = rtol, pgt = 2,
                    )
                    y .+= reinterpret(SVector{2, ComplexF64}, vec(out.gradtarg))
                end
            end
            return y
        end
    else
        error("integral operator not supported by Inti's FMM2D wrapper")
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
            if same_surface
                out = FMM2D.rfmm2d(; sources = sources, charges = charges, eps = rtol, pg = 3)
                H = out.hess        # (3, n): ∂xx, ∂xy, ∂yy
            else
                out = FMM2D.rfmm2d(;
                    sources = sources,
                    charges = charges,
                    targets = targets,
                    eps = rtol,
                    pgt = 3,
                )
                H = out.hesstarg
            end
            @inbounds for i in 1:m
                y[i] = SMatrix{2, 2, Float64, 4}(H[1, i], H[2, i], H[2, i], H[3, i])
            end
            return y
        end
    elseif K isa Inti.HessianSingleLayerKernel{<:Any, <:Inti.Helmholtz{2}}
        # hfmm2d bakes in the (i/4)H₀⁽¹⁾ prefactor, so charges carry no extra constant
        # (same as the Helmholtz single layer). `hess`/`hesstarg` is (3,·) = ∂xx, ∂xy, ∂yy.
        zk = ComplexF64(K.op.k)
        charges = Vector{ComplexF64}(undef, n)
        return LinearMaps.LinearMap{SMatrix{2, 2, ComplexF64, 4}}(m, n) do y, x
            @. charges = weights * x
            if same_surface
                out = FMM2D.hfmm2d(; zk = zk, sources = sources, charges = charges, eps = rtol, pg = 3)
                H = out.hess
            else
                out = FMM2D.hfmm2d(;
                    zk = zk,
                    sources = sources,
                    charges = charges,
                    targets = targets,
                    eps = rtol,
                    pgt = 3,
                )
                H = out.hesstarg
            end
            @inbounds for i in 1:m
                y[i] = SMatrix{2, 2, ComplexF64, 4}(H[1, i], H[2, i], H[2, i], H[3, i])
            end
            return y
        end
    else
        error("Inti's FMM2D charge→Hessian wrapper only supports Laplace/Helmholtz 2D")
    end
end

end # module
