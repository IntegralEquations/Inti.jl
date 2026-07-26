module IntiFMM2DExt

import Inti
import FMM2D
import LinearMaps
using StaticArrays
using StaticArrays # For Stokes types

function __init__()
    return @debug "Loading Inti.jl FMM2D extension"
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
    if K isa Inti.SingleLayerKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.DoubleLayerKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.AdjointDoubleLayerKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.HyperSingularKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.GradientSingleLayerKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.GradientDoubleLayerKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.SourceGradientSingleLayerKernel{<:Inti.Laplace{2}}
        # ∫∇yG(x,y)⋅g = Σ wⱼ gⱼ⋅∇yG : contraction of ∇yG and dipoles with vector
        # strengths g. The W operator
        #   W[g] = -∫∇yG⋅g
        # applies the leading minus when this map is assembled.
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
    elseif K isa Inti.HessianKernel{<:Inti.Laplace{2}}
        # Charge→Hessian realization of the 'Hessian' volume operator used in constructing the
        # `X = ∇W` VDIM correction: a scalar density `ρ` maps to `∫∇ₓ∇ₓG(x,y)ρ(y)dy`
        # (a 2×2 `SMatrix` per target). `rfmm2d` returns the 3 unique second derivatives per point as `(3,·)` in
        # the order ∂xx, ∂xy, ∂yy;
        if K.charge_dipole == :charge
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
        else
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
        end
        # Helmholtz
    elseif K isa Inti.SingleLayerKernel{<:Inti.Helmholtz{2}}
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
    elseif K isa Inti.DoubleLayerKernel{<:Inti.Helmholtz{2}}
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
    elseif K isa Inti.AdjointDoubleLayerKernel{<:Inti.Helmholtz{2}}
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
    elseif K isa Inti.HyperSingularKernel{<:Inti.Helmholtz{2}}
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
    elseif K isa Inti.GradientSingleLayerKernel{<:Inti.Helmholtz{2}}
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
    elseif K isa Inti.GradientDoubleLayerKernel{<:Inti.Helmholtz{2}}
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

    elseif K isa Inti.SourceGradientSingleLayerKernel{<:Inti.Helmholtz{2}}
        # ∫∇yG⋅g = Σ wⱼ gⱼ⋅∇yG :  contraction of ∇yG and dipoles with vector
        # strengths g. The W operator
        #   W[g] = -∫∇yG⋅g
        # applies the leading minus when this map is assembled.
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
    elseif K isa Inti.HessianKernel{<:Inti.Helmholtz{2}}
        # Charge→Hessian realization of the 'Hessian' volume operator used in constructing the
        # `X = ∇W` VDIM correction: a scalar density `ρ` maps to `∫∇ₓ∇ₓG(x,y)ρ(y)dy`
        # (a 2×2 `SMatrix` per target). `hfmm2d` returns the 3 unique second derivatives per point as `(3,·)` in
        # the order ∂xx, ∂xy, ∂yy;
        if K.charge_dipole == :charge
            #`hess`/`hesstarg` is (3,·) = ∂xx, ∂xy, ∂yy.
            charges = Vector{ComplexF64}(undef, n)
            zk = ComplexF64(K.op.k)
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
            # X_forward = +∫∇ₓ∇ₓG⋅g = -∇ₓ(∫∇yG⋅g). The bracket quantity is the
            # ∇yG-dipole field (W forward) and the negative sign is incorporated
            # into the dipole strengths. hfmm2d needs real dipvecs with complex
            # strengths, so the complex density is split into real/imag parts
            # (dipstr = wⱼ resp. i·wⱼ).
        else
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
        end
        # Stokes
    elseif K isa Inti.SingleLayerKernel{<:Inti.Stokes{2}}
        T = SVector{2, Float64}
        stoklet = Matrix{Float64}(undef, 2, n)
        return LinearMaps.LinearMap{SMatrix{2, 2, Float64, 4}}(m, n) do y, x
            # FMM2D returns the raw Stokeslet sum (without the 1/2π scaling), so
            # Inti's single-layer kernel = pot / (2π μ). Fold the constant and the
            # quadrature weights into the Stokeslet strengths.
            stoklet[:] = 1 / (2 * π * K.op.μ) .* reinterpret(Float64, weights .* x)
            if same_surface
                out = FMM2D.stfmm2d(; eps = rtol, sources = sources, stoklet = stoklet, ppreg = 1)
                return copyto!(y, reinterpret(T, out.pot))
            else
                out = FMM2D.stfmm2d(;
                    eps = rtol,
                    sources = sources,
                    stoklet = stoklet,
                    targets = targets,
                    ppregt = 1,
                )
                return copyto!(y, reinterpret(T, out.pottarg))
            end
        end
    elseif K isa Inti.DoubleLayerKernel{<:Inti.Stokes{2}}
        T = SVector{2, Float64}
        normals = Matrix{Float64}(undef, 2, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        # FMM2D returns the raw stresslet sum T_ijk μ_j ν_k (without the 1/2π
        # scaling), and Inti's double-layer kernel = -pot / (2π). Fold the constant
        # and the quadrature weights into the stresslet orientation vectors, and the
        # density into the stresslet strengths.
        strsvec = similar(normals, Float64)
        strslet = similar(normals, Float64)
        for j in 1:n
            strsvec[:, j] = -1 / (2 * π) * view(normals, :, j) .* weights[j]
        end
        return LinearMaps.LinearMap{SMatrix{2, 2, Float64, 4}}(m, n) do y, x
            strslet[:] = reinterpret(Float64, x)
            if same_surface
                out = FMM2D.stfmm2d(;
                    eps = rtol,
                    sources = sources,
                    strslet = strslet,
                    strsvec = strsvec,
                    ppreg = 1,
                )
                return copyto!(y, reinterpret(T, out.pot))
            else
                out = FMM2D.stfmm2d(;
                    eps = rtol,
                    sources = sources,
                    strslet = strslet,
                    strsvec = strsvec,
                    targets = targets,
                    ppregt = 1,
                )
                return copyto!(y, reinterpret(T, out.pottarg))
            end
        end
    else
        error("integral operator not supported by Inti's FMM2D wrapper")
    end
end

end # module
