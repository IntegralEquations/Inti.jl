module IntiFMMLIB2DExt

import Inti
import FMMLIB2D
import LinearMaps
using StaticArrays

function __init__()
    return @debug "Loading Inti.jl FMMLIB2D extension"
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
    elseif K isa Inti.AdjointDoubleLayerKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.GradientSingleLayerKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.GradientDoubleLayerKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.SourceGradientSingleLayerKernel{<:Inti.Laplace{2}}
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
    elseif K isa Inti.HessianKernel{<:Inti.Laplace{2}}
        # Charge→Hessian realization of the 'Hessian' volume operator used in constructing the
        # `X = ∇W` VDIM correction: a scalar density `ρ` maps to `∫∇ₓ∇ₓG(x,y)ρ(y)dy`
        # (a 2×2 `SMatrix` per target). `rfmm2d` returns the 3 unique second derivatives per
        # point as `(3,·)` in the order ∂xx, ∂xy, ∂yy; Inti's Laplace 2D kernel carries the
        # −1/(2π) prefactor, folded into the charges exactly as for the single layer.
        if K.charge_dipole == :charge
            charges = Vector{Float64}(undef, n)
            return LinearMaps.LinearMap{SMatrix{2, 2, Float64, 4}}(m, n) do y, x
                @. charges = -1 / (2 * π) * weights * x
                # FMMLIB2D does no checking for if targets are also sources
                if same_surface
                    out = FMMLIB2D.rfmm2d(;
                        source = sources,
                        charge = charges,
                        ifhess = true,
                        tol = rtol,
                    )
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
            # X forward (PV part) = +∫∇ₓ∇ₓG⋅g = -∇ₓ(∫∇yG⋅g): dipoles with vector strengths g,
            # whose target-gradient is ∫∇ₓ∇yG⋅g = -X_forward, so the strengths are negated
            # (the +1/(2π) Laplace 2D prefactor folds in) and `grad`/`gradtarg` is the
            # `SVector` output directly.
        else
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
        end
        # Helmholtz
    elseif K isa Inti.SingleLayerKernel{<:Inti.Helmholtz{2}}
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
    elseif K isa Inti.GradientSingleLayerKernel{<:Inti.Helmholtz{2}}
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{SVector{2, ComplexF64}}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    charge = charges,
                    ifgrad = true,
                    tol = rtol,
                )
                return copyto!(y, reinterpret(SVector{2, ComplexF64}, vec(out.grad)))
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    charge = charges,
                    target = targets,
                    ifgradtarg = true,
                    tol = rtol,
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
            # FMMLIB2D does no checking for if targets are also sources
            if same_surface
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    dipstr = dipstrs,
                    dipvec = dipvecs,
                    ifgrad = true,
                    tol = rtol,
                )
                return copyto!(y, reinterpret(SVector{2, ComplexF64}, vec(out.grad)))
            else
                out = FMMLIB2D.hfmm2d(;
                    zk = zk,
                    source = sources,
                    target = targets,
                    dipstr = dipstrs,
                    dipvec = dipvecs,
                    ifgradtarg = true,
                    tol = rtol,
                )
                return copyto!(y, reinterpret(SVector{2, ComplexF64}, vec(out.gradtarg)))
            end
        end
    elseif K isa Inti.SourceGradientSingleLayerKernel{<:Inti.Helmholtz{2}}
        # ∫∇yG⋅g = Σ wⱼ gⱼ⋅∇yG : contraction of ∇yG and dipoles with vector
        # strengths g. The W operator
        #   W[g] = -∫∇yG⋅g
        # applies the leading minus when this map is assembled.
        # FMMLIB2D's `hfmm2d` requires real dipole directions (with a complex
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
                # FMMLIB2D does no checking for if targets are also sources
                if same_surface
                    out = FMMLIB2D.hfmm2d(;
                        zk = zk,
                        source = sources,
                        dipstr = dipstr,
                        dipvec = dipvecs,
                        tol = rtol,
                    )
                    y .+= out.pot
                else
                    out = FMMLIB2D.hfmm2d(;
                        zk = zk,
                        source = sources,
                        target = targets,
                        dipstr = dipstr,
                        dipvec = dipvecs,
                        tol = rtol,
                    )
                    y .+= out.pottarg
                end
            end
            return y
        end
    elseif K isa Inti.HessianKernel{<:Inti.Helmholtz{2}}
        # Charge→Hessian realization of the 'Hessian' volume operator used in constructing the
        # `X = ∇W` VDIM correction: a scalar density `ρ` maps to `∫∇ₓ∇ₓG(x,y)ρ(y)dy`
        # (a 2×2 `SMatrix` per target). `hfmm2d` returns the 3 unique second derivatives per
        # point as `(3,·)` in the order ∂xx, ∂xy, ∂yy.
        if K.charge_dipole == :charge
            #`hess`/`hesstarg` is (3,·) = ∂xx, ∂xy, ∂yy.
            charges = Vector{ComplexF64}(undef, n)
            zk = ComplexF64(K.op.k)
            return LinearMaps.LinearMap{SMatrix{2, 2, ComplexF64, 4}}(m, n) do y, x
                @. charges = weights * x
                # FMMLIB2D does no checking for if targets are also sources
                if same_surface
                    out = FMMLIB2D.hfmm2d(;
                        zk = zk,
                        source = sources,
                        charge = charges,
                        ifhess = true,
                        tol = rtol,
                    )
                    H = out.hess
                else
                    out = FMMLIB2D.hfmm2d(;
                        zk = zk,
                        source = sources,
                        charge = charges,
                        target = targets,
                        ifhesstarg = true,
                        tol = rtol,
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
                    # FMMLIB2D does no checking for if targets are also sources
                    if same_surface
                        out = FMMLIB2D.hfmm2d(;
                            zk = zk,
                            source = sources,
                            dipstr = dipstr,
                            dipvec = dipvecs,
                            ifgrad = true,
                            tol = rtol,
                        )
                        y .+= reinterpret(SVector{2, ComplexF64}, vec(out.grad))
                    else
                        out = FMMLIB2D.hfmm2d(;
                            zk = zk,
                            source = sources,
                            target = targets,
                            dipstr = dipstr,
                            dipvec = dipvecs,
                            ifgradtarg = true,
                            tol = rtol,
                        )
                        y .+= reinterpret(SVector{2, ComplexF64}, vec(out.gradtarg))
                    end
                end
                return y
            end
        end
    else
        error("integral operator not supported by Inti's FMMLIB2D wrapper")
    end
end

end # module
