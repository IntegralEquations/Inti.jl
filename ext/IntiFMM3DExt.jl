module IntiFMM3DExt

import Inti
import FMM3D
import LinearMaps
using StaticArrays # For Stokes types

function __init__()
    return @info "Loading Inti.jl FMM3D extension"
end

function Inti._assemble_fmm3d(iop::Inti.IntegralOperator; rtol = sqrt(eps()), ndiv = nothing)
    # unpack the necessary fields in the appropriate format
    m, n = size(iop)
    targets = Matrix{Float64}(undef, 3, m)
    for i in 1:m
        targets[:, i] = Inti.coords(iop.target[i])
    end
    sources = Matrix{Float64}(undef, 3, n)
    for j in 1:n
        sources[:, j] = Inti.coords(iop.source[j])
    end
    weights = [q.weight for q in iop.source]
    K = iop.kernel
    # Laplace
    if K isa Inti.SingleLayerKernel{Float64, <:Inti.Laplace{3}}
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            if !isnothing(ndiv)
                out = FMM3D.lfmm3d_ndiv(rtol, sources, ndiv; charges, targets, pgt = 1)
            else
                out = FMM3D.lfmm3d(rtol, sources; charges, targets, pgt = 1)
            end
            return copyto!(y, out.pottarg)
        end
    elseif K isa Inti.DoubleLayerKernel{Float64, <:Inti.Laplace{3}}
        normals = Matrix{Float64}(undef, 3, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = view(normals, :, j) * x[j] * weights[j]
            end
            if !isnothing(ndiv)
                out = FMM3D.lfmm3d_ndiv(rtol, sources, ndiv; dipvecs, targets, pgt = 1)
            else
                out = FMM3D.lfmm3d(rtol, sources; dipvecs, targets, pgt = 1)
            end
            return copyto!(y, out.pottarg)
        end
    elseif K isa Inti.AdjointDoubleLayerKernel{Float64, <:Inti.Laplace{3}}
        xnormals = Matrix{Float64}(undef, 3, m)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            if !isnothing(ndiv)
                out = FMM3D.lfmm3d_ndiv(rtol, sources, ndiv; charges, targets, pgt = 2)
            else
                out = FMM3D.lfmm3d(rtol, sources; charges, targets, pgt = 2)
            end
            return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
        end
    elseif K isa Inti.HyperSingularKernel{Float64, <:Inti.Laplace{3}}
        xnormals = Matrix{Float64}(undef, 3, m)
        ynormals = Matrix{Float64}(undef, 3, n)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        for j in 1:n
            ynormals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(ynormals, Float64)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = view(ynormals, :, j) * x[j] * weights[j]
            end
            if !isnothing(ndiv)
                out = FMM3D.lfmm3d_ndiv(rtol, sources, ndiv; dipvecs, targets, pgt = 2)
            else
                out = FMM3D.lfmm3d(rtol, sources; dipvecs, targets, pgt = 2)
            end
            return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
        end
    elseif K isa Inti.GradientSingleLayerKernel{<:SVector{3}, <:Inti.Laplace{3}}
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{SVector{3, Float64}}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            if !isnothing(ndiv)
                out = FMM3D.lfmm3d_ndiv(rtol, sources, ndiv; charges, targets, pgt = 2)
            else
                out = FMM3D.lfmm3d(rtol, sources; charges, targets, pgt = 2)
            end
            return copyto!(y, reinterpret(SVector{3, Float64}, vec(out.gradtarg)))
        end
    elseif K isa Inti.GradientDoubleLayerKernel{<:SVector{3}, <:Inti.Laplace{3}}
        normals = Matrix{Float64}(undef, 3, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals)
        return LinearMaps.LinearMap{SVector{3, Float64}}(m, n) do y, x
            dipvecs .= normals .* transpose(x .* weights)
            if !isnothing(ndiv)
                out = FMM3D.lfmm3d_ndiv(rtol, sources, ndiv; dipvecs, targets, pgt = 2)
            else
                out = FMM3D.lfmm3d(rtol, sources; dipvecs, targets, pgt = 2)
            end
            return copyto!(y, reinterpret(SVector{3, Float64}, vec(out.gradtarg)))
        end
    elseif K isa Inti.SourceGradientSingleLayerKernel{<:Any, <:Inti.Laplace{3}}
        # ∇yG(x,y)⋅g : dipoles with vector strengths g (scalar output). The W operator
        # W[g] = -∫∇yG⋅g applies the leading minus when this map is assembled.
        dipvecs = Matrix{Float64}(undef, 3, n)
        return LinearMaps.LinearMap{Float64}(m, n) do y, x
            for j in 1:n
                dipvecs[:, j] = x[j] * weights[j]
            end
            if !isnothing(ndiv)
                out = FMM3D.lfmm3d_ndiv(rtol, sources, ndiv; dipvecs, targets, pgt = 1)
            else
                out = FMM3D.lfmm3d(rtol, sources; dipvecs, targets, pgt = 1)
            end
            return copyto!(y, out.pottarg)
        end
    elseif K isa Inti.HessianSingleLayerKernel{<:Any, <:Inti.Laplace{3}}
        # X forward (PV part) = +∫∇ₓ∇ₓG⋅g = -∇ₓ(∫∇yG⋅g). The bracket is exactly the
        # `SourceGradientSingleLayer` dipole field above (dipoles g, scalar potential),
        # so its target-gradient `gradtarg` (pgt = 2) is `∫∇ₓ∇yG⋅g = -X_forward`. Hence
        # the leading minus is folded into the dipole strengths and `gradtarg` is the
        # `SVector` output directly.
        dipvecs = Matrix{Float64}(undef, 3, n)
        return LinearMaps.LinearMap{SVector{3, Float64}}(m, n) do y, x
            for j in 1:n
                dipvecs[:, j] = -1.0 * x[j] * weights[j]
            end
            if !isnothing(ndiv)
                out = FMM3D.lfmm3d_ndiv(rtol, sources, ndiv; dipvecs, targets, pgt = 2)
            else
                out = FMM3D.lfmm3d(rtol, sources; dipvecs, targets, pgt = 2)
            end
            return copyto!(y, reinterpret(SVector{3, Float64}, vec(out.gradtarg)))
        end
        # Helmholtz
    elseif K isa Inti.SingleLayerKernel{ComplexF64, <:Inti.Helmholtz{3}}
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            if !isnothing(ndiv)
                out = FMM3D.hfmm3d_ndiv(rtol, zk, sources, ndiv; charges, targets, pgt = 1)
            else
                out = FMM3D.hfmm3d(rtol, zk, sources; charges, targets, pgt = 1)
            end
            return copyto!(y, out.pottarg)
        end
    elseif K isa Inti.DoubleLayerKernel{ComplexF64, <:Inti.Helmholtz{3}}
        normals = Matrix{Float64}(undef, 3, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals, ComplexF64)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = view(normals, :, j) * x[j] * weights[j]
            end
            if !isnothing(ndiv)
                out = FMM3D.hfmm3d_ndiv(rtol, zk, sources, ndiv; dipvecs, targets, pgt = 1)
            else
                out = FMM3D.hfmm3d(rtol, zk, sources; dipvecs, targets, pgt = 1)
            end
            return copyto!(y, out.pottarg)
        end
    elseif K isa Inti.AdjointDoubleLayerKernel{ComplexF64, <:Inti.Helmholtz{3}}
        xnormals = Matrix{Float64}(undef, 3, m)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            if !isnothing(ndiv)
                out = FMM3D.hfmm3d_ndiv(rtol, zk, sources; charges, targets, pgt = 2)
            else
                out = FMM3D.hfmm3d(rtol, zk, sources; charges, targets, pgt = 2)
            end
            return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
        end
    elseif K isa Inti.HyperSingularKernel{ComplexF64, <:Inti.Helmholtz{3}}
        xnormals = Matrix{Float64}(undef, 3, m)
        ynormals = Matrix{Float64}(undef, 3, n)
        for j in 1:m
            xnormals[:, j] = Inti.normal(iop.target[j])
        end
        for j in 1:n
            ynormals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(ynormals, ComplexF64)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            # multiply by weights and constant
            for j in 1:n
                dipvecs[:, j] = view(ynormals, :, j) * x[j] * weights[j]
            end
            if !isnothing(ndiv)
                out = FMM3D.hfmm3d_ndiv(rtol, zk, sources, ndiv; dipvecs, targets, pgt = 2)
            else
                out = FMM3D.hfmm3d(rtol, zk, sources; dipvecs, targets, pgt = 2)
            end
            return copyto!(y, sum(xnormals .* out.gradtarg; dims = 1) |> vec)
        end
    elseif K isa Inti.GradientSingleLayerKernel{<:SVector{3}, <:Inti.Helmholtz{3}}
        charges = Vector{ComplexF64}(undef, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{SVector{3, ComplexF64}}(m, n) do y, x
            # multiply by weights and constant
            @. charges = weights * x
            if !isnothing(ndiv)
                out = FMM3D.hfmm3d_ndiv(rtol, zk, sources, ndiv; charges, targets, pgt = 2)
            else
                out = FMM3D.hfmm3d(rtol, zk, sources; charges, targets, pgt = 2)
            end
            return copyto!(y, reinterpret(SVector{3, ComplexF64}, vec(out.gradtarg)))
        end
    elseif K isa Inti.GradientDoubleLayerKernel{<:SVector{3}, <:Inti.Helmholtz{3}}
        normals = Matrix{Float64}(undef, 3, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        dipvecs = similar(normals, ComplexF64)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{SVector{3, ComplexF64}}(m, n) do y, x
            dipvecs .= normals .* transpose(x .* weights)
            if !isnothing(ndiv)
                out = FMM3D.hfmm3d_ndiv(rtol, zk, sources, ndiv; dipvecs, targets, pgt = 2)
            else
                out = FMM3D.hfmm3d(rtol, zk, sources; dipvecs, targets, pgt = 2)
            end
            return copyto!(y, reinterpret(SVector{3, ComplexF64}, vec(out.gradtarg)))
        end
    elseif K isa Inti.SourceGradientSingleLayerKernel{<:Any, <:Inti.Helmholtz{3}}
        # ∇yG(x,y)⋅g : dipoles with vector strengths g (scalar output). The W operator
        # W[g] = -∫∇yG⋅g applies the leading minus when this map is assembled.
        dipvecs = Matrix{ComplexF64}(undef, 3, n)
        zk = ComplexF64(K.op.k)
        return LinearMaps.LinearMap{ComplexF64}(m, n) do y, x
            for j in 1:n
                dipvecs[:, j] = x[j] * weights[j]
            end
            if !isnothing(ndiv)
                out = FMM3D.hfmm3d(rtol, zk, sources, ndiv; dipvecs, targets, pgt = 1)
            else
                out = FMM3D.hfmm3d(rtol, zk, sources; dipvecs, targets, pgt = 1)
            end
            return copyto!(y, out.pottarg)
        end
        # Stokes
    elseif K isa Inti.SingleLayerKernel{SMatrix{3, 3, Float64, 9}, <:Inti.Stokes{3, Float64}}
        T = SVector{3, Float64}
        stoklet = Matrix{Float64}(undef, 3, n)
        return LinearMaps.LinearMap{SMatrix{3, 3, Float64, 9}}(m, n) do y, x
            # multiply by weights and constant
            stoklet[:] = 1 / K.op.μ .* reinterpret(Float64, weights .* x)
            out = FMM3D.stfmm3d(rtol, sources; stoklet, targets, ppregt = 1)
            return copyto!(y, reinterpret(T, out.pottarg))
        end
    elseif K isa Inti.DoubleLayerKernel{SMatrix{3, 3, Float64, 9}, <:Inti.Stokes{3, Float64}}
        T = SVector{3, Float64}
        normals = Matrix{Float64}(undef, 3, n)
        for j in 1:n
            normals[:, j] = Inti.normal(iop.source[j])
        end
        strsvec = similar(normals, Float64)
        strslet = similar(normals, Float64)
        # multiply by weights and constant
        for j in 1:n
            strsvec[:, j] = view(normals, :, j) .* weights[j]
        end
        return LinearMaps.LinearMap{SMatrix{3, 3, Float64, 9}}(m, n) do y, x
            strslet[:] = reinterpret(Float64, x)
            out = FMM3D.stfmm3d(rtol, sources; strslet, strsvec, targets, ppregt = 1)
            return copyto!(y, reinterpret(T, out.pottarg))
        end
    else
        error("integral operator not supported by Inti's FMM3D wrapper")
    end
end

# Charge→Hessian realization of the Hessian single-layer *volume* operator used by the
# `X = ∇W` VDIM correction: a scalar density `ρ` maps to `∫∇ₓ∇ₓG(x,y)ρ(y)dy` (an
# `SMatrix` per target), i.e. the Hessian of the single-layer potential of charges `ρ`.
# This is distinct from the forward `X` map above (a dipole→gradient contraction of the
# Hessian with a *vector* density, `SVector→SVector`), which is why it gets its own
# assembly entry point rather than dispatching on `HessianSingleLayerKernel`.
function Inti._assemble_fmm3d_chargehessian(iop::Inti.IntegralOperator; rtol = sqrt(eps()))
    m, n = size(iop)
    targets = Matrix{Float64}(undef, 3, m)
    for i in 1:m
        targets[:, i] = Inti.coords(iop.target[i])
    end
    sources = Matrix{Float64}(undef, 3, n)
    for j in 1:n
        sources[:, j] = Inti.coords(iop.source[j])
    end
    weights = [q.weight for q in iop.source]
    K = iop.kernel
    if K isa Inti.HessianSingleLayerKernel{<:Any, <:Inti.Laplace{3}}
        charges = Vector{Float64}(undef, n)
        return LinearMaps.LinearMap{SMatrix{3, 3, Float64, 9}}(m, n) do y, x
            @. charges = weights * x
            out = FMM3D.lfmm3d(rtol, sources; charges, targets, pgt = 3)
            # `hesstarg` is (6, m): the unique second derivatives in the order
            # ∂xx, ∂yy, ∂zz, ∂xy, ∂xz, ∂yz. Assemble the symmetric `SMatrix`.
            H = out.hesstarg
            @inbounds for i in 1:m
                y[i] = SMatrix{3, 3, Float64, 9}(
                    H[1, i], H[4, i], H[5, i],
                    H[4, i], H[2, i], H[6, i],
                    H[5, i], H[6, i], H[3, i],
                )
            end
            return y
        end
    else
        error("Inti's FMM3D charge→Hessian wrapper only supports Laplace 3D")
    end
end

end # module
