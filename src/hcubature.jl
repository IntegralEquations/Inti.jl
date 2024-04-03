"""
    hcubature(f, τ̂::ReferenceShape, x̂ₛ = nothing; kwargs...)

Integrate `f` over `τ̂` using `hcubature` where `x̂ₛ ∈ τ̂` denotes the location
of an isolated singularity of `f`.
"""
function HCubature.hcubature(
    f,
    τ̂::ReferenceHyperCube{N},
    x̂ₛ = nothing;
    kwargs...,
) where {N}
    lb, ub = svector(i -> 0.0, N), svector(i -> 1.0, N)
    if x̂ₛ === nothing
        return hcubature(f, lb, ub; kwargs...)
    else
        # manually differentiate lines, squares, and cubes
        if N == 1
            τ₁, τ₂ = decompose(τ̂, x̂ₛ)
            l₁, l₂ = norm(lb - x̂ₛ), norm(ub - x̂ₛ)
            return hcubature(lb, ub; kwargs...) do x
                # do an xᵖ transformation to accumulate points at 0. Taking p
                # too large can lead to rounding errors since e.g. τ₁(x^p) = a +
                # xᵖ*(b-a) can become indistinguishable from to τ₁(0) = 0.
                # Perhaps this can be fixed?
                p = 2; x̃ = x.^p; μ = p*x[1]^(p-1)
                return (f(τ₁(x̃)) * l₁ + f(τ₂(x̃)) * l₂) * μ
            end
        elseif N == 2
            duffy  = (u) -> SVector(u[1], (1 - u[1]) * u[2])
            duffy′ = (u) -> 1 - u[1] # determinant of jacobian
            τ₁, τ₂, τ₃, τ₄ = decompose(τ̂, x̂ₛ)
            a₁, a₂, a₃, a₄ = map(x -> integration_measure(x, lb), (τ₁, τ₂, τ₃, τ₄))
            return hcubature(lb, ub; kwargs...) do x
                x̃ = duffy(x)
                μ = duffy′(x)
                return (a₁ * f(τ₁(x̃)) + a₂ * f(τ₂(x̃)) + a₃ * f(τ₃(x̃)) + a₄ * f(τ₄(x̃))) * μ
            end
        else
            error("not implemented")
        end
    end
end

function HCubature.hcubature(f, τ̂::ReferenceTriangle, x̂ₛ = nothing; kwargs...)
    lb, ub = SVector(0.0, 0.0), SVector(1.0, 1.0) # lower and upper bounds integration square
    duffy  = (u) -> SVector(u[1], (1 - u[1]) * u[2])
    duffy′ = (u) -> 1 - u[1] # determinant of jacobian
    if x̂ₛ === nothing
        return hcubature(lb, ub; kwargs...) do x
            x̃ = duffy(x)
            μ = duffy′(x)
            return f(x̃) * μ
        end
    else
        τ₁, τ₂, τ₃ = decompose(τ̂, x̂ₛ)
        # compute the area of each triangle
        a₁, a₂, a₃ = map(x -> integration_measure(x, lb), (τ₁, τ₂, τ₃))
        return hcubature(lb, ub; kwargs...) do x
            x̃ = duffy(x)
            μ = duffy′(x)
            return (a₁ * f(τ₁(x̃)) + a₂ * f(τ₂(x̃)) + a₃ * f(τ₃(x̃))) * μ
        end
    end
end
