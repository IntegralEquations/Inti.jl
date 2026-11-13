using Inti
using StaticArrays
using Meshes
using GLMakie
using IterativeSolvers
using FMMLIB2D
using LinearAlgebra

## Parameters
k = 10π
λ = 2π / k
meshsize = λ / 10
TEST = false
PLOT = true # can be quite slow at large k

## Create the geometry as an equilateral
Γ = let
    P = 2
    χ = Inti.kress_change_of_variables_periodic(P)
    p1 = SVector(0.0, 0.0)
    p2 = SVector(1.0, 0.0)
    p3 = SVector(0.5, sqrt(3) / 2)

    l1 = Inti.parametric_curve(
        t -> p1 + χ(t) * (p2 - p1) + 0.0 * SVector(0, sin(2π * t)),
        0.0,
        1.0,
    )
    l2 = Inti.parametric_curve(t -> p2 + χ(t) * (p3 - p2), 0.0, 1.0)
    l3 = Inti.parametric_curve(t -> p3 + χ(t) * (p1 - p3), 0.0, 1.0)
    Inti.Domain(l1, l2, l3)
end

msh = Inti.meshgen(Γ; meshsize)
Q = Inti.Quadrature(msh; qorder = 3)

##

## Create integral operators
op = Inti.Helmholtz(; dim = 2, k)
S, D = Inti.single_double_layer(;
    op,
    target = Q,
    source = Q,
    compression = (method = :fmm, tol = 1e-6),
    correction = (method = :dim,),
)
L = I / 2 + D - im * k * S

𝒮, 𝒟 = Inti.single_double_layer_potential(; op, source = Q)

## Test with a source inside for validation
if TEST
    G = Inti.SingleLayerKernel(op)
    uₑ = x -> G(x, SVector(0.4, 0.4))
    rhs = map(q -> uₑ(q.coords), Q)
    σ = gmres(L, rhs; restart = 100, maxiter = 100, abstol = 1e-8, verbose = true)
    uₕ = x -> 𝒟[σ](x) - im * k * 𝒮[σ](x)
    xₜ = SVector(2, 0)
    @show uₕ(xₜ), uₑ(xₜ)
end

## Test with a planewave
θ = 0 * π / 4
uᵢ = x -> exp(im * k * dot(x, SVector(cos(θ), sin(θ))))
rhs = map(q -> -uᵢ(q.coords), Q)

σ = gmres(L, rhs; restart = 1000, maxiter = 1000, abstol = 1e-8, verbose = true)
uₕ = x -> 𝒟[σ](x) - im * k * 𝒮[σ](x)
@info "Done solving..."

## Visualize a solution
if PLOT
    xx = -1:λ/10:2
    yy = -1:λ/10:2
    target = [SVector(x, y) for x in xx, y in yy]

    Spot, Dpot = Inti.single_double_layer(;
        op,
        target = target,
        source = Q,
        compression = (method = :fmm, tol = 1e-4),
        correction = (method = :none, target_location = :outside, maxdist = meshsize),
    )

    vals = Dpot * σ - im * k * Spot * σ
    vals = map(
        i -> Inti.isinside(target[i], Q) ? NaN + im * NaN : vals[i] + uᵢ(target[i]),
        1:length(target),
    )
    vals = reshape(vals, length(xx), length(yy))

    heatmap(
        xx,
        yy,
        real.(vals);
        colorrange = (-1, 1),
        colormap = :inferno,
        axis = (aspect = DataAspect(),),
    )
    viz!(msh[Γ]; showsegments = true)
    Makie.current_figure()
end
