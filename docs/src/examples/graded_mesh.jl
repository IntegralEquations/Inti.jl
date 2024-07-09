using Inti
using StaticArrays
using Meshes
using CairoMakie
using IterativeSolvers
using HMatrices
using LinearAlgebra

TEST = true
PLOT = false # can be quite slow at large k

## Frequency
k = 10π
λ = 2π / k

## Tolerance
tol = 1e-12

## Target points to test solution
target = [5 * SVector(cos(θ), sin(θ)) for θ in 0:2π/100:2π]
vals_pointsource = Dict()
vals_planewave = Dict()

## Geometry definition
Γ = let
    P = 3
    χ = Inti.kress_change_of_variables_periodic(P)
    # χ = identity
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

ppw = [1, 2, 4, 8, 16, 32, 64, 128, 256]

## Loop
for n in ppw
    println("Computing with $n points-per-wavelength")
    meshsize = λ / n
    ## Create the geometry as an equilateral triangle

    msh = Inti.meshgen(Γ; meshsize)
    Q = Inti.Quadrature(msh; qorder = 2)

    # Make sure all quadrature nodes are different and weights nonzero (floating
    # point paranoia)
    qcoords = [q.coords for q in Q]
    @assert length(unique!(qcoords)) == length(Q)
    @assert all(q -> q.weight > 0, Q)

    println("|--- $(length(Q)) dofs")

    ## Create integral operators
    pde = Inti.Helmholtz(; dim = 2, k)
    S, D = Inti.single_double_layer(;
        pde,
        target = Q,
        source = Q,
        compression = (method = :hmatrix, tol),
        # compression = (method = :none,),
        # correction = (
        #     method = :adaptive,
        #     tol = tol,
        #     maxdist = 5 * meshsize,
        #     maxsplit = 100_000,
        # ),
        correction = (method = :dim,),
    )
    L = I / 2 + D - im * k * S

    𝒮, 𝒟 = Inti.single_double_layer_potential(; pde, source = Q)

    ## Test with a source inside for validation
    if TEST
        G = Inti.SingleLayerKernel(pde)
        uₑ = x -> G(x, SVector(0.4, 0.4))
        rhs = map(q -> uₑ(q.coords), Q)
        σ, hist = gmres(L, rhs; restart = 1000, maxiter = 1000, abstol = tol, log = true)
        println("|--- converged in $(hist.iters)")
        uₕ = x -> 𝒟[σ](x) - im * k * 𝒮[σ](x)
        vals_pointsource[n] = map(uₕ, target)
    end

    ## Test with a planewave
    θ = 0 * π / 4
    uᵢ = x -> exp(im * k * dot(x, SVector(cos(θ), sin(θ))))
    rhs = map(q -> -uᵢ(q.coords), Q)
    σ, hist = gmres(L, rhs; restart = 1000, maxiter = 1000, abstol = tol, log = true)
    println("|--- converged in $(hist.iters)")
    uₕ = x -> 𝒟[σ](x) - im * k * 𝒮[σ](x)
    vals_planewave[n] = map(uₕ, target)
    ## Visualize a solution
    if PLOT && n > 8
        PLOT = false
        xx = -1:λ/10:2
        yy = -1:λ/10:2
        target = [SVector(x, y) for x in xx, y in yy]

        Spot, Dpot = Inti.single_double_layer(;
            pde,
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
end

##
nmax           = ppw[end]
er_planewave   = Float64[]
er_pointsource = Float64[]

for n in ppw
    n == nmax && continue
    push!(er_planewave, norm(vals_planewave[n] - vals_planewave[nmax], Inf))
    push!(er_pointsource, norm(vals_pointsource[n] - vals_pointsource[nmax], Inf))
end

fig = Figure()
ax1 = Axis(
    fig[1, 1];
    yscale = log10,
    xscale = log2,
    xlabel = "points-per-wavelength",
    ylabel = "|u - uref|∞",
)
scatterlines!(ax1, ppw[1:end-1], er_pointsource; label = "error pointsource")
refslope = 3
refvals = (1 ./ ppw[1:end-1] .^ refslope) * ppw[end-1]^refslope * er_pointsource[end]
scatterlines!(ax1, ppw[1:end-1], refvals; label = "slope $refslope")
axislegend()

ax2 = Axis(
    fig[1, 2];
    yscale = log10,
    xscale = log2,
    xlabel = "points-per-wavelength",
    ylabel = "|u - uref|∞",
)
scatterlines!(ax2, ppw[1:end-1], er_planewave; label = "error planewave")
refslope = 3
refvals = (1 ./ ppw[1:end-1] .^ refslope) * ppw[end-1]^refslope * er_planewave[end]
scatterlines!(ax2, ppw[1:end-1], refvals; label = "slope $refslope")
axislegend()

fig
