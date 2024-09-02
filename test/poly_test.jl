using Inti
using StaticArrays
using LinearAlgebra
import DynamicPolynomials: @polyvar
using FixedPolynomials

#Run with npts = 1000000
function poly_test(npts)
    #npts = 10000
    pde = Inti.Laplace(; dim = 2)
    interpolation_order = 4
    p, P, γ₁P, multiindices = Inti.polynomial_solutions_vdim(pde, interpolation_order)

    @polyvar x y

    N = 2
    PolArray = Array{Polynomial{Float64}}(undef, length(P))

    tsetup = @elapsed begin
        for (polind, ElemPolySolsPols) in enumerate(P)
            pp = ElemPolySolsPols.f.order2coeff
            exp_data = Matrix{Int64}(undef, N, length(pp))
            coeff_data = Vector{Float64}(undef, length(pp))
            for (i, pol) in enumerate(pp)
                exp_data[:, i] = [q for q in pol[1]]
                coeff_data[i] = pol[2]
            end
            PolArray[polind] = Polynomial(exp_data, coeff_data, [:x, :y])
        end
        PolSystem = System(PolArray)
        pts = Vector{Vector{Float64}}(undef, npts)
        pts[1] = [0, 0]
        cfg = JacobianConfig(PolSystem, pts[1])
    end
    @info "FixedPolynomials.jl setup time: $tsetup"

    pts = Vector{Vector{Float64}}(undef, npts)
    for i in 1:npts
        pts[i] = rand(2)
    end
    cfg = JacobianConfig(PolSystem, pts[1])
    res1 = Matrix{Float64}(undef, length(PolArray), length(pts))
    res2 = Matrix{Float64}(undef, length(PolArray), length(pts))
    res3 = Vector{MVector{length(PolArray),Float64}}(undef, npts)

    cfg = JacobianConfig(PolSystem, pts[1])
    u = Vector{Float64}(undef, length(PolArray))
    tfixed = @elapsed begin
        for i in 1:npts
            evaluate!(view(res1, :, i), PolSystem, pts[i], cfg)
        end
    end
    @info "FixedPolynomials.jl time: $tfixed"
    evaluator = (xx) -> evaluate(PolSystem, xx, cfg)
    tbroadcast = @elapsed begin
        res3 .= evaluator.(pts)
    end
    @info "FixedPolynomials.jl w/ broadcast time: $tbroadcast"
    tregular = @elapsed begin
        for i in 1:length(P)
            res2[i, :] .= P[i].f.(pts)
        end
    end
    @info "ElementaryPDESolutions.jl time: $tregular"

    # Evaluate Jacobian
    u = Vector{Float64}(undef, length(P))
    U = Matrix{Float64}(undef, length(P), 2)
    tjacob = @elapsed begin
        for i in 1:npts
            evaluate_and_jacobian!(u, U, PolSystem, pts[i], cfg)
        end
    end
    @info "FixedPolynomials.jl Jacobian+Eval time: $tjacob"
    return res1, res2, res3, tfixed, tregular, tbroadcast, tjacob
end
