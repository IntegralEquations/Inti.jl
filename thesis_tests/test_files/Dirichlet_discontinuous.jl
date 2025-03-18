G = Inti.SingleLayerKernel(pde)
dG = Inti.DoubleLayerKernel(pde)
α, β = 1, 1
# δ = 1e-8
# u = (qnode) -> (Inti.coords(qnode)[2] + δ) / (2δ)
σ_ref = (qnode) -> Inti.coords(qnode)[2] > 0 ? 1 : 0
u = x -> quadgk(a, b, atol=1e-15) do s
    χ_ = s -> ForwardDiff.derivative(χ, s)
    n  = s -> normalize(SVector(χ_(s)[2], -χ_(s)[1]))
    y  = s -> (coords=χ(s), normal=n(s))
    (α * G(x, χ(s)) * σ_ref(χ(s)) + β * dG(x, y(s)) * σ_ref(χ(s))) * norm(χ_(s))        
end[1]
# u = qnode -> 1
# qorder = 3
# h = 0.1
# msh = Inti.meshgen(Γ; meshsize = h)
# Γ_msh = msh[Γ]
# quad = Inti.Quadrature(Γ_msh; qorder)
# ubnd = map(u, quad) + μ * map(σ_ref, quad)
# (α * Sdim + β * (Ddim + μ*I)) \ ubnd
# σ_vec = map(σ_ref, quad)

for qorder in Q
    err0 = Err0[qorder]
    errl = Errl[qorder]
    errg = Errg[qorder]
    for h in H
        # k = ceil(Int, 0.1 / h)

        # Ω, msh = gmsh_disk(; center = [0.0, 0.0], rx = 1.0, ry = 1.0, meshsize = h, order = 2)
        # Γ = Inti.external_boundary(Ω)

        msh = Inti.meshgen(Γ; meshsize = h)
        Γ_msh = msh[Γ]
        nel = sum(Inti.element_types(Γ_msh)) do E
            return length(Inti.elements(Γ_msh, E))
        end
        @info h, k, nel
        ##

        μ = t == :interior ? -0.5 : 0.5
        quad = Inti.Quadrature(Γ_msh; qorder)
        ubnd = map(u, quad) + μ * map(σ_ref, quad)
        utst = map(u, tset)
        utst_norm = norm(utst, Inf)
        # single and double layer
        S = Inti.IntegralOperator(G, quad)
        Smat  = Inti.assemble_matrix(S)
        Stest = Inti.IntegralOperator(G, tset, quad)

        D = Inti.IntegralOperator(dG, quad)
        Dmat  = Inti.assemble_matrix(D)
        Dtest = Inti.IntegralOperator(dG, tset, quad)

        green_multiplier = fill(-0.5, length(quad))
        # δS, δD = Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)

        # qnodes = Inti.local_bdim_correction(pde, quad, quad; green_multiplier)
        # X = [q.coords[1] for q in qnodes]; Y = [q.coords[2] for q in qnodes]
        # u = [q.normal[1] for q in qnodes]; v = [q.normal[2] for q in qnodes]
        # fig, _, _ = scatter(X, Y)
        # arrows!(X, Y, u, v, lengthscale=0.01)
        # display(fig)

        tldim = @elapsed δS, δD = Inti.local_bdim_correction(
            pde,
            quad,
            quad;
            green_multiplier,
            kneighbor = k,
            maxdist = 10 * h,
            qorder_aux = 20 * ceil(Int, abs(log(h))),
        )
        Sdim = Smat + δS
        Ddim = Dmat + δD
        # Sdim, Ddim = Inti.single_double_layer(;
        #     pde,
        #     target      = quad,
        #     source      = quad,
        #     compression = (method = :none,),
        #     correction  = (method = :ldim,),
        # )
        σ    = (α * Sdim + β * (Ddim + μ*I)) \ ubnd
        # @show σ
        usol = (α * Stest + β * Dtest) * σ
        eloc   = norm(usol - utst, Inf) / utst_norm

        tdim = @elapsed δS, δD =
            Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
        Sdim = Smat + δS
        Ddim = Dmat + δD
        σ    = (α * Sdim + β * (Ddim + μ*I)) \ ubnd
        # @show σ
        usol = (α * Stest + β * Dtest) * σ
        eglo   = norm(usol - utst, Inf) / utst_norm
        # @show norm(e0, Inf)
        @show eloc
        @show eglo
        @show tldim
        @show tdim
        push!(errl, eloc)
        push!(errg, eglo)
    end
end