α, β = 0, 1
T = Inti.default_density_eltype(pde)
c = rand(T)
u = (qnode) -> Inti.SingleLayerKernel(pde)(qnode, xs) * c
# u = qnode -> 1
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

        quad = Inti.Quadrature(Γ_msh; qorder)
        ubnd = map(u, quad)
        utst = map(u, tset)
        utst_norm = norm(utst, Inf)
        # single and double layer
        G = Inti.SingleLayerKernel(pde)
        S = Inti.IntegralOperator(G, quad)
        Smat  = Inti.assemble_matrix(S)
        Stest = Inti.IntegralOperator(G, tset, quad)

        dG = Inti.DoubleLayerKernel(pde)
        D = Inti.IntegralOperator(dG, quad)
        Dmat  = Inti.assemble_matrix(D)
        Dtest = Inti.IntegralOperator(dG, tset, quad)

        μ = t == :interior ? -0.5 : 0.5
        σ    = (α * Smat + β * (Dmat + μ*I)) \ ubnd
        # @show norm(α * Smat * σ - ubnd, Inf)
        # @show norm(ubnd, Inf)
        # @show norm(Stest, Inf)
        usol = (α * Stest + β * Dtest) * σ
        # @show  utst
        e0   = norm(usol - utst, Inf) / utst_norm

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
            qorder_aux = 10 * ceil(Int, abs(log(h))),
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
        usol = (α * Stest + β * Dtest) * σ
        eloc   = norm(usol - utst, Inf) / utst_norm

        tdim = @elapsed δS, δD =
            Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
        Sdim = Smat + δS
        Ddim = Dmat + δD
        σ    = (α * Sdim + β * (Ddim + μ*I)) \ ubnd
        usol = (α * Stest + β * Dtest) * σ
        eglo   = norm(usol - utst, Inf) / utst_norm
        # @show norm(e0, Inf)
        @show e0
        @show eloc
        @show eglo
        @show tldim
        @show tdim
        push!(err0, e0)
        push!(errl, eloc)
        push!(errg, eglo)
    end
end