σ = t == :interior ? 1 / 2 : -1 / 2
T = Inti.default_density_eltype(pde)
c = rand(T)
u = (qnode) -> Inti.SingleLayerKernel(pde)(qnode, xs) * c
dudn = (qnode) -> Inti.AdjointDoubleLayerKernel(pde)(qnode, xs) * c

if TEST_TYPE == "QORDER"
    for qorder in Q
        errl = Errl[qorder]
        errg = Errg[qorder]
        for h in H
            msh = Inti.meshgen(Γ; meshsize = h)
            Γ_msh = msh[Γ]
            nel = sum(Inti.element_types(Γ_msh)) do E
                return length(Inti.elements(Γ_msh, E))
            end
            @info h, k, nel
            ##

            quad = Inti.Quadrature(Γ_msh; qorder)
            γ₀u = map(u, quad)
            γ₁u = map(dudn, quad)
            γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
            γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
            # single and double layer
            G = Inti.SingleLayerKernel(pde)
            S = Inti.IntegralOperator(G, quad)
            Smat = Inti.assemble_matrix(S)
            dG = Inti.DoubleLayerKernel(pde)
            D = Inti.IntegralOperator(dG, quad)
            Dmat = Inti.assemble_matrix(D)
            e0 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm

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
            eloc = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm

            tdim = @elapsed δS, δD =
                Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
            Sdim = Smat + δS
            Ddim = Dmat + δD
            eglo = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
            # @show norm(e0, Inf)
            @show eloc
            @show eglo
            @show tldim
            @show tdim
            push!(errl, eloc)
            push!(errg, eglo)
        end
    end
elseif TEST_TYPE == "K"
    for h in H
        msh = Inti.meshgen(Γ; meshsize = h)
        Γ_msh = msh[Γ]
        nel = sum(Inti.element_types(Γ_msh)) do E
            return length(Inti.elements(Γ_msh, E))
        end
        @info h, qorder, nel
        ##

        quad = Inti.Quadrature(Γ_msh; qorder)
        γ₀u = map(u, quad)
        γ₁u = map(dudn, quad)
        γ₀u_norm = norm(norm.(γ₀u, Inf), Inf)
        γ₁u_norm = norm(norm.(γ₁u, Inf), Inf)
        # single and double layer
        G = Inti.SingleLayerKernel(pde)
        S = Inti.IntegralOperator(G, quad)
        Smat = Inti.assemble_matrix(S)
        dG = Inti.DoubleLayerKernel(pde)
        D = Inti.IntegralOperator(dG, quad)
        Dmat = Inti.assemble_matrix(D)
        e0 = norm(Smat * γ₁u - Dmat * γ₀u - σ * γ₀u, Inf) / γ₀u_norm

        green_multiplier = fill(-0.5, length(quad))
        # δS, δD = Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)

        # qnodes = Inti.local_bdim_correction(pde, quad, quad; green_multiplier)
        # X = [q.coords[1] for q in qnodes]; Y = [q.coords[2] for q in qnodes]
        # u = [q.normal[1] for q in qnodes]; v = [q.normal[2] for q in qnodes]
        # fig, _, _ = scatter(X, Y)
        # arrows!(X, Y, u, v, lengthscale=0.01)
        # display(fig)

        for k in K
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
            eloc = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
            @show eloc, tldim
            push!(Errl[k], eloc)
        end
            
        tdim = @elapsed δS, δD =
            Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
        Sdim = Smat + δS
        Ddim = Dmat + δD
        eglo = norm(Sdim * γ₁u - Ddim * γ₀u - σ * γ₀u, Inf) / γ₀u_norm
        # @show norm(e0, Inf)
        @show eglo, tdim
        push!(Errg, eglo)
    end
end
    