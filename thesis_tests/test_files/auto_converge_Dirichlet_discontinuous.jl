T = Inti.default_density_eltype(pde)
G = Inti.SingleLayerKernel(pde)
dG = Inti.DoubleLayerKernel(pde)
# u = x -> SVector(cos(Inti.coords(x)[1]), Inti.coords(x)[2]*sin(Inti.coords(x)[1]))
σ = x -> cos(Inti.coords(x)[1]) * exp(Inti.coords(x)[2])
# σ = x -> SVector(-sin(x.coords[1])*exp(x.coords[2]), cos(x.coords[1])*exp(x.coords[2])) ⋅ x.normal

qorder_ref, h_ref = 5, 1e-4
msh = Inti.meshgen(Γ; meshsize = h_ref)
Γ_msh = msh[Γ]
quad_ref = Inti.Quadrature(Γ_msh; qorder = qorder_ref)
σ_ref = map(σ, quad_ref)
σref_norm = norm(σ_ref, Inf)

if !onSurf
    Stest = Inti.IntegralOperator(G, tset, quad_ref)
    Dtest = Inti.IntegralOperator(dG, tset, quad_ref)
    utst = (α * Stest + β * Dtest) * σ_ref
    utst_norm = norm(utst, Inf)
end

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
        # reference values
        quad = Inti.Quadrature(Γ_msh; qorder)
        green_multiplier = fill(-0.5, length(quad))
        S = Inti.IntegralOperator(G, quad, quad_ref)
        Sref  = Inti.assemble_matrix(S)
        D = Inti.IntegralOperator(dG, quad, quad_ref)
        Dref  = Inti.assemble_matrix(D)
        δS, δD = Inti.bdim_correction(pde, quad, quad_ref, Sref, Dref; green_multiplier)
        Sdim = Sref + δS
        Ddim = Dref + δD
        σ_vec = map(σ, quad)
        ubnd = (α*Sdim + β*Ddim) * σ_ref + β*μ*σ_vec

        # for calculating αS[σ] + βD[σ] on test set
        Stest = Inti.IntegralOperator(G, tset, quad)
        Dtest = Inti.IntegralOperator(dG, tset, quad)

        # solve for numeric σ
        S = Inti.IntegralOperator(G, quad)
        Smat  = Inti.assemble_matrix(S)

        D = Inti.IntegralOperator(dG, quad)
        Dmat  = Inti.assemble_matrix(D)

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
        σl   = (α*Sdim + β*(Ddim + μ*I)) \ ubnd
        if onSurf
            eloc = norm(σl - σ_vec, Inf) / σref_norm
        else
            usol = (α * Stest + β * Dtest) * σl
            eloc   = norm(usol - utst, Inf) / utst_norm
        end
        # lm =  FunctionMap{Float64}(N*length(quad)) do x
        #     xs = reinterpret(T, x)
        #     ys = (α * Sdim + β * (Ddim + μ*I)) * xs
        #     reinterpret(Float64, ys)
        # end
        # σ = gmres(lm, reinterpret(Float64, ubnd))
        # usol = (α * Stest + β * Dtest) * reinterpret(T, σ)

        tdim = @elapsed δS, δD =
            Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)
        Sdim = Smat + δS
        Ddim = Dmat + δD
        σg   = (α*Sdim + β*(Ddim + μ*I)) \ ubnd
        if onSurf
            eglo = norm(σg - σ_vec, Inf) / σref_norm
        else
            usol = (α * Stest + β * Dtest) * σg
            eglo   = norm(usol - utst, Inf) / utst_norm
        end
        # lm =  FunctionMap{Float64}(N*length(quad)) do x
        #     xs = reinterpret(T, x)
        #     ys = (α * Sdim + β * (Ddim + μ*I)) * xs
        #     reinterpret(Float64, ys)
        # end
        # σ = gmres(lm, reinterpret(Float64, ubnd))
        # usol = (α * Stest + β * Dtest) * reinterpret(T, σ)
        # @show norm(e0, Inf)
        @show eloc
        @show eglo
        @show tldim
        @show tdim
        push!(errl, eloc)
        push!(errg, eglo)
    end
end