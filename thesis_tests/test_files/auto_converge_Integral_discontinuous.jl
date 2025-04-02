# ε = 1
# b = x -> -ε < x < ε ? exp(-1/(1 - (x / ε)^2)) / ε : 0
# u = x -> 0 ≤ x % (2π) < π || -2π ≤ x % (2π) < -π ? 1 : 0
# bu = x -> quadgk(y -> b(y) * u(x - y), -ε, ε, atol=1e-15)[1]
# bu = x -> sin(x)
# u = x -> x[2] > 0 ? 1 : 0
# u = x -> x[2] > 0 ? SVector(0, 1) : SVector(0, -1)
# u = x -> x[2]
# u = x -> x[2] / norm(x)
# u = x -> 1
u = x -> cos(x.coords[1]) * exp(x.coords[2])
# u = x ->
# u = x -> cos(x[1])

qorder_ref, h_ref = 5, 0.5*1e-4
msh = Inti.meshgen(Γ; meshsize = h_ref)
Γ_msh = msh[Γ]
quad_ref = Inti.Quadrature(Γ_msh; qorder=qorder_ref)
uref = map(u, quad_ref)

for qorder in Q
    errl = Errl[qorder]
    errg = Errg[qorder]
    for h in H
        # k = ceil(Int, 0.1 / h)
        
        msh = Inti.meshgen(Γ; meshsize = h)
        Γ_msh = msh[Γ]
        nel = sum(Inti.element_types(Γ_msh)) do E
            return length(Inti.elements(Γ_msh, E))
        end
        @info h, k, nel
        ##

        quad = Inti.Quadrature(Γ_msh; qorder)
        qs = argmin(quad) do q
            norm(q.coords-xt)
        end
        xs = qs.coords
        @info xs

        uvec = map(u, quad)
        # uvec = [bu(atan(x[2], x[1])) for x in Inti.coords.(quad)]
        uvec_norm = norm(uvec, Inf)
        # single and double layer
        

        green_multiplier = [-0.5]
        # δS, δD = Inti.bdim_correction(pde, quad, quad, Smat, Dmat; green_multiplier)

        # qnodes = Inti.local_bdim_correction(pde, quad, quad; green_multiplier)
        # X = [q.coords[1] for q in qnodes]; Y = [q.coords[2] for q in qnodes]
        # u = [q.normal[1] for q in qnodes]; v = [q.normal[2] for q in qnodes]
        # fig, _, _ = scatter(X, Y)
        # arrows!(X, Y, u, v, lengthscale=0.01)
        # display(fig)
        S = Inti.IntegralOperator(G, [xs], quad_ref)
        Smat = Inti.assemble_matrix(S)
        D = Inti.IntegralOperator(dG, [xs], quad_ref)
        Dmat = Inti.assemble_matrix(D)

        δS, δD = Inti.bdim_correction(pde, [xs], quad_ref, Smat, Dmat; green_multiplier)
        Sref = Smat + δS
        Dref = Dmat + δD
        ref = ((α*Sref + β*Dref) * uref)[1]
        @show ref

        S = Inti.IntegralOperator(G, [xs], quad)
        Smat = Inti.assemble_matrix(S)
        D = Inti.IntegralOperator(dG, [xs], quad)
        Dmat = Inti.assemble_matrix(D)

        tldim = @elapsed δS, δD = Inti.local_bdim_correction(
            pde,
            [xs],
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
        eloc = abs(((α*Sdim + β*Ddim) * uvec)[1] - ref) / abs(ref)

        tdim = @elapsed δS, δD =
            Inti.bdim_correction(pde, [xs], quad, Smat, Dmat; green_multiplier)
        Sdim = Smat + δS
        Ddim = Dmat + δD
        eglo = abs(((α*Sdim + β*Ddim) * uvec)[1] - ref) / abs(ref)
        # @show norm(e0, Inf)
        @show eloc
        @show eglo
        @show tldim
        @show tdim
        push!(errl, eloc)
        push!(errg, eglo)
    end
end

# ε = 1
# b = x -> -ε < x < ε ? exp(-1/(1 - (x / ε)^2)) / ε : 0
# u = x -> 0 ≤ x % (2π) < π || -2π ≤ x % (2π) < -π ? 1 : 0
# bu = x -> quadgk(y -> b(y) * u(x - y), -ε, ε, atol=1e-15)[1]

# fig = Figure()
# ax = Axis(fig[1, 1])
# X = LinRange(-2π-1, 2π+1, 1000)
# lines!(ax, X, u.(X))
# lines!(ax, X, b.(X))
# lines!(ax, X, bu.(X))
# display(fig)