using Inti
using Gmsh
using LinearAlgebra
using CairoMakie
using StaticArrays
using PolygonOps
using CurveFit

## integrate r^k log(r) on a disk of radius 1. The exact value is -pi/2 for r=0
gorder = 3
qorder = 3
gmsh.initialize()
gmsh.model.occ.addDisk(0,0,0,1,1)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
er_log = Float64[]
er_area = Float64[]
nn = Int[]
levels = 8
for _ in 1:levels
    Inti.clear_entities!()
    gmsh.model.mesh.setOrder(gorder)
    Ω, msh  = Inti.import_mesh_from_gmsh_model(;dim=2)
    Q = Inti.Quadrature(view(msh,Ω);qorder)
    I_log = Inti.integrate(Q) do q
        r = norm(q.coords)
        r^2 * log(r)
    end
    area = Inti.integrate(q->1,Q)
    EtoQnodes = Q.etype2qtags[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 10, SVector{2, Float64}}]
    elements = Q.mesh.etype2etags
    Els2Nodes = Q.mesh.parent.etype2mat[Inti.LagrangeElement{Inti.ReferenceSimplex{2}, 10, SVector{2, Float64}}]
    for ind_el in 1:size(Els2Nodes, 2)
        # Let's drop the triangle containing the origin
        verts = Els2Nodes[1:3, ind_el]
        trinodes = Q.mesh.parent.nodes[verts]
        trinodes = SVector(trinodes[1], trinodes[2], trinodes[3], trinodes[1])
        if inpolygon(SVector(0.0, 0.0), trinodes) == 1
            I_log = I_log - sum(q -> norm(q.coords)^2 * log(norm(q.coords)) * q.weight, Q.qnodes[EtoQnodes[:, ind_el]])
            area = area - sum(q -> q.weight, Q.qnodes[EtoQnodes[:, ind_el]])
        end
    end
    #push!(er_log,abs(I_log + 2π/4))
    #push!(er_log,abs(I_log + 2π/9))
    push!(er_log,abs(I_log + 2π/16))
    push!(er_area, abs(area - π))
    push!(nn,length(Q))
    gmsh.model.mesh.refine()
end
gmsh.finalize()

fig = Figure()
ax1 = Axis(fig[1,1], title="log(r)", xlabel="refinement level", ylabel="error", yscale=log10)
scatterlines!(ax1,1:levels,er_log)
ax2 = Axis(fig[2,1], title="area", xlabel="refinement level", ylabel="error", yscale=log10)
scatterlines!(ax2, 1:levels,er_area;)
fig

h = 2 * (1/2).^(1:levels)
area_conv = linear_fit(log10.(h), log10.(er_area))
log_conv = linear_fit(log10.(h), log10.(er_log))

##
