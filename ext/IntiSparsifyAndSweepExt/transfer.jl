# Transfer operators between the Cartesian grid and an unstructured node set.
#
#   T_C^Q : Cartesian → nodes,  multilinear interpolation      (rows sum to 1)
#   T_Q^C : nodes → Cartesian,  column-renormalised transpose  (preserves constants)
#
# Both are dimension-generic: the 2^D-point stencil is bilinear in 2D and
# trilinear in 3D.

"""
    interpolation_matrix(pg, nodes)

`T_C^Q`: the sparse `N_Q x n^D` matrix of multilinear interpolation from the
Cartesian nodes of `pg` to the physical points `nodes`, with `2^D` entries per
row summing to one.  `nodes` is anything [`node_coordinates`](@ref) accepts — a
vector of coordinate tuples, or whatever a mesh library teaches it to unpack.

Points are clamped into the interpolation stencil, so a node outside the
Cartesian nodes' convex hull is extrapolated rather than erroring.  That should
not happen for a properly padded [`PhysicalGrid`](@ref); [`interpolation_reach`](@ref)
reports how close the node set gets to the edge.
"""
function interpolation_matrix(pg::PhysicalGrid{D}, nodes_in) where {D}
    nodes = node_coordinates(nodes_in)
    pts = nodes isa AbstractVector ? nodes : collect(nodes)
    g = pg.g
    n, h = g.n, g.h
    Nq = length(pts)
    P = 1 << D                      # 2^D candidate entries per node
    Ir = Vector{Int}(undef, Nq * P)
    Jc = Vector{Int}(undef, Nq * P)
    Vv = Vector{Float64}(undef, Nq * P)
    corners = CartesianIndices(ntuple(_ -> 0:1, D))
    Threads.@threads for k in 1:Nq
        ξ = unitcoords(pg, pts[k])
        base = ntuple(d -> clamp(floor(Int, ξ[d] / h), 1, n - 1), D)
        t = ntuple(d -> (ξ[d] - base[d] * h) / h, D)
        s = (k - 1) * P             # this node's private block
        for (p, c) in enumerate(corners)
            o = Tuple(c)
            w = 1.0
            for d in 1:D
                w *= o[d] == 1 ? t[d] : (1 - t[d])
            end
            # `base .+ o` is in `1:n` by construction, so the index is always
            # valid even when the weight vanishes; zeros are dropped below
            Ir[s + p] = k
            Jc[s + p] = plin(g, base .+ o)
            Vv[s + p] = w
        end
    end
    # a node sitting exactly on a grid line gives zero weights; drop them so the
    # sparsity pattern is unchanged relative to a simple serial implementation
    keep = findall(!iszero, Vv)
    return sparse(Ir[keep], Jc[keep], Vv[keep], Nq, nphys(g))
end

"""
    renormalized_transpose(W)

`T_Q^C` from `W = T_C^Q`: transpose with every row rescaled by the corresponding
column sum of `W`, i.e. `diag(W'I)⁻¹ W'`, with rows having no contributing nodes
set zero. This is Approach II of the hybrid solver.
"""
function renormalized_transpose(W::SparseMatrixCSC)
    s = vec(sum(W; dims = 1))
    Wt = sparse(transpose(W))
    rows = rowvals(Wt)
    vals = nonzeros(Wt)
    for j in 1:size(Wt, 2), p in nzrange(Wt, j)
        r = rows[p]
        vals[p] = s[r] > 0 ? vals[p] / s[r] : 0.0
    end
    return Wt
end

"""
    interpolation_reach(pg, nodes)

`(minimum, maximum)` unit-cube coordinate attained by `nodes`, over all axes.
Both should sit comfortably inside `(h, 1-h)`; if not, the [`PhysicalGrid`](@ref)
padding is too small and the interpolation is extrapolating at the edges.
"""
function interpolation_reach(pg::PhysicalGrid{D}, nodes_in) where {D}
    nodes = node_coordinates(nodes_in)
    lo, hi = Inf, -Inf
    for x in nodes
        ξ = unitcoords(pg, x)
        for d in 1:D
            lo = min(lo, ξ[d])
            hi = max(hi, ξ[d])
        end
    end
    return (lo, hi)
end

"""
    covered_nodes(W)

Indices of the Cartesian nodes that at least one unstructured node interpolates
from, i.e. the rows of `T_Q^C` that are not identically zero.  Everything
outside this set is invisible to the coupling.
"""
covered_nodes(W::SparseMatrixCSC) = findall(>(0), vec(sum(W; dims = 1)))
