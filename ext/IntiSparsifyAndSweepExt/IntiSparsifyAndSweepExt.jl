"""
    IntiSparsifyAndSweepExt

Package extension loaded when both `Inti` and
[`SparsifyAndSweep`](https://github.com/tganderson/SparsifyAndSweep.jl) are
available.  It couples an unstructured Inti discretisation of the
Lippmann-Schwinger equation to `SparsifyAndSweep`'s Cartesian preconditioner
through the hybrid preconditioner

    `P = I + T_C^Q (S^(C) − I) T_Q^C`

where `T_C^Q` interpolates from a structured Cartesian grid with `N_C` points to
the `N_Q` unstructured nodes, `T_Q^C` maps back, and `S^(C)` is the Cartesian
preconditioner: `mode = :direct` (Ying 2015), `mode = :sweep` (`O(N)`), or
`mode = :sweep` with `levels = 2` (the recursive sweep).

`SparsifyAndSweep` itself only ever sees a unit cube of Cartesian unknowns and
knows nothing about meshes; everything that has to talk to both worlds lives
here.  The code is dimension agnostic, although only 2D has been tested.

Extensions are not `using`-able, so reach the names through the module:

```julia
using Inti, SparsifyAndSweep
const SAS = Base.get_extension(Inti, :IntiSparsifyAndSweepExt)

pg  = SAS.PhysicalGrid(Val(2), (-1.0, -1.0), (1.0, 1.0), 255; b = 16, pad = 0.5)
S   = SweepPreconditioner(LSProblem(255, SAS.unit_frequency(pg, k), c; b = 16))
Tcq = SAS.interpolation_matrix(pg, Ωₕ_quad)      # takes an Inti.Quadrature directly
Tqc = SAS.renormalized_transpose(Tcq)
P   = SAS.HybridPreconditioner(S, Tqc, Tcq)      # ready for `gmres(...; Pl = P)`
```

See `docs/src/examples/lippmann_schwinger_sparsify_and_sweep.jl` for a complete
run against the exact Mie series.
"""
module IntiSparsifyAndSweepExt

import Inti

using SparsifyAndSweep
using SparsifyAndSweep: Grid, nphys, plin, offsets
using LinearAlgebra
using SparseArrays

function __init__()
    return @debug "Loading Inti.jl SparsifyAndSweep extension"
end

"""
    node_coordinates(nodes)

Physical coordinates of an unstructured node set, in the form
[`interpolation_matrix`](@ref) wants: a vector of coordinate tuples.

Any iterable of coordinates is already in that form and passes through; the
methods below let an Inti quadrature be handed to the transfer operators
directly, rather than being unpacked by hand.
"""
node_coordinates(nodes) = nodes

"""
    node_coordinates(quad::Inti.Quadrature)

The physical coordinates of an Inti volume or boundary quadrature, in the order
its nodes carry.
"""
node_coordinates(quad::Inti.Quadrature) = [q.coords for q in quad]

"Also accept a bare vector of Inti quadrature nodes."
node_coordinates(nodes::AbstractVector{<:Inti.QuadratureNode}) = [q.coords for q in nodes]

include("physical_grid.jl")
include("transfer.jl")
include("preconditioner.jl")

end # module
