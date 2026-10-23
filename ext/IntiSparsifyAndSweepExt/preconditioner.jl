# The factored hybrid preconditioner, restricted to the Ω block.

"""
    HybridPreconditioner(S, to_grid, from_grid)

The factored hybrid preconditioner of the hybrid Lippmann-Schwinger solver,

    P_Ω = I + T_C^Q (S^(C) - I) T_Q^C ,

acting on the `N_Q` unstructured unknowns.  `S^(C)` is the Cartesian sparsifying
preconditioner — a [`SweepPreconditioner`](@ref), or anything callable as
`apply(S, v)` on a Cartesian vector of length `n^D`.  `to_grid` is `T_Q^C`
(nodes → Cartesian) and `from_grid` is `T_C^Q` (Cartesian → nodes); either may be
a sparse matrix or a callable.

Can be used with `IterativeSolvers.gmres` via `Pl = P` or directly as `P \\ b`.
"""
struct HybridPreconditioner{S,T1,T2}
    sparsifying::S
    to_grid::T1        # T_Q^C
    from_grid::T2      # T_C^Q
    ngrid::Int
    nnodes::Int
end

function HybridPreconditioner(S, to_grid, from_grid;
                              ngrid::Integer = -1, nnodes::Integer = -1)
    ng = ngrid < 0 ? size(to_grid, 1) : Int(ngrid)
    nq = nnodes < 0 ? size(to_grid, 2) : Int(nnodes)
    return HybridPreconditioner{typeof(S),typeof(to_grid),typeof(from_grid)}(
        S, to_grid, from_grid, ng, nq)
end

Base.size(P::HybridPreconditioner) = (P.nnodes, P.nnodes)
Base.eltype(::HybridPreconditioner) = ComplexF64

"Apply the Cartesian sparsifying preconditioner, whatever form it takes."
_apply_S(S::SweepPreconditioner, v) = apply_M(S, v)
_apply_S(S, v) = S(v)

_apply_T(A::AbstractMatrix, v) = A * v
_apply_T(f, v) = f(v)

"""
    apply(P, b)

`P b = b + T_C^Q (S^(C) - I) T_Q^C b`.
"""
function apply(P::HybridPreconditioner, b::AbstractVector)
    d = _apply_T(P.to_grid, ComplexF64.(b))          # T_Q^C b, on the Cartesian grid
    w = _apply_S(P.sparsifying, d)                   # S d
    @. w -= d                                        # (S − I) d
    return _apply_T(P.from_grid, w) .+ b             # back to the nodes, plus b
end

Base.:\(P::HybridPreconditioner, b::AbstractVector) = apply(P, b)

function LinearAlgebra.ldiv!(P::HybridPreconditioner, b::AbstractVector)
    b .= apply(P, b)
    return b
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::HybridPreconditioner, b::AbstractVector)
    y .= apply(P, b)
    return y
end

"""
    roundtrip_error(to_grid, from_grid, v)

`‖T_C^Q T_Q^C v - v‖/‖v‖`.  With the renormalised transpose this vanishes for
constant `v` by construction, so feeding it `ones` checks that the two transfer
operators are consistent.
"""
function roundtrip_error(to_grid, from_grid, v::AbstractVector)
    w = _apply_T(from_grid, _apply_T(to_grid, ComplexF64.(v)))
    return norm(w - v) / norm(v)
end
