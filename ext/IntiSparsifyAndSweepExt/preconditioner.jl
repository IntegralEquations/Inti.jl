# The factored hybrid preconditioner, restricted to the Ω block.

"Accumulated wall time, in seconds, of the stages of a [`HybridPreconditioner`](@ref)."
mutable struct HybridTimer
    napply::Int
    to_grid::Float64       # T_Q^C
    sparsifying::Float64   # S^(C)
    from_grid::Float64     # T_C^Q, the update (S − I) and the sum with b
end
HybridTimer() = HybridTimer(0, 0.0, 0.0, 0.0)

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

Every application is timed, split into its three stages; read the accumulated
times with [`timings`](@ref) and clear them with [`reset_timings!`](@ref).
"""
struct HybridPreconditioner{S, T1, T2}
    sparsifying::S
    to_grid::T1        # T_Q^C
    from_grid::T2      # T_C^Q
    ngrid::Int
    nnodes::Int
    timer::HybridTimer
end

function HybridPreconditioner(
        S, to_grid, from_grid;
        ngrid::Integer = -1, nnodes::Integer = -1
    )
    ng = ngrid < 0 ? size(to_grid, 1) : Int(ngrid)
    nq = nnodes < 0 ? size(to_grid, 2) : Int(nnodes)
    return HybridPreconditioner{typeof(S), typeof(to_grid), typeof(from_grid)}(
        S, to_grid, from_grid, ng, nq, HybridTimer()
    )
end

"""
    timings(P::HybridPreconditioner)

`(; napply, total, to_grid, sparsifying, from_grid)`: the number of applications
of `P` since it was built or last reset, and the wall time in seconds spent in
all of them together and in each stage.
"""
function timings(P::HybridPreconditioner)
    τ = P.timer
    return (;
        napply = τ.napply, total = τ.to_grid + τ.sparsifying + τ.from_grid,
        to_grid = τ.to_grid, sparsifying = τ.sparsifying, from_grid = τ.from_grid,
    )
end

"Zero the accumulated [`timings`](@ref) of `P`."
function reset_timings!(P::HybridPreconditioner)
    τ = P.timer
    τ.napply = 0
    τ.to_grid = τ.sparsifying = τ.from_grid = 0.0
    return P
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
    τ = P.timer
    t0 = time_ns()
    d = _apply_T(P.to_grid, ComplexF64.(b))          # T_Q^C b, on the Cartesian grid
    t1 = time_ns()
    w = _apply_S(P.sparsifying, d)                   # S d
    t2 = time_ns()
    @. w -= d                                        # (S − I) d
    y = _apply_T(P.from_grid, w) .+ b                # back to the nodes, plus b
    t3 = time_ns()
    τ.napply += 1
    τ.to_grid += (t1 - t0) / 1.0e9
    τ.sparsifying += (t2 - t1) / 1.0e9
    τ.from_grid += (t3 - t2) / 1.0e9
    return y
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

"""
    timed_gmres!(u, A, b; Pl = nothing, flexible = is_nonlinear(Pl), reltol, abstol,
                 restart, maxiter, kwargs...) -> (u, history, times)

GMRES on `A u = b` from the initial guess `u`, with its wall time broken down.

By default this is `IterativeSolvers.gmres!(u, A, b; Pl, log = true, ...)`, left
preconditioned, and `history` is its `ConvergenceHistory`.  With `flexible = true`
it is instead a right-preconditioned flexible GMRES (Saad's FGMRES), which stores
the preconditioned basis and so tolerates a preconditioner that changes from one
application to the next — the case when the Cartesian preconditioner runs inner
Krylov iterations (`SweepPreconditioner(...; inner = k)`), which is when it is
switched on by default.  It stops on the true residual `‖b − A u‖ ≤
max(reltol ‖b‖, abstol)`, and `history` is `(; iters, isconverged, resnorms)`.
Remaining `kwargs` go to `IterativeSolvers.gmres!` and are an error when flexible.

`times` is a named tuple with

* `total`: the whole solve;
* `forward`, `nforward`: the forward maps `A x` and how many were applied;
* `precond`, `nprecond`: the preconditioner applications;
* `sparsifying`, `transfer`: for a [`HybridPreconditioner`](@ref), `precond`
  split into the Cartesian preconditioner `S^(C)` and the rest — the transfers
  `T_Q^C`, `T_C^Q` and the vector updates around them — (`NaN` otherwise);
* `nsweeps`: the sweeps performed, when `S^(C)` is a `SweepPreconditioner`
  (inner iterations included; `nothing` otherwise);
* `other`: what remains, i.e. GMRES's own work: Arnoldi orthogonalisation, the
  Givens rotations, the solution updates.

`A` is wrapped in a timed `LinearMap`, so it must support `mul!(y, A, x)`; the
accumulated [`timings`](@ref) of a hybrid `Pl`, and the sweep counter of its
`SweepPreconditioner`, are reset first.
"""
function timed_gmres!(
        u::AbstractVector, A, b::AbstractVector; Pl = nothing,
        flexible::Bool = is_nonlinear(Pl),
        reltol::Real = sqrt(eps(real(eltype(b)))), abstol::Real = 0.0,
        restart::Integer = min(20, size(A, 2)),
        maxiter::Integer = size(A, 2), kwargs...
    )
    τA = Ref(0.0)
    nA = Ref(0)
    Aτ = Inti.LinearMaps.LinearMap{eltype(u)}(size(A)...) do y, x
        t0 = time_ns()
        mul!(y, A, x)
        τA[] += (time_ns() - t0) / 1.0e9
        nA[] += 1
        return y
    end
    Pτ = Pl === nothing ? nothing : _TimedPl(Pl, Ref(0.0), Ref(0))
    Pl isa HybridPreconditioner && reset_timings!(Pl)
    S = _sweeper(Pl)
    S === nothing || (S.nsweeps[] = 0)
    t = @elapsed u, hist = if flexible
        isempty(kwargs) || throw(
            ArgumentError(
                "flexible GMRES takes no further keywords; got $(keys(kwargs))"
            )
        )
        _fgmres!(u, Aτ, b, Pτ; reltol, abstol, restart = Int(restart), maxiter = Int(maxiter))
    else
        kw = (; reltol, abstol, restart, maxiter, log = true, kwargs...)
        Pτ === nothing ? IterativeSolvers.gmres!(u, Aτ, b; kw...) :
            IterativeSolvers.gmres!(u, Aτ, b; Pl = Pτ, kw...)
    end
    tP = Pτ === nothing ? 0.0 : Pτ.t[]
    tS = Pl isa HybridPreconditioner ? timings(Pl).sparsifying : NaN
    times = (;
        total = t, forward = τA[], nforward = nA[],
        precond = tP, nprecond = Pτ === nothing ? 0 : Pτ.n[],
        sparsifying = tS, transfer = tP - tS,
        nsweeps = S === nothing ? nothing : S.nsweeps[],
        other = t - τA[] - tP,
    )
    return u, hist, times
end

"The `SweepPreconditioner` inside `Pl`, if there is one."
_sweeper(Pl) = nothing
_sweeper(S::SweepPreconditioner) = S
_sweeper(P::HybridPreconditioner) = _sweeper(P.sparsifying)

"""
    is_nonlinear(Pl)

Whether applying `Pl` is a nonlinear operation, so that GMRES must be flexible: a
`SweepPreconditioner` with inner iterations, alone or inside a
[`HybridPreconditioner`](@ref).
"""
is_nonlinear(Pl) = (S = _sweeper(Pl); S !== nothing && S.inner > 0)

"""
    _fgmres!(u, A, b, M; reltol, abstol, restart, maxiter)

Right-preconditioned flexible GMRES(`restart`): `z_k = M v_k` is stored, the
Arnoldi relation is `A Z_k = V_{k+1} H̄_k`, and the update is `u += Z_k y`, so `M`
may differ between applications.  Modified Gram-Schmidt, Givens rotations; every
restart recomputes the true residual, which also decides convergence.
"""
function _fgmres!(u, A, b, M; reltol, abstol, restart, maxiter)
    T = promote_type(eltype(u), eltype(b), ComplexF64)
    N = length(b)
    tol = max(reltol * norm(b), abstol)
    r = b - A * u
    β = norm(r)
    resnorms = [β]
    iters = 0
    converged = β <= tol
    V = Matrix{T}(undef, N, restart + 1)
    Z = Matrix{T}(undef, N, restart)
    H = zeros(T, restart + 1, restart)
    cs = zeros(real(T), restart)
    sn = zeros(T, restart)
    g = zeros(T, restart + 1)
    w = Vector{T}(undef, N)
    while !converged && iters < maxiter
        V[:, 1] .= r ./ β
        fill!(g, 0)
        g[1] = β
        k = 0
        while k < restart && iters < maxiter
            k += 1
            iters += 1
            z = view(Z, :, k)
            if M === nothing
                z .= view(V, :, k)
            else
                ldiv!(z, M, view(V, :, k))
            end
            mul!(w, A, z)
            for i in 1:k                                   # modified Gram-Schmidt
                H[i, k] = dot(view(V, :, i), w)
                axpy!(-H[i, k], view(V, :, i), w)
            end
            H[k + 1, k] = norm(w)
            V[:, k + 1] .= w ./ H[k + 1, k]
            for i in 1:(k - 1)                               # earlier rotations
                h = H[i, k]
                H[i, k] = cs[i] * h + sn[i] * H[i + 1, k]
                H[i + 1, k] = -conj(sn[i]) * h + cs[i] * H[i + 1, k]
            end
            c, s_, ρ = LinearAlgebra.givensAlgorithm(H[k, k], H[k + 1, k])
            cs[k], sn[k] = c, s_
            H[k, k], H[k + 1, k] = ρ, 0
            g[k + 1] = -conj(s_) * g[k]
            g[k] = c * g[k]
            push!(resnorms, abs(g[k + 1]))
            abs(g[k + 1]) <= tol && break
        end
        y = UpperTriangular(view(H, 1:k, 1:k)) \ g[1:k]
        mul!(u, view(Z, :, 1:k), y, true, true)
        r .= b .- A * u                                    # true residual
        β = norm(r)
        resnorms[end] = β
        converged = β <= tol
    end
    return u, (; iters, isconverged = converged, resnorms)
end

"A left preconditioner that times every application of the one it wraps."
struct _TimedPl{P}
    P::P
    t::Base.RefValue{Float64}
    n::Base.RefValue{Int}
end
function LinearAlgebra.ldiv!(M::_TimedPl, b::AbstractVector)
    t0 = time_ns()
    ldiv!(M.P, b)
    M.t[] += (time_ns() - t0) / 1.0e9
    M.n[] += 1
    return b
end
function LinearAlgebra.ldiv!(y::AbstractVector, M::_TimedPl, b::AbstractVector)
    t0 = time_ns()
    ldiv!(y, M.P, b)
    M.t[] += (time_ns() - t0) / 1.0e9
    M.n[] += 1
    return y
end

"""
    print_timings([io], times)

One line per entry of the breakdown returned by [`timed_gmres!`](@ref).
"""
print_timings(times) = print_timings(stdout, times)
function print_timings(io::IO, τ)
    pct(x) = 100 * x / τ.total
    per(x, n) = n > 0 ? x / n : 0.0
    Printf.@printf(
        io, "    forward map      %4d × %8.4f s = %7.2f s  (%4.1f%%)\n",
        τ.nforward, per(τ.forward, τ.nforward), τ.forward, pct(τ.forward)
    )
    if τ.nprecond > 0
        Printf.@printf(
            io, "    preconditioner   %4d × %8.4f s = %7.2f s  (%4.1f%%)\n",
            τ.nprecond, per(τ.precond, τ.nprecond), τ.precond, pct(τ.precond)
        )
        if !isnan(τ.sparsifying)
            Printf.@printf(
                io, "      Cartesian S                        %7.2f s  (%4.1f%%)\n",
                τ.sparsifying, pct(τ.sparsifying)
            )
            Printf.@printf(
                io, "      transfers                          %7.2f s  (%4.1f%%)\n",
                τ.transfer, pct(τ.transfer)
            )
        end
    end
    τ.nsweeps === nothing ||
        Printf.@printf(
        io, "    sweeps           %4d  (%.1f per preconditioner application)\n",
        τ.nsweeps, τ.nsweeps / max(τ.nprecond, 1)
    )
    Printf.@printf(
        io, "    GMRES other (orthogonalisation etc.)  %7.2f s  (%4.1f%%)\n",
        τ.other, pct(τ.other)
    )
    return nothing
end
