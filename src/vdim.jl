"""
    vdim_correction(op,X,Y,Y_boundary,S,D,V; green_multiplier, kwargs...)

Compute a correction to the volume potential `V : Y → X` such that `V + δV` is a
more accurate approximation of the underlying volume potential operator. The
correction is computed using the (volume) density interpolation method.

This function requires a `op::AbstractDifferentialOperator`, a target set `X`, a source
quadrature `Y`, a boundary quadrature `Y_boundary`, approximations `S :
Y_boundary -> X` and `D : Y_boundary -> X` to the single- and double-layer
potentials (correctly handling nearly-singular integrals), and a naive
approximation of the volume potential `V`. The `green_multiplier` is a vector of
the same length as `X` storing the value of `μ(x)` for `x ∈ X` in the Green
identity (see [`_green_multiplier`](@ref)).

See [anderson2024fast](@cite) for more details on the method.

## Optional `kwargs`:

- `interpolation_order`: the order of the polynomial interpolation. By default,
  the maximum order of the quadrature rules is used.
- `maxdist`: distance beyond which interactions are considered sufficiently far
  so that no correction is needed. This is used to determine a threshold for
  nearly-singular corrections.
- `kernel_variant`: `:default` for the standard volume potential, `:gradient` for
  the gradient of the volume potential. `S`, `D`, and `V` must be built with the
  same `kernel_variant`.
"""
# Helper: fill one row of bdata from Θ[i,m] — dispatches on element type
_vdim_fill_bdata!(bdata, val, m, ::Type{<:Number}) = (bdata[m, 1] = val)
_vdim_fill_bdata!(bdata, val, m, ::Type{<:SVector}) = (bdata[m, :] .= val)

# Helper: push the weight at quadrature node k into Vs — dispatches on Tout
_vdim_push_weight!(Vs, wdata, k, ::Type{T}) where {T<:Number} = push!(Vs, -wdata[k, 1])
_vdim_push_weight!(Vs, wdata, k, ::Type{SV}) where {SV<:SVector} = push!(Vs, -SV(wdata[k, :]))

function vdim_correction(
        op::AbstractDifferentialOperator{N},
        target,
        source::Quadrature{N},
        boundary::Quadrature,
        Sop,
        Dop,
        Vop;
        green_multiplier::Vector{<:Real},
        interpolation_order = nothing,
        kernel_variant::Symbol = :default,
        maxdist = Inf,
    ) where {N}
    # variables for debugging the condition properties of the method
    vander_cond = vander_norm = rhs_norm = res_norm = shift_norm = -Inf
    Tout = eltype(Vop)
    Tbase = default_kernel_eltype(op)
    # determine type for dense matrices
    DenseOut = Tout <: SMatrix ? BlockArray : Array
    DenseBase = Tbase <: SMatrix ? BlockArray : Array
    @assert eltype(Dop) == eltype(Sop) == Tout "eltype of Sop, Dop, and Vop must match"
    # figure out if we are dealing with a scalar or vector PDE
    num_target, num_source = length(target), length(source)
    # a reasonable interpolation_order if not provided
    isnothing(interpolation_order) &&
        (interpolation_order = maximum(order, values(source.etype2qrule)))
    # check if we are in debug mode to avoid expensive computations
    do_debug = debug_mode()
    # by default basis centered at origin
    basis = polynomial_solutions_vdim(op, interpolation_order, Tbase)
    dict_near = etype_to_nearest_points(target, source; maxdist)
    num_basis = length(basis)
    b = DenseBase{Tbase}(undef, length(source), num_basis)
    γ₀B = DenseBase{Tbase}(undef, length(boundary), num_basis)
    γ₁B = DenseBase{Tbase}(undef, length(boundary), num_basis)
    for k in 1:num_basis, j in 1:length(source)
        b[j, k] = basis[k].source(source[j])
    end
    for k in 1:num_basis, j in 1:length(boundary)
        γ₀B[j, k] = basis[k].solution(boundary[j])
        γ₁B[j, k] = basis[k].neumann_trace(boundary[j])
    end
    Θ = DenseOut{Tout}(undef, num_target, num_basis)
    fill!(Θ, zero(Tout))
    # Compute Θ <-- S * γ₁B - D * γ₀B + V * b + σ * B(x) using in-place matvec
    if DenseOut <: Array || (Sop isa BlockArray && Dop isa BlockArray && Vop isa BlockArray)
        for n in 1:num_basis
            @views mul!(Θ[:, n], Sop, γ₁B[:, n])
            @views mul!(Θ[:, n], Dop, γ₀B[:, n], -1, 1)
            @views mul!(Θ[:, n], Vop, b[:, n], 1, 1)
        end
    else
        # For vector-valued problems with FMM (LinearMap operators), we need to
        # perform multiplication column-by-column since FMM expects vector densities
        # (SVector) not matrix densities (SMatrix). See bdim.jl for similar handling.
        P, Q = size(Tout)
        S = eltype(Tout)
        Θ_data = parent(Θ)::Matrix
        γ₀B_data = parent(γ₀B)::Matrix
        γ₁B_data = parent(γ₁B)::Matrix
        b_data = parent(b)::Matrix
        for k in 1:size(Θ_data, 2)
            y = reinterpret(SVector{P, S}, @view Θ_data[:, k])
            x = reinterpret(SVector{Q, S}, @view γ₁B_data[:, k])
            mul!(y, Sop, x)
            x = reinterpret(SVector{Q, S}, @view γ₀B_data[:, k])
            mul!(y, Dop, x, -1, 1)
            x = reinterpret(SVector{Q, S}, @view b_data[:, k])
            mul!(y, Vop, x, 1, 1)
        end
    end
    # Add σ * B(x) term
    for n in 1:num_basis
        for i in 1:num_target
            if kernel_variant === :gradient
                Θ[i, n] += green_multiplier[i] * basis[n].gradient_solution(target[i])
            else
                Θ[i, n] += green_multiplier[i] * basis[n].solution(target[i])
            end
        end
    end
    # compute sparse correction
    Is = Int[]
    Js = Int[]
    Vs = eltype(Vop)[]
    for (E, qtags) in source.etype2qtags
        near_list = dict_near[E]
        nq, ne = size(qtags)
        @assert length(near_list) == ne
        L_arr = DenseBase{Tbase}(undef, num_basis, nq)
        Ldata = parent(L_arr)::Matrix
        # Preallocate solve buffers. Vector-valued PDEs (Tout <: SMatrix or
        # Tout <: SVector{K,<:SMatrix}) need BlockArray so that LAPACK sees a
        # plain float matrix while we can still index with SMatrix semantics.
        # Scalar/SVector{P,<:Number} PDEs use a plain float matrix directly.
        if Tout <: SMatrix
            b_arr = BlockArray{Tout}(undef, num_basis, 1)
            wei_arr = BlockArray{Tout}(undef, nq, 1)
            bdata = parent(b_arr)::Matrix
            weidata = parent(wei_arr)::Matrix
        elseif Tout <: SVector && eltype(Tout) <: SMatrix
            K = length(Tout)
            SM = eltype(Tout)
            b_kk = BlockArray{SM}(undef, num_basis, 1)
            wei_kk = BlockArray{SM}(undef, nq, 1)
            bdata_kk = parent(b_kk)::Matrix
            weidata_kk = parent(wei_kk)::Matrix
            Vs_mat = Matrix{SM}(undef, nq, K)
        else
            # Scalar (Tout <: Number) and gradient-scalar (Tout <: SVector{P,<:Number})
            # cases share the same structure: P columns in the solve, where P=1 for scalar.
            S = eltype(Tout)
            P = Tout <: Number ? 1 : length(Tout)
            bdata = Matrix{S}(undef, num_basis, P)
            weidata = Matrix{S}(undef, nq, P)
        end
        for n in 1:ne
            isempty(near_list[n]) && continue
            jglob = @view qtags[:, n]
            for k in 1:nq, m in 1:num_basis
                L_arr[m, k] = basis[m].source(view(source, jglob)[k])
            end
            F = svd(Ldata)
            if do_debug
                vander_cond = max(vander_cond, cond(Ldata))
                shift_norm = max(shift_norm, 1)
                vander_norm = max(vander_norm, norm(Ldata))
            end
            for i in near_list[n]
                if Tout <: SMatrix
                    b_arr .= @views transpose(Θ[i:i, :])
                    ldiv!(weidata, F, bdata)
                    if do_debug
                        rhs_norm = max(rhs_norm, norm(bdata))
                        res_norm = max(res_norm, norm(Ldata * weidata - bdata))
                    end
                    for k in 1:nq
                        push!(Is, i)
                        push!(Js, jglob[k])
                        push!(Vs, -transpose(wei_arr[k]))
                    end
                elseif Tout <: SVector && eltype(Tout) <: SMatrix
                    for kk in 1:K
                        for m in 1:num_basis
                            b_kk[m, 1] = transpose(Θ[i, m][kk])
                        end
                        ldiv!(weidata_kk, F, bdata_kk)
                        for k in 1:nq
                            Vs_mat[k, kk] = -transpose(wei_kk[k])
                        end
                    end
                    for k in 1:nq
                        push!(Is, i)
                        push!(Js, jglob[k])
                        push!(Vs, Tout(ntuple(kk -> Vs_mat[k, kk], K)))
                    end
                else
                    for m in 1:num_basis
                        _vdim_fill_bdata!(bdata, Θ[i, m], m, Tout)
                    end
                    ldiv!(weidata, F, bdata)
                    if do_debug
                        rhs_norm = max(rhs_norm, norm(bdata))
                        res_norm = max(res_norm, norm(Ldata * weidata - bdata))
                    end
                    for k in 1:nq
                        push!(Is, i)
                        push!(Js, jglob[k])
                        _vdim_push_weight!(Vs, weidata, k, Tout)
                    end
                end
            end
        end
    end
    @debug """Condition properties of vdim correction:
    |-- max interp. matrix condition: $vander_cond
    |-- max norm of source term:      $rhs_norm
    |-- max residual error:           $res_norm
    |-- max interp. matrix norm :     $vander_norm
    |-- max shift norm :              $shift_norm
    """
    δV = sparse(Is, Js, Vs, num_target, num_source)
    return δV
end

"""
    polynomial_solutions_vdim(op, order, [T])

Build a basis of polynomial solutions for the VDIM method.

For every monomial `pₙ` of degree at most `order`, computes a polynomial solution `Pₙ`
satisfying `ℒ[Pₙ] = pₙ`, where `ℒ` is the differential operator of `op`.

Returns a vector of named tuples with fields `source = pₙ`, `solution = Pₙ`, `neumann_trace = γ₁Pₙ`, and `gradient_solution = ∇Pₙ`.
"""
function polynomial_solutions_vdim(
        op::AbstractDifferentialOperator{N},
        order::Integer,
        ::Type{T} = default_kernel_eltype(op),
    ) where {N, T}
    indices = [I for I in Iterators.product(ntuple(i -> 0:order, N)...) if sum(I) <= order]
    return map(indices) do I
        monomial = Polynomial(I => one(T))
        P, γ₁P = basis_from_monomial(op, monomial)
        ∇P = ElementaryPDESolutions.gradient(P)
        (source = monomial, solution = P, neumann_trace = γ₁P, gradient_solution = ∇P)
    end
end

"""
    basis_from_monomial(op, monomial) -> (solution, neumann_trace)

Compute a polynomial solution `P` to `ℒ[P] = monomial` and its Neumann trace `γ₁P`.

Each operator implements this to handle its specific PDE structure, including any
auxiliary fields (e.g., pressure for Stokes) needed to compute the Neumann trace.
"""
function basis_from_monomial end


# Laplace
function basis_from_monomial(::Laplace{N}, monomial::Polynomial{N, T}) where {N, T}
    P = ElementaryPDESolutions.solve_laplace(-monomial)
    ∇P = ElementaryPDESolutions.gradient(P)
    γ₁P = q -> dot(normal(q), ∇P(coords(q)))
    return P, γ₁P
end

# Helmholtz

function basis_from_monomial(op::Helmholtz{N}, monomial::Polynomial{N, T}) where {N, T}
    P = ElementaryPDESolutions.solve_helmholtz(-monomial, op.k^2)
    ∇P = ElementaryPDESolutions.gradient(P)
    γ₁P = q -> dot(normal(q), ∇P(coords(q)))
    return P, γ₁P
end

# Elastostatic

function basis_from_monomial(op::Elastostatic{N}, monomial::Polynomial{N, T}) where {N, T}
    monomial = -monomial
    @assert T <: StaticMatrix && size(T) == (N, N)
    S = eltype(T)
    ord2coef = monomial.order2coeff
    @assert length(ord2coef) == 1 "Input must be a monomial"
    coef = first(values(ord2coef))
    idx = first(keys(ord2coef))
    μ, λ = op.μ, op.λ
    ν = λ / (2 * (λ + μ))
    # Solve for each column of the tensor
    sol_tuple = ntuple(N) do n
        p = ntuple(d -> Polynomial(idx => coef[d, n]), N)
        u = ElementaryPDESolutions.solve_elastostatic(p; μ, ν)
        ntuple(d -> convert(Polynomial{N, S}, u[d]), N)
    end
    P = flatten_polynomial_ntuple(sol_tuple)
    ∇P = ElementaryPDESolutions.gradient(P)
    # Neumann trace: traction vector
    γ₁P = q -> begin
        n = normal(q)
        x = coords(q)
        M = ∇P(x)  # M[j] = ∂P/∂xⱼ
        cols = svector(N) do m
            gradu = hcat(ntuple(j -> M[j][:, m], N)...)
            divu = tr(gradu)
            λ * divu * n + μ * (gradu + gradu') * n
        end
        reduce(hcat, cols)
    end
    return P, γ₁P
end

# Stokes

function basis_from_monomial(op::Stokes{N}, monomial::Polynomial{N, T}) where {N, T}
    monomial = -monomial
    @assert T <: StaticMatrix && size(T) == (N, N)
    S = eltype(T)
    ord2coef = monomial.order2coeff
    @assert length(ord2coef) == 1 "Input must be a monomial"
    coef = first(values(ord2coef))
    idx = first(keys(ord2coef))
    μ = op.μ
    # Solve for each column: velocity and pressure
    solutions = ntuple(N) do n
        f = ntuple(d -> Polynomial(idx => coef[d, n]), N)
        u, p = ElementaryPDESolutions.solve_stokes(f; μ)
        vel = ntuple(d -> convert(Polynomial{N, S}, u[d]), N)
        pres = convert(Polynomial{N, S}, p)
        (velocity = vel, pressure = pres)
    end
    velocities = ntuple(n -> solutions[n].velocity, N)
    pressures = ntuple(n -> solutions[n].pressure, N)
    U = flatten_polynomial_ntuple(velocities)
    ∇U = ElementaryPDESolutions.gradient(U)
    # Neumann trace: traction using velocity gradient and pressure
    γ₁U = q -> begin
        n = normal(q)
        x = coords(q)
        M = ∇U(x)  # M[j] = ∂U/∂xⱼ
        cols = svector(N) do m
            gradu = hcat(ntuple(j -> M[j][:, m], N)...)
            p_val = pressures[m](x)
            -p_val * n + μ * (gradu + gradu') * n
        end
        reduce(hcat, cols)
    end
    return U, γ₁U
end

function flatten_polynomial_ntuple(P::NTuple{N, NTuple{N, Polynomial{DIM, T}}}) where {N, DIM, T <: Number}
    V = SMatrix{N, N, T, N * N}
    # collect all multi-indices
    idxs = Set{NTuple{DIM, Int}}()
    foreach(p -> union!(idxs, keys(p.order2coeff)), Iterators.flatten(P))
    # now loop over keys and build flattened coefficients
    idx2coef = Dict{NTuple{DIM, Int}, V}()
    for idx in idxs
        coef_tuple = ntuple(N^2) do n
            m = div(n - 1, N) + 1
            l = mod(n - 1, N) + 1
            p = P[m][l]
            get(p.order2coeff, idx, zero(T))
        end
        idx2coef[idx] = V(coef_tuple)
    end
    return Polynomial{DIM, V}(idx2coef)
end

function (P::NTuple{N, <:Polynomial})(x) where {N}
    return svector(n -> P[n](x), N)
end

function (P::Polynomial)(q::QuadratureNode)
    x = coords(q)
    return P(x)
end

function (P::NTuple{N, <:Polynomial})(q::QuadratureNode) where {N}
    x = coords(q)
    return P(x)
end
