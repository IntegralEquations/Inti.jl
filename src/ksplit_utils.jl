using SpecialFunctions
using SparseArrays
using StaticArrays
using OffsetArrays
using NearestNeighbors
using FastGaussQuadrature
using LinearAlgebra
using LinearMaps
using SpecialMatrices
using ClassicalOrthogonalPolynomials

"""
    WfrakLinit(trans, scale, tfrak, npt) -> Matrix{Float64}

Computes the `npt × npt` product integration weight matrix `mathfrak(W)_L` for a
logarithmic singularity, as defined in Appendix A of Helsing & Holst (2015).

This function is used for panel-to-panel (on-surface) corrections, where
`trans` and `scale` define the target panel's position relative to the source
panel in canonical coordinates. It computes the weights for all `npt` target
nodes against all `npt` source nodes simultaneously.

This is called for self-panel corrections (with `trans=0`, `scale=1`) and
neighbor-panel corrections.

See also [`helsing2015variants`](@ref).

# Arguments
- `trans`: Translation of the target panel relative to the source panel,
  defined as `(C_k - C_j) / H_j`.
- `scale`: Scaling of the target panel relative to the source panel,
  defined as `H_k / H_j`.
- `tfrak`: Vector of `npt` canonical Gauss-Legendre nodes on `[-1, 1]`
  (source nodes).
- `npt`: The number of quadrature points per panel.

# Returns
- `Matrix{Float64}`: The `npt × npt` weight matrix `mathfrak(W)_L`.
"""

# Weight matrix for product integration of log singularity when target = source
function WfrakLinit(trans, scale, tfrak, npt)
    A = Vandermonde(tfrak)
    tt = trans .+ scale .* tfrak
    Q = zeros(npt, npt)
    p = zeros(npt + 1)
    c = (1 .- (-1) .^ (1:npt)) ./ (1:npt)

    for m in 1:npt
        p[1] = log(abs((1 - tt[m]) / (1 + tt[m])))
        p1 = log(abs(1 - tt[m]^2))

        for k in 1:npt
            p[k+1] = tt[m] * p[k] + c[k]
        end

        Q[m, 1:2:npt-1] = p1 .- p[2:2:npt]
        Q[m, 2:2:npt] = p[1] .- p[3:2:npt+1]
        Q[m, :] ./= (1:npt)
    end

    return Q / A
end

"""
    WfrakLinit(tt, tfrak, npt) -> Vector{Float64}

Computes the `npt`-element product integration weight vector `mathfrak(w)_L` for a
logarithmic singularity for a single target point `tt`.

This is a specialized, multiple-dispatch version used in adaptive kernel split.
It calculates the weights for one target point relative to all `npt` source nodes of a (sub)panel.

# Arguments
- `tt`: The canonical coordinate `t` of the single target point, mapped into
  the `[-1, 1]` domain of the source (sub)panel.
- `tfrak`: Vector of `npt` canonical Gauss-Legendre nodes on `[-1, 1]`
  (representing the source nodes).
- `npt`: The number of quadrature points per panel.

# Returns
- `Vector{Float64}`: The `npt`-element weight vector `mathfrak(w)_L` for the
  target point `tt`.
"""

# Weight matrix for product integration of log singularity when target = source
# taking the target node as the argument
function WfrakLinit(tt, tfrak, npt)
    A = Vandermonde(tfrak)
    q = zeros(npt)
    p = zeros(npt + 1)
    c = (1 .- (-1) .^ (1:npt)) ./ (1:npt)

    p[1] = log(abs((1 - tt) / (1 + tt)))
    p1 = log(abs(1 - tt^2))

    for k in 1:npt
        p[k+1] = tt * p[k] + c[k]
    end

    q[1:2:npt-1] = p1 .- p[2:2:npt]
    q[2:2:npt] = p[1] .- p[3:2:npt+1]
    q ./= (1:npt)

    return A' \ q
end

"""
    wLCinit(ra, rb, r, rj, nuj, rpwj, npt; target_location) -> Tuple

Computes the log-correction weights (`wcorrL`) and Cauchy-compensation
weights (`wcmpC`) for an off-surface target point `r`.

This function is a Julia implementation of the MATLAB code from Appendix B
of Helsing & Holst (2015).

Crucially, this version is generalized to handle the complex logarithm's
branch cut for both interior and exterior problems via the
`target_location` argument. The sign convention for the
`target_location = :inside` correction is reversed from the
exterior problem presented in the paper to correctly handle interior problems.

See also [`helsing2015variants`](@ref).

# Arguments
- `ra`, `rb`: Complex coordinates of the source panel's start and end points.
- `r`: Complex coordinate of the single off-surface target point.
- `rj`: Vector of `npt` complex coordinates of the source quadrature nodes.
- `nuj`: Vector of `npt` complex unit normals at the source nodes.
- `rpwj`: Vector of `npt` complex weighted velocity values (`r'(t_j) * w_j`).
- `npt`: The number of quadrature points.
- `target_location`: A `Symbol` (e.g., `:inside` or `:outside`) specifying
  the problem domain relative to the boundary. This is required to
  apply the correct branch cut rule for the complex logarithm.

# Returns
- `(wcorrL, wcmpC)`: A tuple containing:
    - `wcorrL`: An `npt`-element `Vector{Float64}` of log-correction weights.
    - `wcmpC`: An `npt`-element `Vector{Float64}` of Cauchy-compensation weights.
"""

# Weight and compensation matrix for product integration of log and Cauchy singularity for generic target
function wLCinit(ra, rb, r, rj, nuj, rpwj, npt; target_location)
    dr = (rb - ra) / 2
    rtr = (r - (rb + ra) / 2) / dr
    rjtr = (rj .- (rb + ra) / 2) ./ dr

    # Construct the Vandermonde matrix
    A = transpose(Vandermonde(rjtr))

    # Initialize variables
    p = zeros(ComplexF64, npt + 1)
    q = zeros(ComplexF64, npt)
    c = (1 .- (-1) .^ (1:npt)) ./ (1:npt)

    # Compute the logarithmic expansion for the target point
    p[1] = log(Complex(1 - rtr)) - log(Complex(-1 - rtr))
    p1 = log(Complex(1 - rtr) * (-1 - rtr))

    # Apply logarithmic branch cut correction 
    if target_location == :inside
        if imag(rtr) < 0 && abs(real(rtr)) < 1
            p[1] += 2im * π
            p1 -= 2im * π
        end
    elseif target_location == :outside
        if imag(rtr) > 0 && abs(real(rtr)) < 1
            p[1] -= 2im * π
            p1 += 2im * π
        end
    end

    # Compute recurrence relation for expansion terms
    for k in 1:npt
        p[k+1] = rtr * p[k] + c[k]
    end

    # Construct weight correction vector q (Helsing's code)
    q[1:2:npt-1] = p1 .- p[2:2:npt]
    q[2:2:npt] = p[1] .- p[3:2:npt+1]

    q ./= (1:npt)

    # Solve for weight corrections wcorrL
    wcorrL =
        real(A \ q * dr .* conj(im * nuj)) ./ abs.(rpwj) .+
        (log(abs(dr)) .- log.(abs.(rj .- r)))

    wcmpC = imag(A \ -p[1:npt]) - imag(rpwj ./ (r .- rj))

    return wcorrL, wcmpC
end

# Constant matrix associated with interpolation using Legendre polynomials 
function L_Legendre_matrix(n)
    L_Leg = Matrix{Float64}(undef, n, n)
    t_Leg_ref, w_Leg_ref = gausslegendre(n)
    for l in 0:n-1
        for m in 1:n
            L_Leg[l+1, m] = (2 * l + 1) / 2 * legendrep(l, t_Leg_ref[m]) * w_Leg_ref[m]
        end
    end
    return L_Leg
end

# derivative of Legendre polynomials
function Legendre_derivative(l, t)
    if l == 0
        return 0
    else
        if t == 1 || t == -1
            error("Derivative of Legendre polynomials is not defined at t = ±1 in Legendre_derivative")
        end
        return l / (t^2 - 1) * (t * legendrep(l, t) - legendrep(l - 1, t))
    end
end

# Interpolation using Legendre polynomials
function curve_interpolate(x_vec, L_leg)
    n = length(x_vec)
    γ_coef = L_leg * x_vec
    Pₙ_vec = [t -> γ_coef[l+1] * legendrep(l, t) for l in 0:n-1]
    Pₙ = t -> mapreduce(f -> f(t), +, Pₙ_vec)
    return Pₙ
end

# Derivative of the interpolation using Legendre polynomials
function curve_interpolate_derivative(x_vec, L_leg)
    n = length(x_vec)
    γ_coef = L_leg * x_vec
    Pₙ_deriv_vec = [t -> γ_coef[l+1] * Legendre_derivative(l, t) for l in 0:n-1]
    Pₙ_deriv = t -> mapreduce(f -> f(t), +, Pₙ_deriv_vec)
    return Pₙ_deriv
end

function NewtonRaphson(f, fp, x0; tol=1e-8, maxIter=1000)
    x = x0
    fx = f(x0)
    iter = 0
    while abs(fx) > tol && iter < maxIter
        x = x - fx / fp(x)
        fx = f(x)
        iter += 1
    end
    return x
end

# Get the n-point Gauss-Legendre quadrature nodes.
function LegendreNodes(n)
    xnodes, _ = gausslegendre(n)
    return xnodes
end

# Compute the barycentric weights for polynomial interpolation on the n-point Legendre nodes.
function LegendreBaryWeights(n)
    xnodes = LegendreNodes(n)
    xweights = ones(n)
    for j in 1:n
        term_j = 1
        for i in 1:n
            i == j && continue
            term_j *= (xnodes[j] - xnodes[i])
        end
        xweights[j] = 1 / term_j
    end
    return xweights
end

# Compute the barycentric weights for polynomial interpolation on a given set of nodes.
function InterpolationBaryWeights(xnodes, n)
    xweights = ones(n)
    for j in 1:n
        term_j = 1
        for i in 1:n
            i == j && continue
            term_j *= xnodes[j] - xnodes[i]
        end
        xweights[j] = 1 / term_j
    end
    return xweights
end

# Compute the barycentric interpolation coefficients for a single target point.
function Barycentric_coef(t_target; n=16, t_interp=nothing, w=nothing)
    if t_interp === nothing
        t_interp = LegendreNodes(n)
    end
    if w === nothing
        w = LegendreBaryWeights(n)
    end

    if length(t_interp) != n || length(w) != n
        error("Length of t_interp and w must match n")
    end

    coef = Vector{Float64}(undef, n)

    for j in 1:n
        if abs(t_target - t_interp[j]) < 1e-15
            coef .= 0
            coef[j] = 1
            return coef
        end
        coef[j] = w[j] / (t_target - t_interp[j])
    end

    coef_sum = sum(coef)
    coef ./= coef_sum

    return coef
end

# scale function (from [-1,1] to [t_a,t_b])
function scale_fn(t_a, t_b, vec_ref; node=true)
    return if node == true
        (t_b - t_a) / 2 .* vec_ref .+ (t_b + t_a) / 2
    else
        (t_b - t_a) / 2 .* vec_ref
    end
end

# Compute the n-th harmonic number (H_n = 1 + 1/2 + ... + 1/n).
function harmonic_sum(n::Int)
    val = 0
    if n == 0
        return val
    else
        for i in range(1, n)
            val += 1 / i
        end
        return val
    end
end

γ = Base.MathConstants.γ # Euler-Mascheroni constant

# kernel split for the Helmholtz equation
# single layer kernel splitting
function hankelh1_0_smooth(z; tol=1e-15, n_term=10000)
    term = ComplexF64[]
    push!(term, (1 + 2 * im / π * (γ - log(2))) * besselj0(z))
    push!(term, 2 * im / π * 1 / 4 * z^2)
    idx = 2

    while maximum([abs(term[idx]) / abs(term[1]), abs(term[idx])]) > tol && idx <= n_term
        idx += 1
        push!(term, (-1)^idx * 2 * im / π * harmonic_sum(idx - 1))
        for i in range(1, idx - 1)
            term[idx] *= (1 / 4 * z^2) / i^2
        end
    end

    term_real = real(term)
    term_imag = imag(term)
    term_sum = sum(sort(term_real; by=abs)) + im * sum(sort(term_imag; by=abs))

    return term_sum
end

# double layer kernel splittings
function hankelh1_1_smooth(z; tol=1e-15, n_term=10000)
    term = ComplexF64[]
    push!(term, (1 - 2 * im / π * log(2)) * besselj1(z))
    push!(term, -im / (2π) * z * (digamma(1) + digamma(2)))

    idx = 2
    while maximum([abs(term[idx]) / abs(term[1]), abs(term[idx])]) > tol && idx <= n_term
        idx += 1
        push!(term, (-1)^(idx - 1) * im / (2π) * z * (digamma(idx - 1) + digamma(idx)))
        for i in range(1, idx - 2)
            term[idx] *= (1 / 4 * z^2) / (i * (i + 1))
        end
    end

    term_real = real(term)
    term_imag = imag(term)
    term_sum = sum(sort(term_real; by=abs)) + im * sum(sort(term_imag; by=abs))

    return term_sum
end

# kernel split for the modified Helmholtz equation
# single layer kernel splitting
function besselk0_smooth(z; tol=1e-15, n_term=10000)
    term = ComplexF64[]
    push!(term, (log(2) - γ) * besseli(0, z))
    push!(term, 1 / 4 * z^2)
    idx = 2

    while maximum([abs(term[idx]) / abs(term[1]), abs(term[idx])]) > tol && idx <= n_term
        push!(term, harmonic_sum(idx))
        for i in range(1, idx)
            term[idx+1] *= (1 / 4 * z^2) / i^2
        end
        idx += 1
    end

    term_real = real(term)
    term_imag = imag(term)
    term_sum = sum(sort(term_real; by=abs)) + im * sum(sort(term_imag; by=abs))

    return term_sum
end

# double layer kernel splitting
function besselk1_smooth(z; tol=1e-15, n_term=10000)
    term = ComplexF64[]
    push!(term, -log(2) * besseli(1, z))
    push!(term, -1 / 4 * z * (digamma(1) + digamma(2)))
    idx = 2

    while maximum([abs(term[idx]) / abs(term[1]), abs(term[idx])]) > tol && idx <= n_term
        push!(term, -1 / 4 * z * (digamma(idx) + digamma(idx + 1)))
        for i in range(1, idx - 1)
            term[idx+1] *= (1 / 4 * z^2) / (i * (i + 1))
        end
        idx += 1
    end

    term_real = real(term)
    term_imag = imag(term)
    term_sum = sum(sort(term_real; by=abs)) + im * sum(sort(term_imag; by=abs))

    return term_sum
end

# Find target points `X` near source elements in `Y` within `maxdist`.
function near_points_vec(X, Y::Inti.Quadrature; maxdist=0.1)
    x = [Inti.coords(q) for q in X]
    y = [Inti.coords(q) for q in Y]

    kdtree = KDTree(y)
    dict = Dict(j => Set{Int}() for j in 1:length(y))

    for i in eachindex(x)
        qtags = inrange(kdtree, x[i], maxdist)
        for qtag in qtags
            push!(dict[qtag], i)
        end
    end

    etype2nearlist = Dict{DataType,Vector{Vector{Int}}}()
    for (E, tags) in Y.etype2qtags
        nq, ne = size(tags)
        nearlist = [Set{Int}() for _ in 1:ne]

        for j in 1:ne
            for q in 1:nq
                qtag = tags[q, j]
                union!(nearlist[j], dict[qtag])
            end
        end

        # Convert Sets to Vectors for the final output
        etype2nearlist[E] = [collect(s) for s in nearlist]
    end

    return etype2nearlist
end

# local mesh refinement (non-recursive implementation)
function create_subdivision(z, h, α, Cε, Rε)
    Δt_max = 2 * Cε / (α * h)
    result = Tuple[]  # To store valid subintervals
    ksplit_bool = Bool[] # Store vector of bools to indicate whether kernel split is needed
    stack_val = []   # Stack for iterative processing

    if abs(real(z)) >= 1
        # Add the entire interval to the stack for recursive bisection
        push!(stack_val, (-1, 1, Rε, Δt_max, z))
    else
        # Compute the center interval
        Δt_direct = 4 * abs(imag(z)) * Rε / (Rε^2 - 1)
        Δt_edge = 2 * (1 - abs(real(z)))

        Δt_c = min(Δt_edge, max(Δt_direct, Δt_max))

        ta = real(z) - Δt_c / 2
        tb = real(z) + Δt_c / 2

        if Δt_c <= Δt_direct
            push!(result, (ta, tb))
            push!(ksplit_bool, false)
        elseif Δt_c <= Δt_max
            push!(result, (ta, tb))
            push!(ksplit_bool, true)
        else
            error("Invalid condition for kernel split")
        end

        # Add the left and right subintervals to the stack
        push!(stack_val, (-1, ta, Rε, Δt_max, z))
        push!(stack_val, (tb, 1, Rε, Δt_max, z))
    end

    # Process the stack iteratively using recursive_bisection logic
    while !isempty(stack_val)
        t1, t2, Rε, Δt_max, z = pop!(stack_val)
        recursive_bisection!(t1, t2, Rε, Δt_max, z, result, ksplit_bool, stack_val)
    end

    return result, ksplit_bool
end

# Recursive bisection helper function
function recursive_bisection!(t1, t2, Rε, Δt_max, z, result, ksplit_bool, stack_val)
    if t1 < t2
        Δt_sub = t2 - t1
        z_sub = 2 * (z - t1) / Δt_sub - 1  # Transformation for preimage

        if ρ(z_sub) > Rε
            push!(result, (t1, t2))
            push!(ksplit_bool, false)
            return
        elseif Δt_sub <= Δt_max
            push!(result, (t1, t2))
            push!(ksplit_bool, true)
            return
        else
            t_mid = t1 + Δt_sub / 2
            push!(stack_val, (t1, t_mid, Rε, Δt_max, z))
            push!(stack_val, (t_mid, t2, Rε, Δt_max, z))
            return
        end
    end
end

# Compute the radius of Bernstein ellipse for a given complex number z 
# relative to the interval [-1, 1].
function ellip_rad(z)
    return abs(z + sqrt(complex(z + 1)) * sqrt(complex(z - 1)))
end

ρ = z -> ellip_rad(z)

# Generate the Newton function for root finding
function newton_fn_generator(panel_interpolant, x_target)::Function
    function newton_fn(t::ComplexF64)
        return panel_interpolant(t) - x_target
    end
    return newton_fn
end

# Generate the panel parametrization function
function panel_parametrization_fn(el)
    _body = t -> SVector(el(scale_fn(0, 1, t)))

    # multiple dispatch to handle different input types
    function panel_parametrization(t::Float64)::SVector{2,Float64}
        return _body(t)
    end

    function panel_parametrization(t)
        return _body(t)
    end

    return panel_parametrization
end

"""
    adaptive_refinement(
        source_panel,
        source_el,
        L_Leg,
        α,
        Cε,
        Rε,
        target;
        affine_preimage::Bool=true,
    )

Implements the per-target adaptive refinement algorithm from Fryklund et al. (2022)
to ensure accurate kernel-split quadrature for parameter-dependent PDEs (e.g.,
modified Helmholtz) where the parameter `α` is large.

For each target point, the algorithm computes its complex preimage `z` on the
source panel's canonical interval `[-1, 1]`. It then generates a list of
subintervals, `(t_a, t_b)`, that are "acceptable" for quadrature.

The algorithm (from `create_subdivision` and `recursive_bisection!`)
iteratively bisects the interval `[-1, 1]` until every resulting subinterval
`(t1, t2)` is "acceptable". An interval is deemed acceptable if it meets one
of two stopping conditions:

1.  Well-Separated (Standard Quadrature): The interval is sufficiently
    far from the target's preimage `z`, relative to its own length.
    -   Condition: `ρ(z_sub) > Rε`
    -   Action: Add `(t1, t2)` to the list. Mark as `ksplit_bool = false`.

2.  Sufficiently Small (Kernel-Split Quadrature): The interval is
    close to `z` (`ρ(z_sub) <= Rε`), but it is small enough for the
    kernel-split quadrature to be accurate given the large `α` parameter.
    -   Condition: `Δt_sub <= Δt_max` (where `Δt_max = 2*Cε / (α*h)`)
    -   Action: Add `(t1, t2)` to the list. Mark as `ksplit_bool = true`.

If an interval is both close and too large, it is bisected, and its
children are checked again.

This process ensures that the smooth coefficient functions (like `G^L`) of the
kernel-split are accurately resolved by polynomials on each subpanel,
preventing the accuracy degradation that occurs when `α*h` is large.

See also [`fryklund2022adaptive`](@ref).

# Arguments
- `source_panel`: The `Quadrature` object for the source panel. Used to compute
  the panel's total arc length `h`.
- `source_el`: The parametrization function for the source panel.
- `L_Leg`: Precomputed Legendre interpolation matrix, used only if
  `affine_preimage=false`.
- `α`: The PDE parameter (e.g., `λ` in `(Δ - λ²)u = 0`).
- `Cε`: The accuracy constant used to determine the max subinterval length
  `Δt_max` (from Eq. 75).
- `Rε`: The cutoff radius for the near-field criterion (from Eq. 72).
- `target`: A vector of target points to generate subdivisions for.
- `affine_preimage`:
    - `true` (default): Use a fast, direct affine map to find the preimage `z`.
    - `false`: Use the affine map as an initial guess, then refine `z` with
      Newton's method on a polynomial panel interpolant.

# Returns
- `panel_subdivision::Vector{Vector{Tuple}}`: A list where `panel_subdivision[j]`
  is the vector of `(t_start, t_end)` subintervals on `[-1, 1]` for `target[j]`.
- `ksplit_bool::Vector{Vector{Bool}}`: A list where `ksplit_bool[j][k]` is `true`
  if the `k`-th subinterval for `target[j]` requires kernel-split quadrature.
"""
function adaptive_refinement(
    source_panel,
    source_el,
    L_Leg,
    α,
    Cε,
    Rε,
    target;
    affine_preimage::Bool=true,
)
    # Convert target points to complex numbers
    nᵢ = length(target)
    target_complex = Vector{ComplexF64}(undef, nᵢ)
    for i in axes(target)[1]
        x = Inti.coords(target[i])
        target_complex[i] = x[1] + im * x[2]
    end

    # Get panel endpoints (needed for both affine and interpolant methods)
    panel_parametrization = t -> source_el(scale_fn(0, 1, t))
    z_a_complex = panel_parametrization(-1)[1] + im * panel_parametrization(-1)[2]
    z_b_complex = panel_parametrization(1)[1] + im * panel_parametrization(1)[2]

    # Pre-calculate affine map components for efficiency
    z_mid = (z_a_complex + z_b_complex) / 2
    z_diff = (z_b_complex - z_a_complex)

    panel_arc_length = sum(q -> Inti.weight(q), source_panel)
    panel_subdivision = Vector{Tuple}[]
    ksplit_bool = Vector{Bool}[]

    # Construct interpolant if needed
    local panel_interpolant, panel_interpolant_deriv
    if affine_preimage == false
        # This setup is only performed if the interpolant method is used.
        qnodes_i = [Inti.coords(qnodes) for qnodes in source_panel]
        qnodes_i_complex = [nodes[1] + im * nodes[2] for nodes in qnodes_i]

        panel_interpolant = curve_interpolate(qnodes_i_complex, L_Leg)
        panel_interpolant_deriv = curve_interpolate_derivative(qnodes_i_complex, L_Leg)
    end

    # Loop over target points to compute preimages and subdivisions
    for j in axes(target_complex)[1]
        x_target = target_complex[j]
        local z_approx # Ensure z_approx is in the loop's scope

        if affine_preimage == false
            # Calculate the robust initial guess (affine inverse)
            initial_guess = 2 * (x_target - z_mid) / z_diff

            # Refine the guess using the polynomial interpolant
            newton_nonlinear_fn = newton_fn_generator(panel_interpolant, x_target)
            z_approx = NewtonRaphson(
                newton_nonlinear_fn,
                panel_interpolant_deriv,
                initial_guess;
                tol=1e-14,
            )
        else
            # Default: Use Affine Inverse Directly
            z_approx = 2 * (x_target - z_mid) / z_diff
        end

        # Subdivide the panel based on the preimage
        panel_subdivision_temp_j, ksplit_bool_temp_j =
            create_subdivision(z_approx, panel_arc_length, α, Cε, Rε)

        push!(panel_subdivision, panel_subdivision_temp_j)
        push!(ksplit_bool, ksplit_bool_temp_j)
    end

    return panel_subdivision, ksplit_bool
end

function log_punctured(x, y)
    d = norm(x - y)
    d ≤ Inti.SAME_POINT_TOLERANCE && return 0
    return log(d)
end

"""
    build_neighbor_information(source_el, n_el; tol=1e-12)

Builds connectivity maps for a set of panels (elements).

# Returns
- `panel_to_nodes_map`: A vector where the `i`-th element is a tuple `(start_node_id, end_node_id)` for the `i`-th panel.
- `node_to_panel_map`: A vector of vectors where the `j`-th element is a list of panel indices connected to the `j`-th unique node.
"""
function build_neighbor_information(source_el, n_el; tol=1e-12)
    # Collect all unique nodes and map panel endpoints to unique node IDs
    all_nodes = Vector{typeof(source_el[1](0))}()
    panel_to_nodes_map = Vector{Tuple{Int,Int}}(undef, n_el)

    for i in 1:n_el
        start_node = source_el[i](0)
        end_node = source_el[i](1)

        start_id = findfirst(n -> norm(n - start_node) < tol, all_nodes)
        if isnothing(start_id)
            push!(all_nodes, start_node)
            start_id = length(all_nodes)
        end

        end_id = findfirst(n -> norm(n - end_node) < tol, all_nodes)
        if isnothing(end_id)
            push!(all_nodes, end_node)
            end_id = length(all_nodes)
        end

        panel_to_nodes_map[i] = (start_id, end_id)
    end

    # Build a map from each unique node ID to the panels it connects
    n_unique_nodes = length(all_nodes)
    node_to_panel_map = [Vector{Int}() for _ in 1:n_unique_nodes]

    for i in 1:n_el
        start_id, end_id = panel_to_nodes_map[i]
        push!(node_to_panel_map[start_id], i)
        push!(node_to_panel_map[end_id], i)
    end

    for (node_id, panels) in enumerate(node_to_panel_map)
        if length(panels) != 2
            @warn "Node ID $node_id connects $(length(panels)) panels. Expected 2 for a simple closed curve."
        end
    end

    return panel_to_nodes_map, node_to_panel_map
end

"""
    _get_ksplit_kernels(op, layer_type, T)

Returns a NamedTuple (G_S, G_L, G_C) containing the smooth part,
logarithmic singularity, and Cauchy singularity of the kernel split for a
given operator, layer type, and operator element type `T`. The formula
is given as follows:

G(x, y) = G_S(x, y) + G_L(x, y) log|x - y| + G_C(x, y) frac{(x - y) ⋅ n(y)}{|x - y|^2}

"""
function _get_ksplit_kernels(op, layer_type, T)
    if op isa Inti.Laplace{2}
        if layer_type == :single
            G_S = (x, y) -> 0
            G_L = (x, y) -> -1 / (2π)
            G_C = (x, y) -> 0
        elseif layer_type == :double
            G_S = (x, y) -> 0
            G_L = (x, y) -> 0
            G_C = (x, y) -> 1 / (2π)
        else
            error("Unsupported layer type for Laplace operator")
        end
    elseif op isa Inti.Helmholtz{2,Float64}
        k = op.k
        if layer_type == :single
            G_S =
                (x, y) -> begin
                    x_coor = Inti.coords(x)
                    y_coor = Inti.coords(y)
                    r = norm(x_coor - y_coor)
                    im / 4 * hankelh1_0_smooth(k * r) - 1 / (2π) * besselj0(k * r) * log(k)
                end
            G_L = (x, y) -> begin
                x_coor = Inti.coords(x)
                y_coor = Inti.coords(y)
                r = norm(x_coor - y_coor)
                -1 / (2π) * besselj0(k * r)
            end
            G_C = (x, y) -> 0
        elseif layer_type == :double
            G_S =
                (x, y) -> begin
                    x_coor = Inti.coords(x)
                    y_coor = Inti.coords(y)
                    r = norm(x_coor - y_coor)
                    ny = Inti.normal(y)
                    (r ≤ Inti.SAME_POINT_TOLERANCE) && return zero(T)
                    im * k / 4 *
                    (hankelh1_1_smooth(k * r) + 2 * im / π * besselj1(k * r) * log(k)) *
                    dot(x_coor - y_coor, ny) / r
                end
            G_L = (x, y) -> begin
                x_coor = Inti.coords(x)
                y_coor = Inti.coords(y)
                r = norm(x_coor - y_coor)
                ny = Inti.normal(y)
                (r ≤ Inti.SAME_POINT_TOLERANCE) && return zero(T)
                -k / (2π) * besselj1(k * r) * dot(x_coor - y_coor, ny) / r
            end
            G_C = (x, y) -> 1 / (2π)
        else
            error("Unsupported layer type for Helmholtz operator")
        end
    elseif op isa Inti.Yukawa{2,Float64}
        λ = op.λ
        if layer_type == :single
            G_S =
                (x, y) -> begin
                    x_coor = Inti.coords(x)
                    y_coor = Inti.coords(y)
                    r = norm(x_coor - y_coor)
                    1 / (2π) * besselk0_smooth(λ * r) -
                    1 / (2π) * besseli(0, λ * r) * log(λ)
                end
            G_L = (x, y) -> begin
                x_coor = Inti.coords(x)
                y_coor = Inti.coords(y)
                r = norm(x_coor - y_coor)
                -1 / (2π) * besseli(0, λ * r)
            end
            G_C = (x, y) -> 0
        elseif layer_type == :double
            G_S =
                (x, y) -> begin
                    x_coor = Inti.coords(x)
                    y_coor = Inti.coords(y)
                    r = norm(x_coor - y_coor)
                    ny = Inti.normal(y)
                    (r ≤ Inti.SAME_POINT_TOLERANCE) && return zero(T)
                    λ / (2π) *
                    (besselk1_smooth(λ * r) + besseli(1, λ * r) * log(λ)) *
                    dot(x_coor - y_coor, ny) / r
                end
            G_L = (x, y) -> begin
                x_coor = Inti.coords(x)
                y_coor = Inti.coords(y)
                r = norm(x_coor - y_coor)
                ny = Inti.normal(y)
                (r ≤ Inti.SAME_POINT_TOLERANCE) && return zero(T)
                λ / (2π) * besseli(1, λ * r) * dot(x_coor - y_coor, ny) / r
            end
            G_C = (x, y) -> 1 / (2π)
        else
            error("Unsupported layer type for Yukawa operator")
        end
    else
        error("Operator type not supported")
    end

    return (G_S=G_S, G_L=G_L, G_C=G_C)
end