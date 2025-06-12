# Plasmonic Eigenvalue Problem

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this example"
      - Reformulate a transmission eigenvalue problem using integral equations
      - Implement and solve the Neumann-Poincaré eigenvalue problem
      - Use a periodic Green's function to solve problems with periodic structures
      - Visualize plasmonic eigenfunctions for different geometries

## Problem Definition  

Plasmonic resonances play a crucial role in nanophotonics, metamaterials, and sensing
applications. At these resonances, the electromagnetic field becomes highly concentrated
near material interfaces, leading to enhanced optical effects like extraordinary
transmission, surface-enhanced Raman scattering, and localized heating. This tutorial
demonstrates how to compute these resonances (in a much simplified context!) by solving the
plasmonic eigenvalue problem using boundary integral methods. To keep things simple, we
provide a brief and non-rigorous overview of the plasmonic eigenvalue problem; see e.g.
[grieser2014plasmonic](@cite) for a more detailed mathematical discussion and
e.g. [maier2007plasmonics](@cite) for a detailed discussion on plasmonic resonances and
their physical relevance.

In what follows we let ``\Omega \subset \RR^2`` be a bounded domain with smooth (e.g.
``C^2``) boundary
``\Gamma := \partial\Omega``. We are interested in finding a non-zero
function ``u : \RR^2 \to \RR`` and a scalar ``\kappa \in (-\infty, 0)`` solving the
following eigenvalue problem:

```math
\begin{aligned}
    \nabla \cdot \left(a(\bx) \nabla u \right) &= 0, \quad a(\bx) = \begin{cases}
        1 & \text{if } \bx \in \Omega \\
        \kappa & \text{if } \bx \in \Omega^c
    \end{cases}, \quad u(\bx) \underset{|\bx|\to \infty}{=} \mathcal{O}(|\bx|^{-1}).
    
\end{aligned}
```

Since the coefficient $a(\boldsymbol{x})$ is piecewise constant, we can reformulate the
problem in terms of the jump conditions across the boundary ``\Gamma``. More precisely,
denoting by $\Omega^\pm$ the exterior and interior of $\Omega$, respectively, and ``u^\pm``
the restriction of ``u`` to ``\Omega^\pm``, we can rewrite the problem as:

```math
\begin{aligned}
\Delta u^\pm &= 0 \quad &&\text{in } \Omega^\pm, \\
u^+ &= u^- \quad &&\text{on } \Gamma,\\
\kappa \partial_\nu u^{+} &= \partial_\nu u^{-} \quad &&\text{on } \Gamma, \\
\end{aligned}
```

where ``\partial_\nu`` denotes the normal derivative with respect to the outward normal on the boundary $\Gamma$.

To solve this problem efficiently, we reformulate it as an integral equation using a
single-layer potential ansatz:

```math
\begin{aligned}
 u(\boldsymbol{x}) = \int_{\Gamma} G(\boldsymbol{x},\boldsymbol{y}) \sigma(\boldsymbol{y}) \, \textup{d} s({\boldsymbol{y}})
\end{aligned}
```

where:

- ``G(\boldsymbol{x},\boldsymbol{y})`` is the Green's function for the Laplace equation
- ``\sigma(\boldsymbol{y})`` is an unknown density function defined on ``\Gamma``

Note that this ansatz automatically satisfies Laplace's equation ($\Delta u = 0$) in both
$\Omega^+$ and $\Omega^-$ by construction, so we only need to enforce the jump conditions on
the boundary $\Gamma$ and possibly at ``\infty``. Using some properties of the single-layer
potential (in particular, its continuity across the boundary and the jump in the normal
derivative), we can derive the following boundary integral equation:

```math
\begin{aligned}
    \left( K^{\star} \sigma \right)(\boldsymbol{x}) = \lambda \sigma(\boldsymbol{x}) \quad &&\text{for } \boldsymbol{x} \in \Gamma
\end{aligned}
```

where ``K^{\star}`` is the adjoint single-layer operator defined as:

```math
\begin{aligned}
    K^{\star} \sigma(\boldsymbol{x}) = \int_{\Gamma} \partial_{\nu({\bx})} G(\boldsymbol{x},\boldsymbol{y})  \sigma(\boldsymbol{y}) \, \textup{d} s({\boldsymbol{y}})
\end{aligned}
```

For the condition at infinity, it can be shown that the single-layer representation
satisfies the decay condition at infinity if the density $\sigma$ has zero mean over the
boundary $\Gamma$, which is the case if $\lambda \neq -1/2$ [faria2024complex; Lemma
29](@cite). We thus have an equivalence between the original plasmonic eigenvalue problem
and the Neumann-Poincaré eigenvalue problem (PEP), where $\lambda$ is related to the
original parameter ``\kappa`` by the transformation:

```math
\kappa = \frac{2 \lambda + 1}{2 \lambda - 1}
```

Next, we focus on the numerical discretization of the Neumann-Poincaré operator $K^{\star}$
using `Inti`'s boundary integral methods.

!!! note "Rigorous formulation"
    To keep the discussion simple, we have chosen to avoid the details of the appropriate
    function spaces and precise regularity conditions. A rigorous treatment is beyond the
    scope of this tutorial, but details can be found in the literature. Note that the
    two-dimensional case is somewhat special [grieser2014plasmonic; ``\S 2.4``](@cite).

## Numerical Implementation

Now, let's implement a numerical solution to this problem. We'll create a function that:

1. Takes a curve $\Gamma$ and discretization parameters
2. Assembles a matrix representation of the $K^{\star}$ operator
3. Computes its eigenvalues and eigenfunctions
4. Returns the results as eigenvalue-eigenfunction pairs

Because later we will also consider periodic structures, we will allow the user to specify a
period for the Green's function. If no period is specified, we will use the free-space Green's
function.

```@example NPEP
using Inti
using StaticArrays
using LinearAlgebra
using GLMakie

function npep(Γ; meshsize, qorder, period = Inf)
    # Step 1: Discretize the curve Γ with a composite quadrature rule
    Q = Inti.Quadrature(Γ; meshsize, qorder)
    
    # Step 2: Choose the appropriate Green's function (periodic or standard)
    op = if isfinite(period)
        Inti.LaplacePeriodic1D(; dim=2, period) # Periodic Green's function
    else
        Inti.Laplace(; dim=2)                   # Standard Green's function
    end
    
    # Step 3: Create the Neumann-Poincaré integral operator
    kernel = Inti.AdjointDoubleLayerKernel(op)
    Kop    = Inti.IntegralOperator(kernel, Q, Q)
    
    # Step 4: Assemble the matrix and correct for singular/nearly-singular entries
    K₀ = Inti.assemble_matrix(Kop)         # Basic assembly
    δK = Inti.adaptive_correction(Kop)     # Correction for singular integrals
    K  = K₀ + δK                           # Final operator matrix
    
    # Step 5: Compute eigendecomposition
    F  = eigen(K)
    λᵢ = F.values # Eigenvalues
    
    # Step 6: Construct eigenfunction evaluators from eigenvectors
    uᵢ = map(eachcol(F.vectors)) do v
        # Create a function that evaluates the single-layer potential with density v
        return Inti.SingleLayerPotential(op, Q)[real(v) / norm(v, Inf)]
    end
    
    return λᵢ, uᵢ
end
```

Let's break down what's happening here:

1. We first create a quadrature rule on our curve $\Gamma$ with specified mesh size and quadrature order
2. We select the appropriate Green's function (standard or periodic)
3. We create the adjoint double-layer kernel and the corresponding integral operator
4. We assemble a matrix representation of the operator and apply necessary corrections for singular integrals
5. We compute the eigendecomposition of this matrix
6. For each eigenvector, we create a function that evaluates the corresponding eigenfunction
   at any point

Next, we can test our implementation on a simple case of an ellipse, where we know the
analytical eigenvalues and eigenfunctions. This will allow us to validate our numerical results.

## Validation: Elliptical Domain

For an elliptical domain, we can compare our numerical results with the analytical solution. The eigenvalues of the Neumann-Poincaré operator for an ellipse are known explicitly.

```@example NPEP
# Define an ellipse with semi-axes 2.5 and 1
χ = (s) -> SVector(2.5 * cos(s), sin(s))
Γ = Inti.parametric_curve(χ, 0, 2π) |> Inti.Domain

# Compute exact eigenvalues
r = 1 / 2.5  # Ratio of semi-minor to semi-major axis
nmax = 10
λ = [sign(n) / 2 * exp(-2 * abs(n) * atanh(r)) for n in -nmax:nmax]

# Compute numerical eigenvalues and eigenfunctions
λᵢ, uᵢ = npep(Γ; meshsize = 0.1, qorder = 4)

# Visualize the eigenvalues
fig = Figure(size = (600, 250))
ax = Axis(
    fig[1, 1];
    title = "Eigenvalues of the Neumann-Poincaré Operator for an Ellipse",
    xlabel = "Re(λ)",
    ylabel = "Im(λ)",
    limits = (-0.5, 0.5, -0.1, 0.1),
)
scatter!(
    ax,
    real(λ),
    imag(λ);
    label = "Analytical",
    color = :gray,
    markersize = 20,
    marker = :rect,
    alpha = 0.6,
)
scatter!(
    ax,
    real(λᵢ),
    imag(λᵢ);
    label = "Numerical",
    color = :blue,
    markersize = 14,
    marker = :cross,
)
axislegend(ax)
fig # hide
```

The excellent agreement between the analytical and numerical eigenvalues confirms the
accuracy of our implementation. The eigenvalue at ``\lambda = -1/2`` corresponds to constant
functions, which are not plasmonic modes since they do not satisfy the decay condition at
infinity, but are still part of the spectrum of the Neumann-Poincaré operator.

Let's visualize an eigenfunction to better understand the physical nature of these
resonances. Each eigenfunction corresponds to a specific plasmonic mode.

```@example NPEP
fig = Figure(size = (500, 400))
n = 8 # Choose which eigenvalue/eigenfunction to visualize
ax = Axis(
    fig[1, 1];
    title = "Eigenfunction with λ ≈ $(trunc(real(λᵢ[n]), sigdigits = 2))",
    xlabel = "x",
    ylabel = "y",
    aspect = DataAspect(),
)

# Define a function to evaluate the eigenfunction on a grid
fun = (x, y) -> real(uᵢ[n](SVector(x, y)))

# Create a grid for visualization
l = 4
xx = yy = range(-l, l, 100)

# Plot the eigenfunction as a heatmap
hm = heatmap!(ax, xx, yy, fun; colormap = :viridis, interpolate = true)

# Draw the boundary curve
s = range(0, 2π, 100)
lines!(ax, getindex.(χ.(s), 1), getindex.(χ.(s), 2); color = :black, linewidth = 2, label = "Γ")

# Add a colorbar
Colorbar(fig[1, 2], hm)

fig # hide
```

As can be seen in the plot, the eigenfunctions are localized near the boundary, which is a
typical feature of plasmonic modes. Changing the eigenvalue index `n` will show different modes.

## Exploring Different Geometries

Having validated our implementation on a simple elliptical domain, we can now explore more
complex geometries. The boundary integral approach is particularly powerful because it
allows us to complex domains without needing to mesh the interior/exterior. In the next
example we consider a kite-shaped domain.

```@example NPEP
# Define a kite-shaped curve
χ = (s) -> SVector(cos(s[1]) + 0.65 * cos(2 * s[1]) - 0.65, 1.5 * sin(s[1]))
Γ = Inti.parametric_curve(χ, 0, 2π) |> Inti.Domain

# Compute eigenvalues and eigenfunctions
λᵢ, uᵢ = npep(Γ; meshsize = 0.1, qorder = 4)

# Visualize an eigenfunction
fig = Figure(size = (500, 400))
n = 8 # Choose which eigenfunction to visualize
ax = Axis(
    fig[1, 1];
    title = "Eigenfunction with λ ≈ $(trunc(real(λᵢ[n]), sigdigits = 2))",
    xlabel = "x",
    ylabel = "y",
    aspect = DataAspect(),
)

# Define a function to evaluate the eigenfunction on a grid
fun = (x, y) -> real(uᵢ[n](SVector(x, y)))

# Create a grid for visualization
l = 4
xx = yy = range(-l, l, 100)

# Plot the eigenfunction as a heatmap
hm = heatmap!(ax, xx, yy, fun; colormap = :viridis, interpolate = true)

# Draw the boundary curve
s = range(0, 2π, 100)
lines!(ax, getindex.(χ.(s), 1), getindex.(χ.(s), 2); color = :black, linewidth = 2, label = "Γ")

# Add a colorbar
Colorbar(fig[1, 2], hm)

fig
```

Notice how the eigenfunction adapts to the geometry of the domain, and as before we observe
a strong localization of the field near the interface $\Gamma$.

## Periodic Structures

Many applications in nanophotonics involve periodic structures, such as diffraction gratings
or metamaterials. We can extend our approach to handle periodic problems by using a periodic
Green's function. In the periodic case, the problem is posed on $[-\ell/2, \ell/2] \times
\mathbb{R}$ instead of $\mathbb{R}^2$, where $\ell$ is the period, and $u$ must satisfy
periodic boundary conditions in the first coordinate.

Almost everything we have done so far can be adapted to this case, provided a periodic
Green's function is used. The periodic Green's function for the Laplace equation in 2D is
given by (see [these lecture notes](https://people.math.ethz.ch/~grsam/HS17/MaCMiPaP/Lecture%20Notes/Lecture%204.pdf)):

```math
G_p(\bx, \by) = \frac{-1}{4\pi} \log\left(\sin^2\left( \frac{\pi(x_1 - y_1)}{\ell}\right) + \sinh^2\left( \frac{\pi(x_2 - y_2)}{\ell}\right)\right)
```

where $\ell$ is the period in the x-direction.

Our implementation already supports this case—we just need to specify a finite period:

```@example NPEP
# Specify the period
period = 4

# Compute eigenvalues and eigenfunctions for the periodic problem
λᵢ, uᵢ = npep(Γ; meshsize = 0.1, qorder = 4, period)

# Visualize an eigenfunction
fig = Figure(size = (500, 400))
n = 7 # Choose an eigenvalue to visualize
ax = Axis(
    fig[1, 1];
    title = "Periodic Eigenfunction with λ ≈ $(trunc(real(λᵢ[n]), sigdigits = 2))",
    xlabel = "x",
    ylabel = "y",
    aspect = DataAspect(),
)

# Define a function to evaluate the eigenfunction on a grid
fun = (x, y) -> real(uᵢ[n](SVector(x, y)))

# Create a larger grid to show periodicity
l = 1.5*period
xx = yy = range(-l, l, 100)

# Plot the eigenfunction as a heatmap
hm = heatmap!(ax, xx, yy, fun; colormap = :viridis, interpolate = true)

# Draw the boundary curves for the central and neighboring cells
s = range(0, 2π, 100)
lines!(ax, getindex.(χ.(s), 1), getindex.(χ.(s), 2); color = :black, linewidth = 2)
lines!(ax, getindex.(χ.(s), 1) .+ period, getindex.(χ.(s), 2); color = :black, linewidth = 2)
lines!(ax, getindex.(χ.(s), 1) .- period, getindex.(χ.(s), 2); color = :black, linewidth = 2)

# Draw the cell boundaries
vlines!(ax, [-period/2, period/2], color = :black, linewidth = 2, linestyle = :dash)

# Add a colorbar
Colorbar(fig[1, 2], hm)

fig
```

This plot shows the eigenfunction for a periodic array of kite-shaped inclusions (only three
cells are shown). Notice how the solution repeats with period $\ell$ in the x-direction. The
dashed lines indicate the boundaries of the unit cell.

## Three-dimensional Problems

Plasmonic eigenvalue problems are not limited to two dimensions. In three dimensions, many
of the physical and mathematical principles are similar, but the computational complexity
increase. The boundary integral approach remains highly effective, as it avoids volumetric
meshing and naturally incorporates the radiation condition at infinity.

The following example demonstrates how to compute and visualize plasmonic eigenmodes for a
toroidal inclusion. Unlike the two-dimensional case, we avoid assembling the full (dense)
matrix, and use instead a fast multipole method (FMM) to compute the action of the operator
on a vector. We then use a Krylov based eigensolver to compute a few of the eigenvalues,
instead of the full eigendecomposition. This is particularly important as the size of the
problem grows.

```@example NPEP
using FMM3D, KrylovKit
Ω = Inti.torus() |> Inti.Domain
Γ = Inti.boundary(Ω)
Q = Inti.Quadrature(Γ; meshsize = 0.1, qorder = 4)
op = Inti.Laplace(; dim = 3)
kernel = Inti.AdjointDoubleLayerKernel(op)
Kop = Inti.IntegralOperator(kernel, Q, Q)
K₀ = Inti.assemble_fmm(Kop; rtol = 1e-4)
δK = Inti.adaptive_correction(Kop) # Correction for singular integrals
K = K₀ + δK # Final operator matrix
λᵢ, vᵢ, info = eigsolve(K, rand(size(K, 1)), 10)
@assert norm(imag(λᵢ), Inf) < 1e-6 # hide
@assert all(real(λᵢ) .> -0.6) # hide
info
```

Notice that only the matrix-vector product is required by `eigsolve`, and the `info` object
above displays the convergence information. Here is what the few computed eigenvalues look
like:

```@example NPEP
scatter(
    real(λᵢ),
    imag(λᵢ);
    markersize = 10,
    marker = :cross,
    color = :blue,
    label = "Eigenvalues",
)
```

Finally, we can visualize one of the eigenfunctions by evaluating our single-layer ansatz on
a points inside the volume. Since there are many target points, we will again use the fast
multipole method to compute the action of the operator on the eigenfunction.
The visualization will be done on a 3D volume slices using `Makie`.

```@example NPEP
using Meshes # to visualize the mesh using `viz!`
vₙ = vᵢ[8]
pts_per_dim = 100
xx = yy = zz = range(-2,2,pts_per_dim)
targets = [SVector(x, y, z) for x in xx, y in yy, z in zz] |> vec
Kpot = Inti.IntegralOperator(Inti.SingleLayerKernel(op), targets, Q)
Kpot_fmm = Inti.assemble_fmm(Kpot; rtol = 1e-4)
uₙ = Kpot_fmm * real(vₙ)
fig = Figure()
ax = Axis3(fig[1, 1]; aspect = :data, elevation = π/6, azimuth = π/3,
           title = "Eigenfunction with λ ≈ $(trunc(real(λᵢ[n]), sigdigits = 2))")
hidedecorations!(ax)
plt = volumeslices!(ax, xx, yy, zz, reshape(uₙ,pts_per_dim,pts_per_dim,pts_per_dim); interpolate = true)
plt[:update_yz][](pts_per_dim ÷ 2)
plt[:update_xz][](pts_per_dim ÷ 2)
plt[:update_xy][](length(zz) ÷ 2)
viz!(Inti.mesh(Q); showsegments = true, color = :lightgray, alpha = 0.5)
current_figure() # hide
```

As before, the eigenfunction is localized near the boundary of the toroidal inclusion, which
is a characteristic feature of plasmonic modes.

## Further generalizations

Some interesting generalizations are described next. If you are interested in any of these,
feel free to open a draft PR to discuss the implementation details!

### Multiple inclusions

The approach can be extended to handle multiple inclusions (disconnected domains). In this
case, the boundary $\Gamma$ would consist of multiple closed curves, and the quadrature
would need to be defined on each component. This presents no fundamental challenges, and is
simply a matter of defining a more complex domain. As long as the inclusions are smooth,
everything should work as expected.

### Helmholtz equation

The eigenvalue problem can be stated for the Helmholtz equation as well, where a different
wavenumber is used in the exterior and interior domains (and their dependency on the
spectral parameter must be specified through a model). Reformulating the problem in terms of
boundary integral equations is still possible, but becomes more involved. 

Furthermore, when the domain is composed of periodic structures, the solution ``u`` is
usually quasi-periodic, and the computation of quasi-periodic Green's functions requires
more involved techniques.

### Domains with Corners

For domains with corners, the solutions can exhibit singular behavior near the corners, and
the Neumann-Poincaré operator loses its compactness, introducing a continuous spectrum.
Although there are ways to handle this situation, they all require a somewhat intricate
analysis of the corners. See [this
repository](https://github.com/fmonteghetti/neumann-poincare-complex-scaling) for one
possible method, based on complex scaling, implemented using `Inti`.

## Conclusion

In this tutorial, we've demonstrated how to:

1. Formulate the plasmonic eigenvalue problem in terms of boundary integral equations
2. Implement a numerical solver using the Neumann-Poincaré operator
3. Compute and visualize eigenfunctions for different geometries
4. Extend the approach to periodic structures

The boundary integral approach offers good accuracy and efficiency, especially for problems
with smooth boundaries. It naturally handles the unbounded domain and radiation conditions,
making it ideal for scattering and resonance problems.
