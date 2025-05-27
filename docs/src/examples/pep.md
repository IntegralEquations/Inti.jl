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

!!! details "Motivation"
    Plasmonic resonances play a crucial role in nanophotonics, metamaterials, and sensing
    applications. At these resonances, the electromagnetic field becomes highly concentrated
    near material interfaces, leading to enhanced optical effects like extraordinary
    transmission, surface-enhanced Raman scattering, and localized heating. This tutorial
    demonstrates how to compute these resonances (in a much simplified context!) by solving the
    plasmonic eigenvalue problem using boundary integral methods.

Let ``\Omega \subset \RR^2`` be a bounded domain with boundary ``\Gamma := \partial\Omega``.
In this tutorial we are interested in finding a non-zero function ``u : \RR^2 \to \RR`` and a
scalar ``\kappa \in (-\infty, 0)`` solving the following eigenvalue problem:

```math
\begin{aligned}
    \nabla \cdot \left(a(\bx) \nabla u \right) &= 0 \quad, \quad a(\bx) = \begin{cases}
        1 & \text{if } \bx \in \Omega \\
        \kappa & \text{if } \bx \in \Omega^c
    \end{cases}.
\end{aligned}
```

Since the coefficient $a(\boldsymbol{x})$ is piecewise constant, we can reformulate the
problem in terms of the jump conditions across the boundary $\Gamma$. More precisely,
denoting by $\Omega^\pm$ the exterior and interior of $\Omega$, respectively, and ``u^\pm``
the restriction of ``u`` to ``\Omega^\pm``, we can rewrite the problem as:

```math
\begin{aligned}
\Delta u^\pm &= 0 \quad &&\text{in } \Omega^\pm, \\
u^+ &= u^- \quad &&\text{on } \Gamma \\
\kappa \partial_\nu u^{+} &= \partial_\nu u^{-} \quad &&\text{on } \Gamma \\
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
$\Omega^+$ and $\Omega^-$ by construction, so we only need to enforce the jump conditiosn on
the boundary $\Gamma$. Using some properties of the single-layer potential (in particular,
its continuity across the boundary and the jump in the normal derivative), we can derive the
following boundary integral equation:

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

The spectral parameter $\lambda$ is related to the original parameter $\kappa$ by the
transformation:

```math
\kappa = \frac{2 \lambda + 1}{2 \lambda - 1}
```

This is called the Neumann-Poincaré eigenvalue problem (NPEP), and is precisely the problem
we will solve numerically in this tutorial.

!!! note "Functional spaces and conditions at infinty"
    To keep the discussion simple, we have chosen to avoid function spaces and the
    appropriate decay conditions at infinity, but this can all be made rigorous.

## Numerical Implementation

Now, let's implement a numerical solution to this problem. We'll create a function that:

1. Takes a curve $\Gamma$ and discretization parameters
2. Assembles a matrix representation of the $K^{\star}$ operator
3. Computes its eigenvalues and eigenfunctions
4. Returns the results as eigenvalue-eigenfunction pairs

Because later we will also consider periodic structures, we will allow the user to specify a
period for the Green's function. If no period is specified, we will use the standard Green's
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
accuracy of our implementation. The eigenvalue at `\lambda = -1/2` corresponds to constant
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
Green's function is used. The periodic Green's function for the Laplace equation in 2D is given by:

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
n = length(λᵢ) - 2 # Choose an eigenvalue to visualize
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

This plot shows the eigenfunction for a periodic array of kite-shaped inclusions. Notice how
the solution repeats with period $\ell$ in the x-direction. The dashed lines indicate the
boundaries of the unit cell.

## Advanced Topics

### Domains with Corners

For domains with corners, the eigenfunctions can exhibit singular behavior near the corners.
To accurately capture this behavior, one would typically need to:

1. Use adaptive mesh refinement near corners
2. Employ special quadrature methods that can handle the singularities
3. Consider using a graded mesh that places more points near corners

Inti provides tools for handling these situations, though they require more careful setup than the smooth domains we've considered so far.

### Multiple Inclusions

The approach can be extended to handle multiple inclusions (disconnected domains). In this case, the boundary $\Gamma$ would consist of multiple closed curves, and the quadrature would need to be defined on each component.

### Quasi-periodic and Helmholtz problems

For quasi-periodic problems or those involving Helmholtz equations, the approach remains
similar, but the Green's function and boundary conditions may change. Integral operators
operator can still be used, but care must be taken to ensure the correct form of the Green's
function is employed, and the boundary conditions are properly defined.

## Conclusion

In this tutorial, we've demonstrated how to:

1. Formulate the plasmonic eigenvalue problem in terms of boundary integral equations
2. Implement a numerical solver using the Neumann-Poincaré operator
3. Compute and visualize eigenfunctions for different geometries
4. Extend the approach to periodic structures

The boundary integral approach offers good accuracy and efficiency, especially for problems
with smooth boundaries. It naturally handles the unbounded domain and radiation conditions,
making it ideal for scattering and resonance problems.
