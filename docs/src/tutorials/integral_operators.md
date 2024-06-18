# Boundary integral operators

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this tutorial"
    - Define layer potentials and the four integral operators of Calderón calculus
    - Construct block operators
    - Set up a custom kernel

A central piece of integral equation methods is the efficient and accurate
computation of integral operators. In the first part of this tutorial we will
cover how to assemble and manipulate the four integral operators of Calderón
calculus, namely the single-layer, double-layer, hypersingular, and adjoint
operators [nedelec2001acoustic, colton2013integral](@cite), for some predefined
kernels in Inti.jl. In the second part we will show how to extend the package to
handle custom kernels.

## Predefined kernels and integral operators

To simplify the construction of integral operators for some commonly used PDEs,
Inti.jl defines a few [`AbstractPDE`](@ref)s types. For each of these PDEs, the
package provides a [`SingleLayerKernel`](@ref), [`DoubleLayerKernel`](@ref),
[`HyperSingularKernel`](@ref), and [`AdjointDoubleLayerKernel`](@ref) that can
be used to construct the corresponding kernel functions, e.g.:

```@example integral_operators
using Inti, StaticArrays, LinearAlgebra
pde = Inti.Helmholtz(; dim = 2, k = 2π)
G   = Inti.SingleLayerKernel(pde)
```

Typically, we are not interested in the kernels themselves, but in the integral
operators they define. Two functions, [`single_double_layer`](@ref) and
[`adj_double_layer_hypersingular`](@ref), are provided as a high-level syntax to
construct the four integral operators of Calderón calculus:

```@example integral_operators
Γ = Inti.parametric_curve(s -> SVector(cos(s), sin(s)), 0, 2π) |> Inti.Domain
Q = Inti.Quadrature(Γ; meshsize = 0.1, qorder = 5)
S, D = Inti.single_double_layer(; 
    pde, 
    target = Q, 
    source = Q, 
    compression = (method = :none,), 
    correction = (method = :dim,)
)
K, N = Inti.adj_double_layer_hypersingular(; 
    pde, 
    target = Q, 
    source = Q, 
    compression = (method = :none,), 
    correction = (method = :dim,)
)
nothing # hide
```

Much goes on under the hood in the function above, and the sections on
[correction](@ref "Correction methods") and [compression](@ref "Compression
methods") methods will provide more details on the options available. The
important thing to keep in mind is that `S`, `D`, `K`, and `H` are discrete
approximations of the following (linear) operators:

```math
\begin{aligned}
    S[\sigma](\boldsymbol{x}) &:= \int_{\Gamma} G(\boldsymbol{x}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \mathrm{d} s_{\boldsymbol{y}}, \quad 
    &&D[\sigma](\boldsymbol{x}) := \mathrm{p.v.} \int_{\Gamma} \frac{\partial G}{\partial \nu_{\boldsymbol{y}}}(\boldsymbol{x}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \mathrm{d} s_{\boldsymbol{y}} \\
    D'[\sigma](\boldsymbol{x}) &:=  \mathrm{p.v.} \int_{\Gamma} \frac{\partial G}{\partial \nu_{\boldsymbol{x}}}(\boldsymbol{x}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \mathrm{d} s_{\boldsymbol{y}}, \quad
    &&N[\sigma](\boldsymbol{x}) := \mathrm{f.p.} \int_{\Gamma} \frac{\partial^2 G}{\partial \nu_{\boldsymbol{x}} \partial \nu_{\boldsymbol{y}}}(\boldsymbol{x}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \mathrm{d} s_{\boldsymbol{y}}
\end{aligned}
```

The actual type of `S`, `D`, `K`, and `H` depends on the `compression` and
`correction` methods. In the simple case above, these are simply matrices:

```@example integral_operators
@assert all(T -> T == Matrix{ComplexF64}, map(typeof, (S, D, K, N))) # hide
map(typeof, (S, D, K, N))
```

If we turn on a compression method, such as `:fmm`, the types may change into
something different:

```@example integral_operators
using FMM2D # will load the extension
Sfmm, Dfmm = Inti.single_double_layer(; 
    pde, 
    target = Q, 
    source = Q, 
    compression = (method = :fmm, tol = 1e-10), 
    correction = (method = :dim, )
)
Kfmm, Nfmm = Inti.adj_double_layer_hypersingular(; 
    pde, 
    target = Q, 
    source = Q, 
    compression = (method = :fmm, tol = 1e-10), 
    correction = (method = :dim,)
)
typeof(Sfmm)
```

This is because the FMM method is used to approximate the matrix-vector in a
matrix-free way: the only thing *guaranteed* is that `S` and `D` can be applied
to a vector:

```@example integral_operators
x = map(q -> cos(q.coords[1] + q.coords[2]), Q)
@assert norm(Sfmm*x - S*x, Inf) / norm(S*x, Inf) < 1e-8 # hide
norm(Sfmm*x - S*x, Inf)
```

The `Sfmm` object above in fact combines two linear maps:

```@example integral_operators
Sfmm
```

The `FunctionMap` computes a matrix-vector by performing a function call to the
`FMM2D` library. The `WrappedMap` accounts for a sparse matrix used to correct
for singular and nearly singular interactions. These two objects are added
lazily using [LinearMaps](https://github.com/JuliaLinearAlgebra/LinearMaps.jl).

## Operator composition

Effortlessly and efficiently composing operators is a powerful abstraction for
integral equations, as it allows for the construction of complex systems from
simple building blocks. To show this, let us show how to construct the
Calderón projectors:

```math
\begin{aligned}
H = \begin{bmatrix}
    -D & S \\
    -N & D'
\end{bmatrix} 
\end{aligned}
```

As is well-known [nedelec2001acoustic; Theorem 3.1.3](@cite), the operators
``C_\pm = I/2 \pm H`` are the projectors (i.e. ``C_{\pm}^2 = C_{\pm}``):

```@example integral_operators
using LinearMaps
# create the block operator
H = [-Dfmm Sfmm; -Nfmm Kfmm]
C₊ = I / 2 + H
C₋ = I / 2 - H
# define two density functions on Γ
u = map(q -> cos(q.coords[1] + q.coords[2]), Q)
v = map(q-> q.coords[1], Q)
x = [u; v]
# compute the error in the projector identity
e₊ = norm(C₊*(C₊*x) - C₊*x, Inf)
e₋ = norm(C₋*(C₋*x) - C₋*x, Inf)
@assert e₊ < 1e-5 && e₋ < 1e-5 # hide
println("projection error for C₊: $e₊")
println("projection error for C₋: $e₋")
```

We see that the error in the projector identity is small, as expected. Note that
such compositions are not limited to the Calderón projectors, and can be used
e.g. to construct the combined field integral equation (CFIE), or to compose a
formulation with an operator preconditioner.

## Custom kernels

So far we have focused on problems for which Inti.jl provides predefined
kernels, and used the high-level syntax of e.g. `single_double_layer` to
construct the integral operators. We will now dig into the details of how to set
up your own kernel function, and how to build an integral operator from it.

!!! note "Integral operators coming from PDEs"
    If your integral operator arises from a PDE, it is recommended to define a
    new [`AbstractPDE`](@ref) type, and implement the required methods for
    [`SingleLayerKernel`](@ref), [`DoubleLayerKernel`](@ref),
    [`AdjointDoubleLayerKernel`](@ref), and [`HyperSingularKernel`](@ref). This
    will enable the use of the high-level syntax for constructing boundary
    integral operators, as well as the use of the compression and correction
    methods specific to integral operators arising from PDEs.

For the sake of simplicity, let us consider the following kernel representing
the half-space Dirichlet Green function for Helmholtz's equation in 2D:

```math
    G_D(\boldsymbol{x}, \boldsymbol{y}) = \frac{i}{4} H^{(1)}_0(k |\boldsymbol{x} - \boldsymbol{y}|) - \frac{i}{4} H^{(1)}_0(k |\boldsymbol{x} - \boldsymbol{y}^*|),
```

where ``\boldsymbol{y}^* = (y_1, -y_2)``. We can define this kernel as a

```@example integral_operators
using SpecialFunctions # for hankelh1
function helmholtz_kernel(target, source, k)
    x, y  = Inti.coords(target), Inti.coords(source)
    yc = SVector(y[1], -y[2])
    d, dc  = norm(x-y), norm(x-yc)
    # the singularity at x = y needs to be handled separately, so just put a zero
    d == 0 ? zero(ComplexF64) : im / 4 * ( hankelh1(0, k * d) - hankelh1(0, k * dc))
end
```

Let us now consider the integral operator ``S`` defined by:

```math
    S[\sigma](\boldsymbol{x}) = \int_{\Gamma} G_D(\boldsymbol{x}, \boldsymbol{y}) \sigma(\boldsymbol{y}) \mathrm{d} s_{\boldsymbol{y}}, \quad \boldsymbol{x} \in \Gamma.
```

We can represent `S` by an `IntegralOperator` type:

```@example integral_operators
k = 50π
λ = 2π/k
meshsize = λ / 10
geo = Inti.parametric_curve(s -> SVector(cos(s), 2 + sin(s)), 0, 2π)
Γ = Inti.Domain(geo)
msh = Inti.meshgen(Γ; meshsize)
Q = Inti.Quadrature(msh; qorder = 5)
# create a local scope to capture `k`
K = let k = k
    (t,q) -> helmholtz_kernel(t,q,k)
end
Sop = Inti.IntegralOperator(K, Q, Q)
```

!!! note "Signature of custom kernels"
    Kernel functions passed to `IntegralOperator` should always take two
    arguments, `target` and `source`, which are both of
    [`QuadratureNode`](@ref). This allows for extracting not only the
    [`coords`](@ref) of the nodes, but also the [`normal`](@ref) vector if
    needed (e.g. for double-layer or hypersingular kernels).

The approximation of `Sop` now involves two steps:

- build a dense operator `S₀` that efficiently computes the matrix-vector
  product `Sop * x` for any vector `x`
- correct for the inaccuracies of `S₀` due to singular/nearly-singular
  interactions by adding to it a correction matrix `δS`

For the first step, we will use a hierarchical matrix:

```@example integral_operators
using HMatrices
S₀ = Inti.assemble_hmatrix(Sop; rtol = 1e-4)
```

The correction matrix `δS` will be constructed using [`adaptive_correction`](@ref):

```@example integral_operators
δS = Inti.adaptive_correction(Sop; tol = 1e-4, maxdist = 5*meshsize)
```

How exactly you add `S₀` and `δS` to get the final operator depends on the usage
that you have in mind. For instance, you can use the `LinearMap` type to simply
add them lazily:

```@example integral_operators
using LinearMaps
S = LinearMap(S₀) + LinearMap(δS)
```

You can add `δS` to `S₀` to create a new object:

```@example integral_operators
S = S₀ + δS
```

or if performance/memory is a concern, you may want to directly add `δS` to `S₀` in-place:

```@example integral_operators
axpy!(1.0, δS, S₀)
```

All of these should give an identical matrix-vector product, but the later two
allow e.g. for the use of direct solvers though an LU factorization.

!!! warning "Limitations"
    Integral operators defined from custom kernel functions do not support all
    the features of the predefined ones. In particular, some singular
    integration methods (e.g. the Density Interpolation Method) and acceleration
    routines (e.g. Fast Multipole Method) used to correct for singular and
    nearly singular integral operators, and to accelerate the matrix vector
    products, are only available for specific kernels. Check the
    [corrections](@ref "Correction methods") and [compression](@ref "Compression
    methods") for more details concerning which methods are compatible with
    custom kernels.
