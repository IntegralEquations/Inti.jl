# Boundary integral operators

```@meta
CurrentModule = Inti
```

!!! note "Important points covered in this tutorial"
    - Define layer potentials and the four integral operators of Calderón calculus
    - Block operator construction and composition
    - Volume integral operators

A central piece of integral equation methods is the efficient and accurate
computation of integral operators. In the first part of this tutorial we will
cover how to assemble and manipulate the four integral operators of Calderón
calculus, namely the single-layer, double-layer, hypersingular, and adjoint
operators [nedelec2001acoustic, colton2013integral](@cite), for some predefined
kernels in Inti.jl. In the second part we will show how to extend the
package to handle custom kernels.

## Predefined kernels and integral operators

To simplify the construction of integral operators for some commonly used PDEs,
Inti.jl defines a few [`AbstractPDE`](@ref)s types:

```@example integral_operators
using Inti, StaticArrays, LinearAlgebra
using InteractiveUtils: subtypes # hide
subtypes(Inti.AbstractPDE)
```

For each of these PDEs, the package provides a [`SingleLayerKernel`](@ref),
[`DoubleLayerKernel`](@ref), [`HyperSingularKernel`](@ref), and
[`AdjointDoubleLayerKernel`](@ref) that can be used to construct the corresponding kernel
functions, e.g.:

```@example integral_operators
pde = Inti.Helmholtz(; dim = 2, k = 1.2)
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
`FMM2D` library. The `WrappedMap` accounts for a sparse matrix used to account
for singular and nearly singular interactions. These two objects are added
lazily using [LinearMaps](https://github.com/JuliaLinearAlgebra/LinearMaps.jl).

Effortlessly and efficiently composing operators is a powerful abstraction for
integral equations, as it allows for the construction of complex systems from
simple building blocks. To show this, let us show how one may construct the
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
H = [-Dfmm Sfmm; -Nfmm Kfmm]
C₊ = I / 2 + H
C₋ = I / 2 - H
u = map(q -> cos(q.coords[1] + q.coords[2]), Q)
v = map(q-> q.coords[1], Q)
x = [u; v]
e₊ = norm(C₊*(C₊*x) - C₊*x, Inf)
e₋ = norm(C₋*(C₋*x) - C₋*x, Inf)
println("projection error for C₊: $e₊")
println("projection error for C₋: $e₋")
```
