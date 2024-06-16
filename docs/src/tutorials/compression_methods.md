# Compression methods

```@meta
CurrentModule = Inti
```

Inti.jl wraps several external libraries providing acceleration routines for
integral operators. In general, acceleration routines have the signature
`assemble_*(iop, args...; kwargs...)`, and take an [`IntegralOperator`](@ref) as
a first argument. They return a new object that represents a compressed version
of the operator. The following methods are available:

- [`assemble_matrix`](@ref): create a dense `Matrix` representation of the
  integral operator. Not really a compression method, but useful for debugging
  and small problems.
- [`assemble_hmatrix`](@ref): assemble a hierarchical matrix representation of
  the operator using the
  [`HMatrices`](https://github.com/IntegralEquations/HMatrices.jl) library.
- [`assemble_fmm`](@ref): return a `LinearMap` object that represents the
  operator using the fast multipole method. This method is powered by the
  [`FMM2D`](https://github.com/flatironinstitute/fmm2d/) and
  [`FMM3D`](https://fmm3d.readthedocs.io) libraries, and is only available for
  certain kernels.

!!! warning "Singular kernels"
    Acceleration methods do not correct for singular or nearly-singular
    interactions. When the underlying kernel is singular, a *correction* is
    usually necessary in order to obtain accurate results (see the [section on
    correction methods](@ref "Correction mehods") for more details).
  
To illustrate the use of compression methods, we will use the following problem
as an example. Note that for such a small problem, compression methods are not
likely not necessary, but they are useful for larger problems.

```@example compression
using Inti
using LinearAlgebra
# define the quadrature
geo = Inti.ball()
Ω = Inti.Domain(geo)
Γ = Inti.boundary(Ω)
Q = Inti.Quadrature(Γ; meshsize = 0.4, qorder = 5)
# create the operator
pde = Inti.Helmholtz(; dim = 3, k = 2π)
K = Inti.SingleLayerKernel(pde)
Sop = Inti.IntegralOperator(K, Q, Q)
x = rand(eltype(Sop), length(Q))
rtol = 1e-8
nothing # hide
```

In what follows we compress `Sop` using the different methods available.

## Dense matrix

```@docs; canonical = false
assemble_matrix
```

Typically used for small problems, the dense matrix representation converts the
`IntegralOperator` into a `Matrix` object. The underlying type of the `Matrix`
is determined by the `eltype` of the `IntegralOperator`, and depends on the
inferred type of the kernel. Here is how `assemble_matrix` can be used:

```@example compression
Smat = Inti.assemble_matrix(Sop; threads=true)
@assert Sop * x ≈ Smat * x # hide
er = norm(Sop * x - Smat * x, Inf) / norm(Sop * x, Inf)
println("Forward map error: $er")
```

Since the returned object is plain Julia `Matrix`, it can be used with any of
the linear algebra routines available in Julia (e.g. `\`, `lu`, `qr`, `*`, etc.)

## Hierarchical matrix

```@docs; canonical = false
assemble_hmatrix
```

The hierarchical matrix representation is a compressed representation of the
underlying operator; as such, it takes a tolerance parameter that determines the
relative error of the compression. Here is an example of how to use the
`assemble_hmatrix` method to compress the previous problem:

```@example compression
using HMatrices
Shmat = Inti.assemble_hmatrix(Sop; rtol = 1e-8)
er = norm(Smat * x - Shmat * x, Inf) / norm(Smat * x, Inf)
@assert er < 10*rtol # hide
println("Forward map error: $er")
```

Note that `HMatrices` are said to be *kernel-independent*, meaning that they
efficiently compress a wide range of integral operators provided they satisfy a
certain asymptotically smooth criterion (see e.g. [bebendorf2008hierarchical,
hackbusch2015hierarchical](@cite)).

The `HMatrix` object can be used to solve linear systems, both iteratively
through e.g. GMRES, or directly using an `LU` factorization.

## Fast multipole method

```@docs; canonical = false
assemble_fmm
```

The fast multipole method (FMM) is an acceleration technique based on an
analytic multipole expansion of the kernel in the integral operator
[rokhlin1985rapid, greengard1987fast](@cite). It provides a very
memory-efficient and fast way to evaluate certain types of integral operators.
Here is how `assemble_fmm` can be used:

```@example compression
using FMM3D
Sfmm = Inti.assemble_fmm(Sop; rtol = 1e-8)
er = norm(Sop * x - Sfmm * x, Inf) / norm(Sop * x, Inf)
@assert er < 10*rtol # hide
println("Forward map error: $er")
```

## Tips on choosing a compression method

The choice of compression method depends on the problem at hand, as well as on
the available hardware. Here is a rough guide on how to choose a compression:

1. For small problems (say less than 5k degrees of freedom), use the dense
   matrix representation. It is the simplest and most straightforward method,
   and does not require any additional packages. It is also the most accurate
   since it does not introduce any approximation errors.
2. If the integral operator is supported by the `assemble_fmm`, and if you can
   afford an iterative solver, use it. The FMM is a very efficient method for
   certain types of kernels, and can handle problems with up to a few million
   degrees of freedom on a laptop.
3. If the kernel is not supported by `assemble_fmm`, if iterative solvers are
   not an option, or if you need to solve your system for many right-hand sides,
   use the `assemble_hmatrix` method. It is a very general method that can
   handle a wide range of kernels, and although assembling the `HMatrix` can be
   time and memory consuming (the complexity is still log-linear in the DOFs for
   many kernels of interest, but the constants can be large), the resulting
   `HMatrix` object is very efficient to use. For example, the forward map is
   usually significantly faster than the one obtained through `assemble_fmm`.
