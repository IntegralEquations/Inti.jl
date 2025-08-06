# Linear solvers

```@meta
CurrentModule = Inti
```

!!! warning "Work in progress"
    This tutorial is still a work in progress. We will update it with more
    details and examples in the future.

Inti.jl does not provide its own linear solvers, but relies on external
libraries such as
[IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)
or the [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
standard library for the solving the linear systems that arise in the
discretization of integral equations.
