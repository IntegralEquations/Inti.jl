# Compression methods

```@meta
CurrentModule = Inti
```

Inti.jl wraps several external libraries providing acceleration routines for
integral operators. All compression methods in Inti.jl take an
`IntegralOperator` and return a new object that represents a compressed version
of the operator. The following methods are available:

- [`assemble_matrix`](@ref) - Assemble the operator as a dense matrix. Not
  really a compression method, but useful for debugging and small problems.
- [`assemble_hmatrix`](@ref) - Assemble the operator as a hierarchical matrix.
- [`assemble_fmm`](@ref) - Assemble the operator using the fast multipole method.

All of the compression methods take an `IntegralOperator` and return a new
object that represents a compressed version of the operator. Not all methods
work with all operators; the next sections provide more details on their usage
and limitations.

## Dense matrix

```@docs; canonical = false
assemble_matrix
```

## Hierarchical matrix

```@docs; canonical = false
assemble_hmatrix
```

## Fast multipole method

```@docs; canonical = false
assemble_fmm
```
