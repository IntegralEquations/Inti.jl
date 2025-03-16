# Correction methods

```@meta
CurrentModule = Inti
```

!!! warning "Work in progress"
    This tutorial is still a work in progress. We will update it with more
    details and examples in the future.

!!! note "Important points covered in this tutorial"
    - Overview of the correction methods available in Inti.jl
    - Details and limitations of the various correction methods
    - Guideline on how to choose a correction method

When the underlying kernel is singular, a *correction* is usually necessary in
order to obtain accurate results in the approximation of the underlying integral
operator by a quadrature. At present, Inti.jl provides the following functions
to correct for singularities:

- [`local_correction`](@ref)
- [`bdim_correction`](@ref)
- [`vdim_correction`](@ref)

They have different strengths and weaknesses, and we will discuss them in the
following sections.

!!! note "High-level API"
    Note that the [`single_double_layer`](@ref),
    [`adj_double_layer_hypersingular`](@ref), and [`volume_potential`](@ref)
    functions have high-level API with a `correction` keyword argument that
    allows one to specify the correction method to use when constructing the
    integral operators; see the documentation of these functions for more
    details.

## Local correction

[guiggiani1992general](@cite)

## Boundary density interpolation method

## Volume density interpolation method

## Martensen-Kussmaul method
