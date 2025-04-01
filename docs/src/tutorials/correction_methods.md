# Correction methods

```@meta
CurrentModule = Inti
```

!!! note "Important Points Covered in This Tutorial"
    - Overview of the correction methods available in Inti.jl.
    - Details and limitations of the various correction methods.
    - Guidelines on how to choose a correction method.

When the underlying kernel is singular, a *correction* is usually necessary to obtain
accurate results in the approximation of the underlying integral operator by a quadrature.
Currently, Inti.jl provides the following functions to correct for singularities:

- [`adaptive_correction`](@ref)
- [`bdim_correction`](@ref)
- [`vdim_correction`](@ref)

Each method has its own strengths and weaknesses, which will be discussed in the following
sections.

!!! note "High-Level API"
    The [`single_double_layer`](@ref), [`adj_double_layer_hypersingular`](@ref), and
    [`volume_potential`](@ref) functions provide a high-level API with a `correction`
    keyword argument. This allows users to specify the correction method to use when
    constructing the integral operators. See the documentation of these functions for more
    details.

## Adaptive Correction

The [`adaptive_correction`](@ref) method combines adaptive quadrature for nearly singular
integrals with a direct evaluation method for singular integrals, based on
[guiggiani1992general](@cite). It is a robust method suitable for a wide range of kernels,
as long as the singularities are no worse than a Hadamard finite-part (e.g., ``1/r^3`` in 3D
and ``1/r^2`` in 2D). This makes it a good default choice for most problems.

### Strengths

- Robust method that works for a wide range of kernels.
- Conceptually straightforward and easy to use.
- Handles open surfaces.

### Weaknesses

- Can be slow for large problems and high accuracy requirements.
- Sometimes difficult to tune parameters for optimal performance.
- Round-off errors in certain cases can make achieving high accuracy challenging.

### Docstrings

```@docs; canonical = false
adaptive_correction
```

## Boundary Density Interpolation Method

The [`bdim_correction`](@ref) method implements the general-purpose version of the density
interpolation method proposed in [faria2021general](@cite). Is a global correction method
that uses solutions of the underlying PDE, together with Green's identities, to interpolate
the density on the boundary. It works best for low to moderate-order quadratures and is
particularly useful for smooth boundaries when the PDE.

### Strengths

- Can be faster and more accurate for standard problems, such as scattering by closed
  surfaces.
- Easier parameter tuning, as it only requires knowing whether the target surface is inside,
  outside, or on the boundary of the source.

### Weaknesses

- Only suitable for closed surfaces.
- The underlying kernel must be related to the fundamental solution of a PDE.

### Docstrings

```@docs; canonical = false
bdim_correction
```

## Volume density interpolation method

TODO
