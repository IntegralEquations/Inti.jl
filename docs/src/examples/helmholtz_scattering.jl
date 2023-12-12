import Pkg               #src
docsdir = joinpath(@__DIR__,"../..") #src
Pkg.activate(docsdir) #src

md"""
# [Helmholtz scattering](@id helmholtz_scattering)
"""

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](helmholtz_scattering.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/helmholtz_scattering.ipynb)

md"""

Testing how to use `Literate.jl` with `Documenter.jl` to generate documentation.

!!! note "Mock admonition"
    - This should work using `Documenter Markdown` syntax
    - If it does not, check back again

"""

using Inti
