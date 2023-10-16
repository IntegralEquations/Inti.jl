import Pkg               #src
docsdir = joinpath(@__DIR__,"../..") #src
Pkg.activate(docsdir) #src

md"""
# Mock example
"""

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](mock_example.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/mock_example.ipynb)

md"""

Testing how to use `Literate.jl` with `Documenter.jl` to generate documentation.

!!! note "Mock admonition"
    - This should work using `Documenter Markdown` syntax
    - If it does not, check back again

"""

using Inti
