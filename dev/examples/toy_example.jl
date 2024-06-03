using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

#nb ## Environment setup
#nb const DEPENDENCIES = ["GLMakie"];
#nb ## __NOTEBOOK_SETUP__

# # Toy example

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](toy_example.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/toy_example.ipynb)

#=

Template for Init.jl examples. The `make.jl` script in the `docs` folder will
use [Literate.jl](https://fredrikekre.github.io/Literate.jl/v2/) to generate the
markdown and notebook files for all examples in the `examples` variable of the
`make.jl` file. The `#md` lines above add badges for downloading the notebook
and viewing it on nbviewer.

=#

using GLMakie
