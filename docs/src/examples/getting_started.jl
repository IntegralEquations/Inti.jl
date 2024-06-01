using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

#nb ## Environment setup
#nb const DEPENDENCIES = ["GLMakie", "LinearAlgebra"];
#nb ## __NOTEBOOK_SETUP__

# # [Getting started](@id getting_started)

#md # [![ipynb](https://img.shields.io/badge/download-ipynb-blue)](getting_started.ipynb)
#md # [![nbviewer](https://img.shields.io/badge/show-nbviewer-blue.svg)](@__NBVIEWER_ROOT_URL__/examples/generated/getting_started.ipynb)

# !!! note "Important points covered in this tutorial"
#       - Create a domain and mesh
#       - Solve a basic boundary integral equation
#       - Visualize the solution

# This first tutorial provides a simple introduction to the basic functinality
# of `Inti`...

using Inti

kite   = (s) -> (cos(2π * s[1]) + 0.65 * cos(4π * s[1]) - 0.65, 1.5 * sin(2π * s[1]))
circle = (s) -> (cos(2π * s[1]), sin(2π * s[1]))
e1     = Inti.parametric_curve(f, 0.0, 1.0)
e2     = Inti.parametric_curve(circle, 0.0, 1.0)
Γ      = Inti.Domain(e1, e2)
dict   = Dict(e1 => (100,), e2 => (100,))
msh    = Inti.meshgen(Γ, dict)

using Meshes
using GLMakie

viz(msh)
