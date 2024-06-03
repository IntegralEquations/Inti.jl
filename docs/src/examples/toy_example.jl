using Markdown                        #src
import Pkg                            #src
docsdir = joinpath(@__DIR__, "../..") #src
Pkg.activate(docsdir)                 #src

#nb ## Environment setup
#nb const DEPENDENCIES = ["GLMakie"];
#nb ## __NOTEBOOK_SETUP__

# Toy example

using GLMakie
