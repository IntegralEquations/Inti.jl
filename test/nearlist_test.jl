using Test
using LinearAlgebra
using Inti
using Random
using Meshes
using GLMakie

include("test_utils.jl")
Random.seed!(1)

N = 2
t = :interior
pde = Inti.Laplace(; dim = N)
h = 0.2

Inti.clear_entities!()
Ω, msh = gmsh_disks([([0.0,0.0],1.0,1.0), ([-2.1,0.0],1.0,1.0)];meshsize = h, order = 2)
Γ = Inti.external_boundary(Ω)
quad = Inti.Quadrature(msh[Γ]; qorder = 3)

# Nl = Inti.near_elements(quad;tol=0.2)
Ncl = Inti.near_components(quad;tol=0.2)
fig, _, _ = viz(msh;showsegments = false,alpha=0.3)

E = first(keys(Ncl))[1]; i = 1
viz!(Inti.elements(msh[Γ], E)[i];color=:red)
for (E_, j) in Ncl[(E, i)][2]
    viz!(Inti.elements(msh[Γ], E_)[j];color=:blue,alpha=0.3)
end

# for (E, nl) in Nl
#     i = 1
#     viz!(Inti.elements(msh[Γ], E)[i];color=:red)
#     for j in nl[i]
#         viz!(Inti.elements(msh[Γ], E)[j];color=:blue,alpha=0.3)
#     end  
# end
display(fig)