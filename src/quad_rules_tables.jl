#=
Tabulated Gaussian quadrature rules from GMSH.
Obtained from `https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_0/Numeric/GaussQuadratureTri.cpp`
and `https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_0/Numeric/GaussQuadratureTet.cpp`.
=#

include("quad_rules_tables_tetrahedron.jl")
include("quad_rules_tables_triangle.jl")

##
# Dictionaries that contains the quadrature rules
# for various `ReferenceShape`. The dictionary
# key represents the number of quadrature nodes.

const TRIANGLE_GAUSS_QRULES = Dict(1 => TRIANGLE_G1N1,
                                   3 => TRIANGLE_G2N3,
                                   4 => TRIANGLE_G3N4,
                                   6 => TRIANGLE_G4N6,
                                   7 => TRIANGLE_G5N7,
                                   12 => TRIANGLE_G6N12,
                                   13 => TRIANGLE_G7N13,
                                   16 => TRIANGLE_G8N16,
                                   19 => TRIANGLE_G9N19,
                                   25 => TRIANGLE_G10N25,
                                   33 => TRIANGLE_G12N33)

# map a desired quadrature order to the number of nodes
const TRIANGLE_GAUSS_ORDER_TO_NPTS = Dict(1 => 1,
                                          2 => 3,
                                          3 => 4,
                                          4 => 6,
                                          5 => 7,
                                          6 => 12,
                                          7 => 13,
                                          8 => 16,
                                          9 => 19,
                                          10 => 25,
                                          12 => 33)
const TRIANGLE_GAUSS_NPTS_TO_ORDER = Dict((v, k) for (k, v) in TRIANGLE_GAUSS_ORDER_TO_NPTS)

const TETAHEDRON_GAUSS_QRULES = Dict(1 => TETAHEDRON_G1N1,
                                     4 => TETAHEDRON_G2N4,
                                     5 => TETAHEDRON_G3N5,
                                     11 => TETAHEDRON_G4N11,
                                     14 => TETAHEDRON_G5N14,
                                     24 => TETAHEDRON_G6N24,
                                     31 => TETAHEDRON_G7N31,
                                     43 => TETAHEDRON_G8N43)

# map a desired quadrature order to the number of nodes
const TETRAHEDRON_GAUSS_ORDER_TO_NPTS = Dict(1 => 1, 2 => 4, 3 => 5, 4 => 11, 5 => 14,
                                             6 => 24, 7 => 31, 8 => 43)
const TETRAHEDRON_GAUSS_NPTS_TO_ORDER = Dict((v, k)
                                             for (k, v) in TETRAHEDRON_GAUSS_ORDER_TO_NPTS)

##
const GAUSS_QRULES = Dict(ReferenceTriangle => TRIANGLE_GAUSS_QRULES,
                          ReferenceTetrahedron => TETAHEDRON_GAUSS_QRULES)
