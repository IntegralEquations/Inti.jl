#=
Tabulated Gaussian quadrature rules from GMSH.
Obtained from `https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_0/Numeric/GaussQuadratureTri.cpp`
and `https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_0/Numeric/GaussQuadratureTet.cpp`.

Also includes Vioreanu-Rokhlin interpolating quadratures, see
`Spectra of Multiplication Operators as a Numerical Tool', B. Vioreanu and V. Rokhlin, SIAM J Sci. Comput. (2014).
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

const TRIANGLE_VR_QRULES = Dict(1 => TRIANGLE_VR1N1,
                                3 => TRIANGLE_VR2N3,
                                6 => TRIANGLE_VR4N6,
                                10 => TRIANGLE_VR5N10,
                                15 => TRIANGLE_VR7N15,
                                21 => TRIANGLE_VR9N21,
                                28 => TRIANGLE_VR11N28,
                                36 => TRIANGLE_VR12N36,
                                45 => TRIANGLE_VR14N45,
                                55 => TRIANGLE_VR16N55,
                                66 => TRIANGLE_VR17N66)
const TRIANGLE_VR_ORDER_TO_NPTS = Dict(1 => 1,
                                       2 => 3,
                                       4 => 6,
                                       5 => 10,
                                       7 => 15,
                                       9 => 21,
                                       11 => 28,
                                       12 => 36,
                                       14 => 45,
                                       16 => 55,
                                       17 => 66)
const TRIANGLE_VR_NPTS_TO_ORDER = Dict((v, k) for (k, v) in TRIANGLE_VR_ORDER_TO_NPTS)

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

# TODO
const TETRAHEDRON_VR_QRULES = Dict(1 => TETRAHEDRON_VR1N1,
                                   4 => TETRAHEDRON_VR2N4,
                                   10 => TETRAHEDRON_VR3N10,
                                   20 => TETRAHEDRON_VR5N20,
                                   35 => TETRAHEDRON_VR6N35)
const TETRAHEDRON_VR_ORDER_TO_NPTS = Dict(1 => 1,
                                          2 => 4,
                                          3 => 10,
                                          5 => 20,
                                          6 => 35)
const TETRAHEDRON_VR_NPTS_TO_ORDER = Dict((v, k) for (k, v) in TETRAHEDRON_VR_ORDER_TO_NPTS)

##
const GAUSS_QRULES = Dict(ReferenceTriangle => TRIANGLE_GAUSS_QRULES,
                          ReferenceTetrahedron => TETAHEDRON_GAUSS_QRULES)

const VR_QRULES = Dict(ReferenceTriangle => TRIANGLE_VR_QRULES,
                       ReferenceTetrahedron => TETRAHEDRON_VR_QRULES)
