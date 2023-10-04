## -----------------------------------------------------------------------------
#*! Quadrature rule for an interpolation of order 1 on the tetrahedron *#
#* 'Higher-order Finite Elements', P.Solin, K.Segeth and I. Dolezel *#

const TETAHEDRON_G1N1 = ((SVector(0.25, 0.25, 0.25), 0.166666666666667),)

## 0 negative weights, 0 points outside of the tetrahedron,  total sum of the
## weights is 1/6

## -----------------------------------------------------------------------------
#*! Quadrature rule for an interpolation of order 2 on the tetrahedron *#
#* 'Higher-order Finite Elements', P.Solin, K.Segeth and I. Dolezel *#

const TETAHEDRON_G2N4 = ((SVector(0.138196601125, 0.138196601125, 0.138196601125),
                          0.0416666666666667),
                         (SVector(0.585410196625, 0.138196601125, 0.138196601125),
                          0.0416666666666667),
                         (SVector(0.138196601125, 0.585410196625, 0.138196601125),
                          0.0416666666666667),
                         (SVector(0.138196601125, 0.138196601125, 0.585410196625),
                          0.0416666666666667))

## 0 negative weights, 0 points outside of the tetrahedron,  total sum of the
## weights is 1/6

## -----------------------------------------------------------------------------
#*! Quadrature rule for an interpolation of order 3 on the tetrahedron *#
#* 'Higher-order Finite Elements', P.Solin, K.Segeth and I. Dolezel *#

const TETAHEDRON_G3N5 = ((SVector(0.250000000000, 0.250000000000, 0.250000000000),
                          -0.133333333333333),
                         (SVector(0.166666666667, 0.166666666667, 0.166666666667),
                          +0.075000000000000),
                         (SVector(0.166666666667, 0.166666666667, 0.500000000000),
                          +0.075000000000000),
                         (SVector(0.166666666667, 0.500000000000, 0.166666666667),
                          +0.075000000000000),
                         (SVector(0.500000000000, 0.166666666667, 0.166666666667),
                          +0.075000000000000))

## 1 negative weights, 0 points outside of the tetrahedron,  total sum of the
## weights is 1/6

## -----------------------------------------------------------------------------
#*! Quadrature rule for an interpolation of order 4 on the tetrahedron *#
#* 'Higher-order Finite Elements', P.Solin, K.Segeth and I. Dolezel *#

const TETAHEDRON_G4N11 = ((SVector(0.2500000000000, 0.2500000000000, 0.2500000000000),
                           -0.0131555555555),
                          (SVector(0.0714285714286, 0.0714285714286, 0.0714285714286),
                           +0.0076222222222),
                          (SVector(0.0714285714286, 0.0714285714286, 0.7857142857140),
                           +0.0076222222222),
                          (SVector(0.0714285714286, 0.7857142857140, 0.0714285714286),
                           +0.0076222222222),
                          (SVector(0.7857142857140, 0.0714285714286, 0.0714285714286),
                           +0.0076222222222),
                          (SVector(0.3994035761670, 0.3994035761670, 0.1005964238330),
                           +0.0248888888888),
                          (SVector(0.3994035761670, 0.1005964238330, 0.3994035761670),
                           +0.0248888888888),
                          (SVector(0.1005964238330, 0.3994035761670, 0.3994035761670),
                           +0.0248888888888),
                          (SVector(0.3994035761670, 0.1005964238330, 0.1005964238330),
                           +0.0248888888888),
                          (SVector(0.1005964238330, 0.3994035761670, 0.1005964238330),
                           +0.0248888888888),
                          (SVector(0.1005964238330, 0.1005964238330, 0.3994035761670),
                           +0.0248888888888))

## 1 negative weights, 0 points outside of the tetrahedron,  total sum of the
## weights is 1/6

## -----------------------------------------------------------------------------
#*! Quadrature rule for an interpolation of order 5 on the tetrahedron *#
#* 'Higher-order Finite Elements', P.Solin, K.Segeth and I. Dolezel *#

const TETAHEDRON_G5N14 = ((SVector(0.0927352503109, 0.0927352503109, 0.0927352503109),
                           0.01224884051940),
                          (SVector(0.7217942490670, 0.0927352503109, 0.0927352503109),
                           0.01224884051940),
                          (SVector(0.0927352503109, 0.7217942490670, 0.0927352503109),
                           0.01224884051940),
                          (SVector(0.0927352503109, 0.0927352503109, 0.7217942490670),
                           0.01224884051940),
                          (SVector(0.3108859192630, 0.3108859192630, 0.3108859192630),
                           0.01878132095300),
                          (SVector(0.0673422422101, 0.3108859192630, 0.3108859192630),
                           0.01878132095300),
                          (SVector(0.3108859192630, 0.0673422422101, 0.3108859192630),
                           0.01878132095300),
                          (SVector(0.3108859192630, 0.3108859192630, 0.0673422422101),
                           0.01878132095300),
                          (SVector(0.4544962958740, 0.4544962958740, 0.0455037041256),
                           0.00709100346285),
                          (SVector(0.4544962958740, 0.0455037041256, 0.4544962958740),
                           0.00709100346285),
                          (SVector(0.0455037041256, 0.4544962958740, 0.4544962958740),
                           0.00709100346285),
                          (SVector(0.4544962958740, 0.0455037041256, 0.0455037041256),
                           0.00709100346285),
                          (SVector(0.0455037041256, 0.4544962958740, 0.0455037041256),
                           0.00709100346285),
                          (SVector(0.0455037041256, 0.0455037041256, 0.4544962958740),
                           0.00709100346285))

## 0 negative weights, 0 points outside of the tetrahedron,  total sum of the
## weights is 1

## -----------------------------------------------------------------------------
#*! Quadrature rule for an interpolation of order 6 on the tetrahedron *#
#* 'Higher-order Finite Elements', P.Solin, K.Segeth and I. Dolezel *#

const TETAHEDRON_G6N24 = ((SVector(0.2146028712590, 0.2146028712590, 0.2146028712590),
                           0.006653791709700),
                          (SVector(0.3561913862230, 0.2146028712590, 0.2146028712590),
                           0.006653791709700),
                          (SVector(0.2146028712590, 0.3561913862230, 0.2146028712590),
                           0.006653791709700),
                          (SVector(0.2146028712590, 0.2146028712590, 0.3561913862230),
                           0.006653791709700),
                          (SVector(0.0406739585346, 0.0406739585346, 0.0406739585346),
                           0.001679535175883),
                          (SVector(0.8779781243960, 0.0406739585346, 0.0406739585346),
                           0.001679535175883),
                          (SVector(0.0406739585346, 0.8779781243960, 0.0406739585346),
                           0.001679535175883),
                          (SVector(0.0406739585346, 0.0406739585346, 0.8779781243960),
                           0.001679535175883),
                          (SVector(0.3223378901420, 0.3223378901420, 0.3223378901420),
                           0.009226196923950),
                          (SVector(0.0329863295732, 0.3223378901420, 0.3223378901420),
                           0.009226196923950),
                          (SVector(0.3223378901420, 0.0329863295732, 0.3223378901420),
                           0.009226196923950),
                          (SVector(0.3223378901420, 0.3223378901420, 0.0329863295732),
                           0.009226196923950),
                          (SVector(0.0636610018750, 0.0636610018750, 0.2696723314580),
                           0.008035714285717),
                          (SVector(0.0636610018750, 0.2696723314580, 0.0636610018750),
                           0.008035714285717),
                          (SVector(0.0636610018750, 0.0636610018750, 0.6030056647920),
                           0.008035714285717),
                          (SVector(0.0636610018750, 0.6030056647920, 0.0636610018750),
                           0.008035714285717),
                          (SVector(0.0636610018750, 0.2696723314580, 0.6030056647920),
                           0.008035714285717),
                          (SVector(0.0636610018750, 0.6030056647920, 0.2696723314580),
                           0.008035714285717),
                          (SVector(0.2696723314580, 0.0636610018750, 0.0636610018750),
                           0.008035714285717),
                          (SVector(0.2696723314580, 0.0636610018750, 0.6030056647920),
                           0.008035714285717),
                          (SVector(0.2696723314580, 0.6030056647920, 0.0636610018750),
                           0.008035714285717),
                          (SVector(0.6030056647920, 0.0636610018750, 0.2696723314580),
                           0.008035714285717),
                          (SVector(0.6030056647920, 0.0636610018750, 0.0636610018750),
                           0.008035714285717),
                          (SVector(0.6030056647920, 0.2696723314580, 0.0636610018750),
                           0.008035714285717))

## 0 negative weights, 0 points outside of the tetrahedron,  total sum of the
## weights is 1

## -----------------------------------------------------------------------------
#*! Quadrature rule for an interpolation of order 7 on the tetrahedron *#
#* 'Higher-order Finite Elements', P.Solin, K.Segeth and I. Dolezel *#

const TETAHEDRON_G7N31 = ((SVector(0.50000000000000, 0.50000000000000, 0.00000000000000),
                           +0.000970017636685),
                          (SVector(0.50000000000000, 0.00000000000000, 0.50000000000000),
                           +0.000970017636685),
                          (SVector(0.00000000000000, 0.50000000000000, 0.50000000000000),
                           +0.000970017636685),
                          (SVector(0.00000000000000, 0.00000000000000, 0.50000000000000),
                           +0.000970017636685),
                          (SVector(0.00000000000000, 0.50000000000000, 0.00000000000000),
                           +0.000970017636685),
                          (SVector(0.50000000000000, 0.00000000000000, 0.00000000000000),
                           +0.000970017636685),
                          (SVector(0.25000000000000, 0.25000000000000, 0.25000000000000),
                           +0.018264223466167),
                          (SVector(0.07821319233030, 0.07821319233030, 0.07821319233030),
                           +0.010599941524417),
                          (SVector(0.07821319233030, 0.07821319233030, 0.76536042300900),
                           +0.010599941524417),
                          (SVector(0.07821319233030, 0.76536042300900, 0.07821319233030),
                           +0.010599941524417),
                          (SVector(0.76536042300900, 0.07821319233030, 0.07821319233030),
                           +0.010599941524417),
                          (SVector(0.12184321666400, 0.12184321666400, 0.12184321666400),
                           -0.062517740114333),
                          (SVector(0.12184321666400, 0.12184321666400, 0.63447035000800),
                           -0.062517740114333),
                          (SVector(0.12184321666400, 0.63447035000800, 0.12184321666400),
                           -0.062517740114333),
                          (SVector(0.63447035000800, 0.12184321666400, 0.12184321666400),
                           -0.062517740114333),
                          (SVector(0.33253916444600, 0.33253916444600, 0.33253916444600),
                           +0.004891425263067),
                          (SVector(0.33253916444600, 0.33253916444600, 0.00238250666074),
                           +0.004891425263067),
                          (SVector(0.33253916444600, 0.00238250666074, 0.33253916444600),
                           +0.004891425263067),
                          (SVector(0.00238250666074, 0.33253916444600, 0.33253916444600),
                           +0.004891425263067),
                          (SVector(0.10000000000000, 0.10000000000000, 0.20000000000000),
                           +0.027557319224000),
                          (SVector(0.10000000000000, 0.20000000000000, 0.10000000000000),
                           +0.027557319224000),
                          (SVector(0.10000000000000, 0.10000000000000, 0.60000000000000),
                           +0.027557319224000),
                          (SVector(0.10000000000000, 0.60000000000000, 0.10000000000000),
                           +0.027557319224000),
                          (SVector(0.10000000000000, 0.20000000000000, 0.60000000000000),
                           +0.027557319224000),
                          (SVector(0.10000000000000, 0.60000000000000, 0.20000000000000),
                           +0.027557319224000),
                          (SVector(0.20000000000000, 0.10000000000000, 0.10000000000000),
                           +0.027557319224000),
                          (SVector(0.20000000000000, 0.10000000000000, 0.60000000000000),
                           +0.027557319224000),
                          (SVector(0.20000000000000, 0.60000000000000, 0.10000000000000),
                           +0.027557319224000),
                          (SVector(0.60000000000000, 0.10000000000000, 0.20000000000000),
                           +0.027557319224000),
                          (SVector(0.60000000000000, 0.10000000000000, 0.10000000000000),
                           +0.027557319224000),
                          (SVector(0.60000000000000, 0.20000000000000, 0.10000000000000),
                           +0.027557319224000))

## 4 negative weights, 0 points outside of the tetrahedron

## -----------------------------------------------------------------------------
#*! Quadrature rule for an interpolation of order 8 on the tetrahedron *#
#* 'Higher-order Finite Elements', P.Solin, K.Segeth and I. Dolezel *#

const TETAHEDRON_G8N43 = ((SVector(0.2500000000000, 0.2500000000000, 0.2500000000000),
                           -0.020500188658667),
                          (SVector(0.2068299316110, 0.2068299316110, 0.2068299316110),
                           +0.014250305822867),
                          (SVector(0.2068299316110, 0.2068299316110, 0.3795102051680),
                           +0.014250305822867),
                          (SVector(0.2068299316110, 0.3795102051680, 0.2068299316110),
                           +0.014250305822867),
                          (SVector(0.3795102051680, 0.2068299316110, 0.2068299316110),
                           +0.014250305822867),
                          (SVector(0.0821035883105, 0.0821035883105, 0.0821035883105),
                           +0.001967033313133),
                          (SVector(0.0821035883105, 0.0821035883105, 0.7536892350680),
                           +0.001967033313133),
                          (SVector(0.0821035883105, 0.7536892350680, 0.0821035883105),
                           +0.001967033313133),
                          (SVector(0.7536892350680, 0.0821035883105, 0.0821035883105),
                           +0.001967033313133),
                          (SVector(0.0057819505052, 0.0057819505052, 0.0057819505052),
                           +0.000169834109093),
                          (SVector(0.0057819505052, 0.0057819505052, 0.9826541484840),
                           +0.000169834109093),
                          (SVector(0.0057819505052, 0.9826541484840, 0.0057819505052),
                           +0.000169834109093),
                          (SVector(0.9826541484840, 0.0057819505052, 0.0057819505052),
                           +0.000169834109093),
                          (SVector(0.0505327400189, 0.0505327400189, 0.4494672599810),
                           +0.004579683824467),
                          (SVector(0.0505327400189, 0.4494672599810, 0.0505327400189),
                           +0.004579683824467),
                          (SVector(0.4494672599810, 0.0505327400189, 0.0505327400189),
                           +0.004579683824467),
                          (SVector(0.0505327400189, 0.4494672599810, 0.4494672599810),
                           +0.004579683824467),
                          (SVector(0.4494672599810, 0.0505327400189, 0.4494672599810),
                           +0.004579683824467),
                          (SVector(0.4494672599810, 0.4494672599810, 0.0505327400189),
                           +0.004579683824467),
                          (SVector(0.2290665361170, 0.2290665361170, 0.0356395827885),
                           +0.005704485808683),
                          (SVector(0.2290665361170, 0.0356395827885, 0.2290665361170),
                           +0.005704485808683),
                          (SVector(0.2290665361170, 0.2290665361170, 0.5062273449780),
                           +0.005704485808683),
                          (SVector(0.2290665361170, 0.5062273449780, 0.2290665361170),
                           +0.005704485808683),
                          (SVector(0.2290665361170, 0.0356395827885, 0.5062273449780),
                           +0.005704485808683),
                          (SVector(0.2290665361170, 0.5062273449780, 0.0356395827885),
                           +0.005704485808683),
                          (SVector(0.0356395827885, 0.2290665361170, 0.2290665361170),
                           +0.005704485808683),
                          (SVector(0.0356395827885, 0.2290665361170, 0.5062273449780),
                           +0.005704485808683),
                          (SVector(0.0356395827885, 0.5062273449780, 0.2290665361170),
                           +0.005704485808683),
                          (SVector(0.5062273449780, 0.2290665361170, 0.0356395827885),
                           +0.005704485808683),
                          (SVector(0.5062273449780, 0.2290665361170, 0.2290665361170),
                           +0.005704485808683),
                          (SVector(0.5062273449780, 0.0356395827885, 0.2290665361170),
                           +0.005704485808683),
                          (SVector(0.0366077495532, 0.0366077495532, 0.1904860419350),
                           +0.002140519141167),
                          (SVector(0.0366077495532, 0.1904860419350, 0.0366077495532),
                           +0.002140519141167),
                          (SVector(0.0366077495532, 0.0366077495532, 0.7362984589590),
                           +0.002140519141167),
                          (SVector(0.0366077495532, 0.7362984589590, 0.0366077495532),
                           +0.002140519141167),
                          (SVector(0.0366077495532, 0.1904860419350, 0.7362984589590),
                           +0.002140519141167),
                          (SVector(0.0366077495532, 0.7362984589590, 0.1904860419350),
                           +0.002140519141167),
                          (SVector(0.1904860419350, 0.0366077495532, 0.0366077495532),
                           +0.002140519141167),
                          (SVector(0.1904860419350, 0.0366077495532, 0.7362984589590),
                           +0.002140519141167),
                          (SVector(0.1904860419350, 0.7362984589590, 0.0366077495532),
                           +0.002140519141167),
                          (SVector(0.7362984589590, 0.0366077495532, 0.1904860419350),
                           +0.002140519141167),
                          (SVector(0.7362984589590, 0.0366077495532, 0.0366077495532),
                           +0.002140519141167),
                          (SVector(0.7362984589590, 0.1904860419350, 0.0366077495532),
                           +0.002140519141167))

## 1 negative weights, 0 points outside of the tetrahedron
