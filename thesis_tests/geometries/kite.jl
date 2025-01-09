##### Kite
χ = s -> SVector(
    2.5 + cos(s) + 0.65 * cos(2s) - 0.65,
    1.5 * sin(s),
)
Γ = Inti.parametric_curve(χ, 0.0, 2π)
if t == :interior
    xs = ntuple(i -> 3, N)
    tset = [SVector(0.5cos(s)+2.5, 0.5sin(s)) for s in LinRange(-π, π, 10)]
else
    xs = SVector(2.5, 0.1)
    tset = [SVector(3cos(s)+2.5, 3sin(s)) for s in LinRange(-π, π, 10)]
end

k = 10