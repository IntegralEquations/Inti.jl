##### Circle
χ = s -> SVector(cos(s), sin(s))
Γ = Inti.parametric_curve(χ, -π, π) |> Inti.Domain
if t == :interior
    xs = ntuple(i -> 3, N)
    tset = [0.5 * χ(s) for s in LinRange(-π, π, 10)]
    # tset = [SVector(10, 0)]
else
    xs = ntuple(i -> 0.1, N)
    tset = [2 * χ(s) for s in LinRange(-π, π, 10)]
end

k = 10