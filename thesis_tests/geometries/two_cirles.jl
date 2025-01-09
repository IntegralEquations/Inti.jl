##### Two circles
δ = 0.001
χ = s -> SVector(cos(s), sin(s))
Γ₁ = Inti.parametric_curve(0.0, 2π) do s
    return χ(s) - SVector(1 + δ/2, 0)
end |> Inti.Domain
Γ₂ = Inti.parametric_curve(0.0, 2π) do s
    return χ(s) + SVector(1 + δ/2, 0)
end |> Inti.Domain
Γ = Γ₁ ∪ Γ₂
if t == :interior
    xs = ntuple(i -> 3, N)
    tset = [SVector(0.5cos(s) - 1 - δ/2, 0.5sin(s)) for s in LinRange(-π, π, 10)] ∪
           [SVector(0.5cos(s) + 1 + δ/2, 0.5sin(s)) for s in LinRange(-π, π, 10)]
else
    xs = SVector(1, 0.1)
    tset = [SVector(5cos(s), 5sin(s)) for s in LinRange(-π, π, 10)]
end

k = 15