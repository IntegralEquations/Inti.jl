##### 8-like
δ = 1e-4
a, b = 0, 1
χ = s -> SVector(
    (1 + cos(2s * 2π)/2) * cos(s * 2π),
    (1 + (2-δ)*cos(2s * 2π)/2) * sin(s * 2π),
)
xt = χ((3a+b)/4)
Γ1 = Inti.parametric_curve(χ, a, (a+b)/2) |> Inti.Domain
Γ2 = Inti.parametric_curve(χ, (a+b)/2, b) |> Inti.Domain
Γ = Γ1 ∪ Γ2
tset = [0.5SVector(cos(s), sin(s)) - SVector(0.8, 0) for s in LinRange(-π, π, 10)] ∪
        [0.5SVector(cos(s), sin(s)) + SVector(0.8, 0) for s in LinRange(-π, π, 10)]

k = 20