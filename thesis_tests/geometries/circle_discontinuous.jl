##### Circle
a, b = 0, 1
χ = s -> SVector(cos(2π * s), sin(2π * s))
xt = χ((3a+b)/4)
Γ1 = Inti.parametric_curve(χ, a, (a+b)/2) |> Inti.Domain
Γ2 = Inti.parametric_curve(χ, (a+b)/2, b) |> Inti.Domain
Γ = Γ1 ∪ Γ2

k = 10
