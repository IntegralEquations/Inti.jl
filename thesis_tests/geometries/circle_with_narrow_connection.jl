δ = 0.001
Γ1 = Inti.parametric_curve(1, -1) do x
    SVector(x, δ)
end |> Inti.Domain
Γ2 = Inti.parametric_curve(-1, 1) do x
    SVector(x,-δ)
end |> Inti.Domain
r, θ = √(1+δ^2), atan(δ)
Γ3 = Inti.parametric_curve(θ-π, π-θ) do s
    SVector(r*cos(s)+2, r*sin(s))
end |> Inti.Domain
Γ4 = Inti.parametric_curve(θ, 2π-θ) do s
    SVector(r*cos(s)-2, r*sin(s))
end |> Inti.Domain
Γ = Γ1 ∪ Γ2 ∪ Γ3 ∪ Γ4

if t == :interior
    xs = ntuple(i -> 3, N)
    tset = [SVector(0.5cos(s)+2, 0.5sin(s)) for s in LinRange(-π, π, 10)] ∪
           [SVector(0.5cos(s)-2, 0.5sin(s)) for s in LinRange(-π, π, 10)]
else
    xs = SVector(2.5, 0.1)
    tset = [SVector(5cos(s), 5sin(s)) for s in LinRange(-π, π, 10)]
end

k = 10