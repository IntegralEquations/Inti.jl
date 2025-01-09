##### 8-like
δ = 0.001
χ = s -> SVector(
    (1 + cos(2s)/2) * cos(s),
    (1 + (2-δ)*cos(2s)/2) * sin(s),
)
Γ = Inti.parametric_curve(χ, -π, π) |> Inti.Domain
if t == :interior
    xs = ntuple(i -> 1, N)
    tset = [0.5SVector(cos(s), sin(s)) - SVector(0.8, 0) for s in LinRange(-π, π, 10)] ∪
           [0.5SVector(cos(s), sin(s)) + SVector(0.8, 0) for s in LinRange(-π, π, 10)]
else
    xs = SVector(1, 0.1)
    tset = [SVector(5cos(s), 5sin(s)) for s in LinRange(-π, π, 10)]
end

k = 20