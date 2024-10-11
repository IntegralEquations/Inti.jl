using Test
using Inti

@testset "Issue 98" begin
    Ω = Inti.Domain(Inti.GeometricEntity("ellipsoid"))
    Γ = Inti.boundary(Ω)
    mesh = Inti.meshgen(Γ; meshsize = 0.4)
    quad = Inti.Quadrature(mesh; qorder = 1)

    pde = Inti.Laplace(; dim = 3)

    @test_throws ArgumentError Inti.single_double_layer(;
        pde,
        source = quad,
        target = quad,
        compression = (method = :none,),
        correction = (method = :dim,),
    )
end
