using Test
using Inti
using Gmsh
using LinearAlgebra
using StaticArrays

include("test_utils.jl")

# Curving a mesh whose boundary admits no single smooth global parametrization
# (here a sphere) requires an *atlas* of overlapping charts. See the docstrings
# of `Inti.ParametricChart` and `Inti.sphere_atlas`.
@testset "Multi-patch curved mesh (sphere)" begin
    Inti.clear_entities!()
    radius = 1.0
    meshsize = 0.4
    Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = radius, meshsize = meshsize)
    Γ = Inti.external_boundary(Ω)

    truevol = 4 / 3 * π * radius^3
    truearea = 4 * π * radius^2

    atlas = Inti.sphere_atlas(; radius = radius)
    @test length(atlas) == 6

    @testset "chart inverses" begin
        for (i, c) in enumerate(atlas)
            for α in (SVector(0.0, 0.0), SVector(0.7, -1.3), SVector(-2.0, 1.9))
                x = SVector{3, Float64}(c.param(α))
                @test norm(x) ≈ radius
                @test c.inverse(x) ≈ α
            end
            # a point on the opposite hemisphere is not on this chart
            @test isnothing(c.inverse(-SVector{3, Float64}(c.param(SVector(0.0, 0.0)))))
        end
    end

    θ = 3 # smoothness order of curved elements
    crvmsh = Inti.curve_mesh(msh, atlas, θ)
    Ωₕ = crvmsh[Ω]
    Γₕ = crvmsh[Γ]

    qorder = 5
    Ωₕ_quad = Inti.Quadrature(Ωₕ; qorder = qorder)
    Γₕ_quad = Inti.Quadrature(Γₕ; qorder = qorder)

    # the curved boundary elements lie on the sphere to machine precision, no
    # matter which chart of the atlas each element was curved through
    @test maximum(q -> abs(norm(q.coords) - radius), Γₕ_quad) < 1.0e-12
    # curved volume elements stay inside the ball
    @test maximum(q -> norm(q.coords), Ωₕ_quad) < radius + 1.0e-12

    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), truevol, rtol = 1.0e-5)
    @test isapprox(Inti.integrate(x -> 1, Γₕ_quad), truearea, rtol = 1.0e-5)

    # the divergence theorem holds only if the curved mesh is watertight, i.e. if
    # elements curved through different charts agree on their shared faces
    F = (x) -> x / 3
    @test isapprox(
        Inti.integrate(q -> dot(F(q.coords), q.normal), Γₕ_quad),
        truevol,
        rtol = 1.0e-5,
    )

    @testset "refinement improves the approximation" begin
        Inti.clear_entities!()
        Ω2, msh2 = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = radius, meshsize = 0.25)
        Γ2 = Inti.external_boundary(Ω2)
        crv2 = Inti.curve_mesh(msh2, Inti.sphere_atlas(; radius = radius), θ)
        q1 = Inti.Quadrature(crvmsh[Γ]; qorder = qorder)
        q2 = Inti.Quadrature(crv2[Γ2]; qorder = qorder)
        e1 = abs(Inti.integrate(x -> 1, q1) - truearea)
        e2 = abs(Inti.integrate(x -> 1, q2) - truearea)
        @test e2 < e1 / 4
    end

    @testset "atlas must cover every element" begin
        # charts too small to contain elements straddling two sextants
        bad = Inti.sphere_atlas(; radius = radius, overlap = 1.02)
        @test_throws ErrorException Inti.curve_mesh(msh, bad, θ)
    end
end

# A surface which is a deformation of the sphere is charted by pushing the
# cube-sphere charts through the deformation; the transition maps are then those
# of the sphere, so the atlas stays compatible. The bean is such a surface.
@testset "Multi-patch curved mesh (bean)" begin
    # Reference values, computed independently of `curve_mesh`: the area by
    # integrating the area element of the six exact patches over [-1,1]², and the
    # volume from det(∇G) = A(z)B(z), which depends only on z. The same area
    # routine reproduces 4π for the undeformed sphere.
    refarea = 9.917518512
    refvol = 2.4605164616

    atlas = Inti.bean_atlas()
    @test length(atlas) == 6
    for c in atlas, α in (SVector(0.0, 0.0), SVector(1.1, -0.7), SVector(-2.2, 2.0))
        @test c.inverse(SVector{3, Float64}(c.param(α))) ≈ α
    end

    # mesh the unit ball and deform it into the bean; the boundary nodes then lie
    # exactly on the bean surface, and `ElementIterator` reads the nodes lazily so
    # the straight elements follow along
    Inti.clear_entities!()
    Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.4)
    for i in eachindex(msh.nodes)
        msh.nodes[i] = Inti._bean_deformation(msh.nodes[i])
    end
    Γ = Inti.external_boundary(Ω)

    crvmsh = Inti.curve_mesh(msh, atlas, 4)
    Ωₕ_quad = Inti.Quadrature(crvmsh[Ω]; qorder = 5)
    Γₕ_quad = Inti.Quadrature(crvmsh[Γ]; qorder = 5)

    # the curved boundary lies on the exact bean surface, i.e. its preimage under
    # the deformation lies on the unit sphere
    @test maximum(
        q -> abs(norm(Inti._bean_deformation_inverse(q.coords)) - 1),
        Γₕ_quad,
    ) < 1.0e-12

    @test isapprox(Inti.integrate(x -> 1, Ωₕ_quad), refvol, rtol = 1.0e-4)
    @test isapprox(Inti.integrate(x -> 1, Γₕ_quad), refarea, rtol = 1.0e-4)
    # watertightness across charts
    @test isapprox(
        Inti.integrate(q -> dot(q.coords, q.normal) / 3, Γₕ_quad),
        refvol,
        rtol = 1.0e-4,
    )

    @testset "tangled elements are detected" begin
        # this mesh is clean
        @test Inti.curve_mesh(msh, atlas, 4; check_jacobian = :error) isa Inti.Mesh
        # this one contains a curved element that folds over itself, which
        # corrupts volume integrals by an amount no quadrature order can fix
        Inti.clear_entities!()
        _, bad = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.25)
        for i in eachindex(bad.nodes)
            bad.nodes[i] = Inti._bean_deformation(bad.nodes[i])
        end
        @test_throws ErrorException Inti.curve_mesh(
            bad, Inti.bean_atlas(), 4; check_jacobian = :error,
        )
        @test_logs (:warn,) Inti.curve_mesh(bad, Inti.bean_atlas(), 4)
        # opting out is silent
        @test_logs Inti.curve_mesh(bad, Inti.bean_atlas(), 4; check_jacobian = :none)
    end

    @testset "numerically inverted charts agree with analytic ones" begin
        numeric = Inti.deformed_sphere_atlas(Inti._bean_deformation)
        @test all(isnothing(c.inverse) for c in numeric)
        crv = Inti.curve_mesh(msh, numeric, 4)
        q = Inti.Quadrature(crv[Γ]; qorder = 5)
        @test isapprox(Inti.integrate(x -> 1, q), refarea, rtol = 1.0e-4)
    end
end

# The general case: an atlas of arbitrary overlapping patches, about which
# nothing is known. Their transition maps are not projective, so the `:chart`
# face map (parametrization ∘ affine map in the parameter plane) disagrees
# between neighbouring elements curved through different charts and the mesh
# cracks. The `:projection` face map is chart-independent and fixes it.
@testset "Arbitrary (incompatible) atlas" begin
    # a bumpy, non-axisymmetric, star-shaped surface r(x̂)x̂
    rfun(x̂) = 1 + 0.25 * x̂[1] * x̂[2] + 0.15 * x̂[3]^3
    fsurf(x̂) = rfun(x̂) * SVector{3}(x̂)
    function fball(x)
        n = norm(x)
        n < 1.0e-14 && return SVector{3, Float64}(x)
        return rfun(x / n) * SVector{3, Float64}(x)
    end
    # reference values from a separate high-order quadrature over the six exact
    # cube-sphere patches (area element, and ∫_{S²} r³/3 for the volume)
    refarea = 12.8859721773
    refvol = 4.2815419879

    # two rotated spherical-coordinate patches of that surface, each smooth on
    # its box and overlapping the other generously
    sphdir(θ, ϕ) = SVector(sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ))
    roty(p) = SVector(p[3], p[2], -p[1])
    atlas = [
        Inti.ParametricChart(
            α -> fsurf(sphdir(α[1], α[2])),
            SVector(0.25, -π), SVector(π - 0.25, π); period = SVector(Inf, 2π),
        ),
        Inti.ParametricChart(
            α -> fsurf(roty(sphdir(α[1], α[2]))),
            SVector(0.25, -π), SVector(π - 0.25, π); period = SVector(Inf, 2π),
        ),
    ]

    Inti.clear_entities!()
    Ω, msh = gmsh_ball(; center = [0.0, 0.0, 0.0], radius = 1.0, meshsize = 0.2)
    for i in eachindex(msh.nodes)
        msh.nodes[i] = fball(msh.nodes[i])
    end
    Γ = Inti.external_boundary(Ω)

    # `:projection` is the default for a multi-chart atlas
    crv = Inti.curve_mesh(msh, atlas, 4)
    Γq = Inti.Quadrature(crv[Γ]; qorder = 5)
    Ωq = Inti.Quadrature(crv[Ω]; qorder = 5)
    @test isapprox(Inti.integrate(x -> 1, Γq), refarea, rtol = 1.0e-6)
    @test isapprox(Inti.integrate(x -> 1, Ωq), refvol, rtol = 1.0e-6)
    # watertight: the divergence theorem holds across the chart interface
    @test isapprox(
        Inti.integrate(q -> dot(q.coords, q.normal) / 3, Γq),
        refvol,
        rtol = 1.0e-6,
    )
    # every curved element lies on the exact surface either way
    @test maximum(q -> abs(norm(q.coords) - rfun(q.coords / norm(q.coords))), Γq) < 1.0e-12

    @testset ":chart is inaccurate for such an atlas" begin
        bad = Inti.curve_mesh(msh, atlas, 4; face_map = :chart)
        q = Inti.Quadrature(bad[Γ]; qorder = 5)
        # elements still lie on the surface, but neighbours disagree on the
        # shared edges, so the total area is badly wrong
        @test maximum(q -> abs(norm(q.coords) - rfun(q.coords / norm(q.coords))), q) < 1.0e-12
        @test abs(Inti.integrate(x -> 1, q) - refarea) / refarea > 1.0e-3
    end

    @testset "more projection iterations do not hurt" begin
        crv2 = Inti.curve_mesh(msh, atlas, 4; projection_iterations = 6)
        q = Inti.Quadrature(crv2[Γ]; qorder = 5)
        @test isapprox(Inti.integrate(x -> 1, q), refarea, rtol = 1.0e-6)
    end
end

# A torus is globally parametrizable, so a single (periodic) chart suffices, and
# passing a bare parametrization must remain equivalent to passing that chart.
@testset "Single chart is a special case of an atlas" begin
    Inti.clear_entities!()
    r1, r2 = 1.0, 0.5
    Ω, msh = gmsh_torus(; center = [0.0, 0.0, 0.0], r1 = r1, r2 = r2, meshsize = 0.2)
    Γ = Inti.external_boundary(Ω)
    ψ = (v) -> SVector(
        (r1 + r2 * sin(v[1])) * cos(v[2]),
        (r1 + r2 * sin(v[1])) * sin(v[2]),
        r2 * cos(v[1]),
    )
    θ = 3
    chart = Inti.ParametricChart(
        ψ,
        SVector(0.0, 0.0),
        SVector(2π, 2π);
        period = SVector(2π, 2π),
    )
    q_fun = Inti.Quadrature(Inti.curve_mesh(msh, ψ, θ)[Ω]; qorder = 5)
    q_atlas = Inti.Quadrature(Inti.curve_mesh(msh, [chart], θ)[Ω]; qorder = 5)
    truevol = 2 * π^2 * r2^2 * r1
    @test Inti.integrate(x -> 1, q_fun) ≈ Inti.integrate(x -> 1, q_atlas)
    @test isapprox(Inti.integrate(x -> 1, q_atlas), truevol, rtol = 1.0e-6)

    # a single chart has no transitions, so `:chart` is used by default; asking
    # for `:projection` must give the same geometry to high accuracy
    q_proj = Inti.Quadrature(
        Inti.curve_mesh(msh, [chart], θ; face_map = :projection)[Ω];
        qorder = 5,
    )
    @test isapprox(Inti.integrate(x -> 1, q_proj), truevol, rtol = 1.0e-6)
end

@testset "Closest point projection stays on its branch" begin
    # The closest point is unique only within the reach of the surface. Past the
    # medial axis the Gauss--Newton iteration can converge to a foot point on
    # another sheet, which would leave a gap against the neighbouring face; the
    # excursion bound is what prevents that.
    c = Inti.sphere_atlas()[1]
    P = α -> SVector{3}(c.param(SVector{2}(α)))
    mid = (c.lc + c.hc) / 2
    cp = (y, α₀, n, ex) ->
    Inti._closest_point_on_chart(c, SVector{3}(y), SVector{2}(α₀), n, ex)

    @testset "the bound does not bind on healthy input" begin
        # An asymmetric chord, so that α₀ is not already a critical point of the
        # distance and the iteration genuinely has to move.
        for h in (0.4, 0.2, 0.1, 0.05)
            α₁ = mid + SVector(0.31, -0.17)
            α₂ = α₁ + SVector(h, 0.4h)
            y = (P(α₁) + P(α₂)) / 2          # chord midpoint, inside the sphere
            α₀ = (α₁ + α₂) / 2               # the O(h²)-accurate starting guess
            ρ = 2 * norm(α₂ - α₁)
            # bounded and unbounded iterations must agree exactly, and the
            # result must land on the sphere
            @test cp(y, α₀, 12, ρ) == cp(y, α₀, 12, Inf)
            @test norm(cp(y, α₀, 12, ρ)) ≈ 1 atol = 1.0e-14
        end
    end

    @testset "the bound holds the branch when the closest point is not unique" begin
        α₀ = mid + SVector(0.31, -0.17)
        ρ = 2 * 0.28
        n̂ = P(α₀) / norm(P(α₀))
        # y on the far side of the sphere, off the radial line through ψ(α₀):
        # the true closest point is not on this chart at all
        y = -0.9 * n̂ + SVector(0.0, 0.35, -0.2)
        near, far = cp(y, α₀, 20, ρ), cp(y, α₀, 20, Inf)
        @test norm(far - P(α₀)) > 1          # unbounded: walks off the branch
        @test norm(near - P(α₀)) < 0.5       # bounded: stays local
        @test norm(near) ≈ 1 atol = 1.0e-14  # and still lands on the surface
    end
end

@testset "Curved faces far from their flat face are reported" begin
    far = [(7, 0.81), (12, 3.4), (19, 0.62)]
    @test isnothing(Inti._report_far_faces(far, :none))
    @test isnothing(Inti._report_far_faces(Tuple{Int, Float64}[], :error))
    @test_logs (:warn,) Inti._report_far_faces(far, :warn)
    # the message must name the worst offender, not merely count them
    e = try
        Inti._report_far_faces(far, :error)
        nothing
    catch err
        err
    end
    @test e isa ErrorException
    @test occursin("element 12", e.msg)
end

@testset "Second-order projection step" begin
    c = Inti.sphere_atlas()[1]
    ψ = c.param
    P = α -> SVector{3}(ψ(SVector{2}(α)))
    mid = (c.lc + c.hc) / 2

    @testset "it converges faster than Gauss--Newton" begin
        # an asymmetric chord, so the start is not already a critical point
        h = 0.4
        α₁ = mid + SVector(0.31, -0.17)
        α₂ = α₁ + SVector(h, 0.4h)
        y = (P(α₁) + P(α₂)) / 2
        α₀ = SVector{2}((α₁ + α₂) / 2)
        ref = Inti._closest_point_on_chart(c, y, α₀, 60, Inf, true)
        err = (n, so) -> norm(Inti._closest_point_on_chart(c, y, α₀, n, Inf, so) - ref)
        # both converge to the same point, and the second-order step gets there
        # in about half the steps: quadratic against linear
        @test err(3, true) < err(3, false)
        @test err(4, true) < 1.0e-14
        @test err(4, false) > 1.0e-10   # Gauss--Newton is still far at n = 4
        @test err(12, false) < 1.0e-13  # but agrees once converged
    end

    @testset "it falls back where the closest point is not unique" begin
        # y at the sphere's centre is on the medial axis: the Hessian there is
        # the offset metric I - d·II with d = 1, which is singular. The step must
        # fall back rather than blow up.
        α₀ = SVector{2}(mid + SVector(0.31, -0.17))
        for y in (SVector(0.0, 0.0, 0.0), SVector(0.02, -0.01, 0.015))
            x = Inti._closest_point_on_chart(c, SVector{3}(y), α₀, 12, 1.0, true)
            @test all(isfinite, x)
            @test norm(x) ≈ 1 atol = 1.0e-12   # still lands on the sphere
        end
    end

    @testset "both methods curve a mesh to the same geometry" begin
        Inti.clear_entities!()
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)
        gmsh.model.occ.addSphere(0, 0, 0, 1.0)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        msh = Inti.import_mesh(; dim = 3)
        Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, Inti.entities(msh))
        gmsh.finalize()
        Γ = Inti.external_boundary(Ω)
        area = m -> Inti.integrate(
            x -> 1,
            Inti.Quadrature(
                Inti.curve_mesh(
                    msh, Inti.sphere_atlas(), 4;
                    face_map = :projection, projection_iterations = 8,
                    projection_method = m, check_jacobian = :none,
                )[Γ];
                qorder = 12,
            ),
        )
        @test isapprox(area(:newton), area(:gauss_newton), rtol = 1.0e-12)
        @test isapprox(area(:newton), 4π, rtol = 1.0e-10)
        @test_throws ErrorException Inti.curve_mesh(
            msh, Inti.sphere_atlas(), 4; projection_method = :bogus,
        )
    end
end
