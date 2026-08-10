using Inti
using Gmsh

function gmsh_disk(; center, rx, ry, meshsize)
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("disk")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addDisk(center[1], center[2], 0, rx, ry)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        Inti.import_mesh(; dim = 2)
    finally
        gmsh.finalize()
    end
    Ω = Inti.Domain(Inti.entities(msh)) do e
        return Inti.geometric_dimension(e) == 2
    end
    return Ω, msh
end

function gmsh_ball(; center, radius, meshsize)
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("ball")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addSphere(center[1], center[2], center[3], radius)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    Ω = Inti.Domain(Inti.entities(msh)) do e
        return Inti.geometric_dimension(e) == 3
    end
    return Ω, msh
end

function gmsh_torus(; center, r1, r2, meshsize)
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("ball")
        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addTorus(center[1], center[2], center[3], r1, r2)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    Ω = Inti.Domain(Inti.entities(msh)) do e
        return Inti.geometric_dimension(e) == 3
    end
    return Ω, msh
end

function gmsh_cut_ball(; center, radius, meshsize, cutelevation)
    msh = try
        xmin = -1.1 * radius
        ymin = -1.1 * radius
        xmax = 1.1 * radius
        ymax = 1.1 * radius
        zmin = -(1 - cutelevation) * radius
        zmax = (1 - cutelevation) * radius
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("ball")
        # Construct the intersecting box
        p1 = gmsh.model.occ.addPoint(xmin, ymax, zmin)
        p2 = gmsh.model.occ.addPoint(xmax, ymax, zmin)
        p3 = gmsh.model.occ.addPoint(xmax, ymin, zmin)
        p4 = gmsh.model.occ.addPoint(xmin, ymin, zmin)
        p5 = gmsh.model.occ.addPoint(xmin, ymax, zmax)
        p6 = gmsh.model.occ.addPoint(xmax, ymax, zmax)
        p7 = gmsh.model.occ.addPoint(xmax, ymin, zmax)
        p8 = gmsh.model.occ.addPoint(xmin, ymin, zmax)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        l5 = gmsh.model.occ.addLine(p5, p6)
        l6 = gmsh.model.occ.addLine(p6, p7)
        l7 = gmsh.model.occ.addLine(p7, p8)
        l8 = gmsh.model.occ.addLine(p8, p5)
        l9 = gmsh.model.occ.addLine(p1, p5)
        l10 = gmsh.model.occ.addLine(p2, p6)
        l11 = gmsh.model.occ.addLine(p3, p7)
        l12 = gmsh.model.occ.addLine(p4, p8)
        cl1 = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        cl2 = gmsh.model.occ.addCurveLoop([l5, l6, l7, l8])
        cl3 = gmsh.model.occ.addCurveLoop([l1, l10, l5, l9])
        cl4 = gmsh.model.occ.addCurveLoop([l3, l11, l7, l12])
        cl5 = gmsh.model.occ.addCurveLoop([l10, l2, l11, l6])
        cl6 = gmsh.model.occ.addCurveLoop([l4, l9, l8, l12])
        f1 = gmsh.model.occ.addPlaneSurface([cl1])
        f2 = gmsh.model.occ.addPlaneSurface([cl2])
        f3 = gmsh.model.occ.addPlaneSurface([cl3])
        f4 = gmsh.model.occ.addPlaneSurface([cl4])
        f5 = gmsh.model.occ.addPlaneSurface([cl5])
        f6 = gmsh.model.occ.addPlaneSurface([cl6])
        loo = gmsh.model.occ.addSurfaceLoop([f1, f2, f3, f4, f5, f6])
        vol = gmsh.model.occ.addVolume([loo])
        sph = gmsh.model.occ.addSphere(center[1], center[2], center[3], radius)
        gmsh.model.occ.intersect([3, vol], [3, sph])

        # set max and min meshsize to meshsize
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    Ω = Inti.Domain(Inti.entities(msh)) do e
        return Inti.geometric_dimension(e) == 3
    end
    return Ω, msh
end

using StaticArrays

# Manufactured `(f, u, γ₁u)` triples with `ℒu = f`, in the per-index shape these tests use,
# from the same `particular_basis` the corrections are built on. That is not circular: the
# Green identity `μu + D[γ₀u] - S[γ₁u] = V[f]` holds for *any* particular solution, so what a
# reference has to be is internally consistent, not canonical — and that consistency is
# pinned independently, by automatic differentiation, in `particular_basis_test.jl`.
#
# `source` is scalar for every operator (see `particular_basis`); the vector-valued cases
# recover their tensor by multiplying it with a direction, exactly as before.
function manufactured_basis(op, order)
    N = Inti.ambient_dimension(op)
    pb = Inti.particular_basis(op, order)
    c, r = zero(SVector{N, Float64}), 1.0
    b, Ψ, γ₁Ψ = pb(c, r)
    ∇Ψ = Inti.gradient_solution(pb, c, r)
    return map(1:length(pb)) do k
        (
            source = q -> b(Inti.coords(q))[k],
            solution = q -> Ψ(Inti.coords(q))[k],
            neumann_trace = q -> γ₁Ψ(Inti.coords(q), Inti.normal(q))[k],
            gradient_solution = q -> ∇Ψ(Inti.coords(q))[k],
        )
    end
end

# The `W` counterpart: `ℒΨₐⱼ = ∂ⱼpₐ` bundled over the density directions `j`, with the
# `+ pₐνⱼ` of the (3.11) regularization folded into the trace.
function manufactured_basis_W(op, order)
    N = Inti.ambient_dimension(op)
    pb = Inti.particular_basis(op, order)
    c, r = zero(SVector{N, Float64}), 1.0
    b = first(pb(c, r))
    pbj = ntuple(j -> Inti.source_derivative(pb, j)(c, r), N)
    return map(1:length(pb)) do k
        (
            source = q -> b(Inti.coords(q))[k],
            solution = q -> SVector(ntuple(j -> pbj[j][2](Inti.coords(q))[k], N)),
            neumann_trace = q -> SVector(
                ntuple(N) do j
                    x, ν = Inti.coords(q), Inti.normal(q)
                    pbj[j][3](x, ν)[k] + b(x)[k] * ν[j]
                end,
            ),
        )
    end
end

# The `X` counterpart: `Υₐⱼ = ∇Ψₐⱼ` as the columns of an `SMatrix`, plus the two traces of
# (3.12) — `(∂ⱼpₐ)ν + (HessΨₐⱼ)·ν` and `pₐν`.
function manufactured_basis_X(op, order)
    N = Inti.ambient_dimension(op)
    pb = Inti.particular_basis(op, order)
    c, r = zero(SVector{N, Float64}), 1.0
    b = first(pb(c, r))
    Υ = ntuple(j -> Inti.gradient_solution(Inti.source_derivative(pb, j), c, r), N)
    # `∂ν` of each component of `Υⱼ`, i.e. the rows of the Hessian contracted with `ν`
    BνΥ = ntuple(
        j -> ntuple(d -> last(Inti.solution_derivative(Inti.source_derivative(pb, j), d)(c, r)), N),
        N,
    )
    ∂p = ntuple(j -> Inti.derivative_matrix(Inti.source_space(pb), j), N)
    m = Inti.source_space(pb)
    return map(1:length(pb)) do k
        (
            source = q -> b(Inti.coords(q))[k],
            solution = q -> reduce(hcat, ntuple(j -> Υ[j](Inti.coords(q))[k], N)),
            single_trace = q -> begin
                x, ν = Inti.coords(q), Inti.normal(q)
                ∂pv = ntuple(j -> (∂p[j] * m(x))[k], N)
                reduce(
                    hcat,
                    ntuple(j -> SVector(ntuple(d -> ∂pv[j] * ν[d] + BνΥ[j][d](x, ν)[k], N)), N),
                )
            end,
            grad_single_trace = q -> b(Inti.coords(q))[k] * Inti.normal(q),
        )
    end
end
