using Test
using Revise
using Inti
using StaticArrays
using Gmsh
using LinearAlgebra
using GLMakie

@testset "Laurent coefficients" begin
	f = ρ -> ρ^2 + 2ρ + 1
	f₋₂, f₋₁ = Inti.laurent_coefficients(f, Val(2))
	@test norm(f₋₂) < 1e-10
	@test norm(f₋₁) < 1e-10
	f = ρ -> cos(ρ) / ρ^2 + exp(ρ) / ρ
	f₋₂, f₋₁ = Inti.laurent_coefficients(f, Val(2))
	@test f₋₂ ≈ 1.0
	@test f₋₁ ≈ 1.0

	f = ρ -> SVector(cos(ρ), sin(ρ)) / ρ^2 + SVector(exp(ρ), 0.2) / ρ
	f₋₂, f₋₁ = Inti.laurent_coefficients(f, Val(2))
	@test f₋₂ ≈ SVector(1.0, 0.0)
	@test f₋₁ ≈ SVector(1.0, 1.2)

	f = ρ -> π / ρ
	f₋₂, f₋₁ = Inti.laurent_coefficients(f, Val(2))

	## Laplace kernel
	v1 = SVector(0.0, 0.0, 0.0)
	v2 = SVector(1.0, 0.0, 0.0)
	v3 = SVector(1.0, 1.0, 0.0)
	v4 = SVector(0.0, 1.0, 0.0)

	el = Inti.LagrangeSquare(v1, v2, v3, v4)

	K = Inti.HyperSingularKernel(Inti.Laplace(; dim = 3))
	x̂ = SVector(0.5, 0.2)
	x = el(x̂)
	nx = Inti.normal(el, x̂)
	qx = (coords = x, normal = nx)

	F = let K = K, qx = qx, el = el, x̂ = x̂
		(ρ, θ) -> begin
			s, c = sincos(θ)
			ŷ = x̂ + ρ * SVector(c, s)
			y = el(ŷ)
			jac = Inti.jacobian(el, ŷ)
			ny = Inti._normal(jac)
			μ = Inti._integration_measure(jac)
			qy = (coords = y, normal = ny)
			ρ * K(qx, qy) * μ
			# qy = (coords = x, normal = nx)
			# ρ * K(qx, qx)
		end
	end
	g = let F = F
		(ρ) -> F(ρ, 1)
	end
	Inti.laurent_coefficients(g, (Val(2)), 1e-3)
end

@testset "Polar decomposition - Reference square" begin
	# FIXME: write a test for the polar decomposition. E.g. test that it a quadrature in
	# rho/theta correctly integrates all some function over the square.
	ref = Inti.ReferenceSquare()
	x̂  = SVector(0.2, 0.5)

	quad_rho                 = Inti.GaussLegendre(; order = 10)
	quad_theta               = Inti.GaussLegendre(; order = 10)
	x_rho_ref, w_rho_ref     = Inti.qcoords(quad_rho), Inti.qweights(quad_rho) # nodes and weights on [0,1]
	x_theta_ref, w_theta_ref = Inti.qcoords(quad_theta), Inti.qweights(quad_theta) # nodes and weights on [0,2π]
	n_rho, n_theta           = length(x_rho_ref), length(x_theta_ref)
	fig                      = Figure()
	ax                       = Axis(fig[1, 1])
	for (θmin, θmax, ρ) in Inti.polar_decomposition(ref, x̂)
		@test θmin ≤ θmax
		x = SVector{2, Float64}[]
		for θ in θmin:0.1:θmax
			ρmax = ρ(θ)
			# @show θ, ρmax
			for ρ in 0:0.1:ρmax
				push!(x, x̂ + ρ * SVector(cos(θ), sin(θ)))
			end
		end
		scatter!(ax, x; label = "")
		# draw the line from x to x + ρmax * [cos(θ), sin(θ)]
		x = x̂ + ρ(θmin) * SVector(cos(θmin), sin(θmin))
		lines!(ax, [x̂, x]; label = "")
	end
	res = 0.0
	function _function_to_integrate_1(ρ, θ)
		ρ
	end
	for (theta_min, theta_max, rho) in Inti.polar_decomposition(ref, x̂) # loop over the four triangles
		delta_theta = theta_max - theta_min
		for m in 1:n_theta
			theta = theta_min + x_theta_ref[m][1] * delta_theta
			w_theta = w_theta_ref[m] * delta_theta
			rho_max = rho(theta)::Float64
			for n in 1:n_rho
				ρₙ = x_rho_ref[n][1] * rho_max
				w_rho = w_rho_ref[n] * rho_max
				res += _function_to_integrate_1(ρₙ, theta) * w_theta * w_rho
			end
		end
	end
	@test isapprox(res, 1.0, atol = 1e-2)
	# fig
end

@testset "Polar decomposition - Reference triangle" begin
	ref = Inti.ReferenceTriangle()
	x̂  = SVector(0.2, 0.5)

	quad_rho                 = Inti.GaussLegendre(; order = 10)
	quad_theta               = Inti.GaussLegendre(; order = 10)
	x_rho_ref, w_rho_ref     = Inti.qcoords(quad_rho), Inti.qweights(quad_rho) # nodes and weights on [0,1]
	x_theta_ref, w_theta_ref = Inti.qcoords(quad_theta), Inti.qweights(quad_theta) # nodes and weights on [0,2π]
	n_rho, n_theta           = length(x_rho_ref), length(x_theta_ref)
	fig                      = Figure()
	ax                       = Axis(fig[1, 1])
	for (θmin, θmax, ρ) in Inti.polar_decomposition(ref, x̂)
		@test θmin ≤ θmax
		x = SVector{2, Float64}[]
		for θ in θmin:0.01:θmax
			ρmax = ρ(θ)
			# @show θ, ρmax
			for ρ in 0:0.1:ρmax
				push!(x, x̂ + ρ * SVector(cos(θ), sin(θ)))
			end
		end
		scatter!(ax, x; label = "")
		# draw the line from x to x + ρmax * [cos(θ), sin(θ)]
		x = x̂ + ρ(θmin) * SVector(cos(θmin), sin(θmin))
		lines!(ax, [x̂, x]; label = "")
	end
	res1 = 0.0
	function _function_to_integrate_1(ρ, θ)
		ρ
	end
	for (theta_min, theta_max, rho) in Inti.polar_decomposition(ref, x̂) # loop over the four triangles
		delta_theta = theta_max - theta_min
		for m in 1:n_theta
			theta = theta_min + x_theta_ref[m][1] * delta_theta
			w_theta = w_theta_ref[m] * delta_theta
			rho_max = rho(theta)::Float64
			for n in 1:n_rho
				ρₙ = x_rho_ref[n][1] * rho_max
				w_rho = w_rho_ref[n] * rho_max
				res1 += _function_to_integrate_1(ρₙ, theta) * w_theta * w_rho
			end
		end
	end
	@test isapprox(res1, 1 / 2, atol = 1e-2)
	# fig
end

@testset "Plane distorted element" begin
	# Guiggiani plane distoreted element (table 1)
	δ          = 0.5
	z          = 0.0
	y¹         = SVector(-1.0, -1.0, z)
	y²         = SVector(1.0 + δ, -1.0, z)
	y³         = SVector(1.0 - δ, 1.0, z)
	y⁴         = SVector(-1.0, 1.0, z)
	nodes      = (y¹, y², y³, y⁴)
	el         = Inti.LagrangeSquare(nodes)
	K          = (p, q) -> begin
		x = Inti.coords(p)
		y = Inti.coords(q)
		d = norm(x - y)
		1 / (d^3)
	end
	û         = (x̂) -> 1
	a          = SVector(0.5, 0.5)
	b          = SVector(1.66 / 2, 0.5)
	quad_rho   = Inti.GaussLegendre(; order = 10)
	quad_theta = Inti.GaussLegendre(; order = 20)
	va         = Inti.guiggiani_singular_integral(K, û, a, el, quad_rho, quad_theta)
	vb         = Inti.guiggiani_singular_integral(K, û, b, el, quad_rho, quad_theta)
	@test isapprox(va, -5.749237; atol = 1e-4)
	@test isapprox(vb, -9.154585; atol = 1e-4)
	# TODO: add point c and more tests from table 2
end

@testset "Green's identity" begin
	## create a mesh and quadrature
	meshsize = 0.4
	gmsh.initialize(String[], false)
	gmsh.option.setNumber("General.Verbosity", 2)
	gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
	gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
	gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
	gmsh.model.occ.synchronize()
	gmsh.model.mesh.generate(2)
	gmsh.model.mesh.recombine()
	msh = Inti.import_mesh(; dim = 3)
	gmsh.finalize()
	Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
	Γ_msh = msh[Γ]
	Γ_quad = Inti.Quadrature(Γ_msh; qorder = 2)

	##
	op = Inti.Laplace(; dim = 3)
	K′ = Inti.AdjointDoubleLayerKernel(op)
	T = Inti.HyperSingularKernel(op)

	K′_op = Inti.IntegralOperator(K′, Γ_quad, Γ_quad)
	T_op = Inti.IntegralOperator(T, Γ_quad, Γ_quad)

	K′₀ = Inti.assemble_matrix(K′_op)
	T₀ = Inti.assemble_matrix(T_op)

	δK′ = Inti.guiggiani_correction(
		K′_op;
		nearfield_distance = 3 * meshsize,
		nearfield_qorder = 40,
	)
	K′new = K′₀ + δK′

	δT = Inti.guiggiani_correction(
		T_op;
		nearfield_distance = 3 * meshsize,
		nearfield_qorder = 60,
	)
	Tnew = T₀ + δT

	U = ones(Float64, size(T_op, 1))
	γ₁ₓU = zeros(Float64, size(K′_op, 1))
	lhs = γ₁ₓU
	rhs = K′new * γ₁ₓU - Tnew * U
	# @test norm(rhs - lhs) < 1e-6

	U₀ = SVector(ones(Float64, 3)...)
	u₁ = 1.5
	U = [transpose(U₀) * node.coords + u₁ for node in Γ_quad]
	Tₙ = [transpose(U₀) * node.normal for node in Γ_quad]

	lhs = Tₙ / 2
	rhs = K′new * Tₙ - Tnew * U
	@test norm(rhs - lhs) < 1e-2

	##
	op = Inti.Elastostatic(; dim = 3, μ = 1, λ = 1)
	K = Inti.HyperSingularKernel(op)
	T = Inti.IntegralOperator(K, Γ_quad, Γ_quad)
	T₀ = Inti.assemble_matrix(T)
	δT = Inti.guiggiani_correction(
		T;
		nearfield_distance = 3 * meshsize,
		nearfield_qorder = 60,
	)
	Tnew = T₀ + δT
	rhs = [SVector(1.0, 1.0, 1.0) for _ in 1:size(T, 1)]
	@test norm(Tnew * rhs, Inf) < 1e-2

	K = Inti.DoubleLayerKernel(op)
	T = Inti.IntegralOperator(K, Γ_quad, Γ_quad)
	T₀ = Inti.assemble_matrix(T)
	δT = Inti.guiggiani_correction(
		T;
		nearfield_distance = 3 * meshsize,
		nearfield_qorder = 40,
	)
	Tnew = T₀ + δT
	rhs = [SVector(1.0, 1.0, 1.0) for _ in 1:size(T, 1)]
	@test norm(Tnew * rhs + 0.5 * rhs, Inf) < 1e-2

	op = Inti.Elastostatic(; dim = 3, μ = 1, λ = 1)
end
