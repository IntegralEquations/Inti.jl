using Inti
using Test
using Aqua

@testset "Aqua" begin
    Aqua.test_all(
        Inti;
        ambiguities = true,
        unbound_args = true,
        undefined_exports = true,
        project_extras = true,
        stale_deps = true,
        deps_compat = (; broken = true),
        piracies = (; broken = true),
    )
end
