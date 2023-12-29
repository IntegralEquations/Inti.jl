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
        deps_compat = (; broken = true), # ElementaryPDESolutions not registered
        piracies = (; broken = true), # piracy related to ElementaryPDESolutions
        persistent_tasks = (; broken = true) # not sure why yet
    )
end
