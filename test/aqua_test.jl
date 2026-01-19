using Inti
using Test
using Aqua

@testset "Aqua" begin
    Aqua.test_all(
        Inti;
        ambiguities = false, # test only `Inti` for ambiguities later
        unbound_args = (; broken = true), # broken due to use of NTuple in some signatures
        undefined_exports = true,
        project_extras = true,
        stale_deps = true,
        deps_compat = true,
        piracies = (; broken = true), # piracy related to ElementaryPDESolutions
        persistent_tasks = (; broken = false), # fixed?
    )
    Aqua.test_ambiguities(Inti)
end
