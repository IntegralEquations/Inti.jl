using Inti
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Literate
# packages needed for extensions
using Gmsh
using HMatrices
using Meshes
using GLMakie
using FMM2D
using FMM3D

links = InterLinks(
    "Meshes" => "https://juliageometry.github.io/MeshesDocs/dev/objects.inv",
    "HMatrices" => "https://integralequations.github.io/HMatrices.jl/stable/objects.inv",
)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

draft = false

const ON_CI = get(ENV, "CI", "false") == "true"
const GIT_HEAD = chomp(read(`git rev-parse HEAD`, String))
const SETUP = """
#nb import Pkg
#nb Pkg.activate(temp=true)
#nb Pkg.add(url="https://github.com/IntegralEquations/Inti.jl", rev="$GIT_HEAD")
#nb foreach(Pkg.add, DEPENDENCIES)
"""

ON_CI && (draft = false) # always full build on CI

function insert_setup(content)
    ON_CI || return content
    return replace(content, "#nb ## __NOTEBOOK_SETUP__" => SETUP)
end

# Generate examples using Literate
const examples_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples")
const generated_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples", "generated")
const examples = [
    "toy_example.jl",
    "helmholtz_scattering.jl",
    "lippmann_schwinger.jl",
    "poisson.jl",
    "stokes_drag.jl",
]
for t in examples
    println("\n*** Generating $t example")
    @time begin
        src = joinpath(examples_dir, t)
        Literate.markdown(src, generated_dir; mdstrings = true)
        # if draft, skip creation of notebooks
        Literate.notebook(
            src,
            generated_dir;
            mdstrings = true,
            preprocess = insert_setup,
            # execute = ON_CI,
            execute = false,
        )
    end
end

println("\n*** Generating documentation")

DocMeta.setdocmeta!(Inti, :DocTestSetup, :(using Inti); recursive = true)

modules = [Inti]
for extension in
    [:IntiGmshExt, :IntiHMatricesExt, :IntiMakieExt, :IntiFMM2DExt, :IntiFMM3DExt]
    ext = Base.get_extension(Inti, extension)
    isnothing(ext) && "error loading $ext"
    push!(modules, ext)
end

makedocs(;
    modules = modules,
    repo = "",
    sitename = "Inti.jl",
    format = Documenter.HTML(;
        prettyurls = ON_CI,
        canonical = "https://IntegralEquations.github.io/Inti.jl",
        size_threshold = 2 * 2^20, # 2 MiB
        size_threshold_warn = 1 * 2^20, # 1 MiB
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "tutorials/getting_started.md",
            "tutorials/geo_and_meshes.md",
            "tutorials/integral_operators.md",
            "tutorials/layer_potentials.md",
            "tutorials/compression_methods.md",
            "tutorials/correction_methods.md",
            "tutorials/solvers.md",
        ],
        "Examples" => [
            "examples/generated/toy_example.md",
            "examples/generated/helmholtz_scattering.md",
            # "examples/generated/lippmann_schwinger.md",
            # "examples/generated/poisson.md",
            # "examples/generated/stokes_drag.md",
        ],
        "References" => "references.md",
        "Docstrings" => "docstrings.md",
    ],
    warnonly = ON_CI ? false : Documenter.except(:linkcheck_remotes),
    # warnonly = true,
    pagesonly = true,
    checkdocs = :none,
    draft,
    plugins = [bib, links],
)

deploydocs(;
    repo = "github.com/IntegralEquations/Inti.jl",
    devbranch = "main",
    push_preview = true,
)
