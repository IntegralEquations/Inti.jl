using Inti
using Documenter
using Literate
# packages needed for extensions
using Gmsh
using HMatrices

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
    replace(content, "#nb ## __NOTEBOOK_SETUP__" => SETUP)
end

# Generate examples using Literate
const examples_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples")
const generated_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples", "generated")
for example in ["helmholtz_scattering.jl", "poisson.jl", "transmission.jl", "lippmann_schwinger.jl"]
    println("\n*** Generating $example example")
    @time begin
        src = joinpath(examples_dir, example)
        Literate.markdown(src, generated_dir; mdstrings = true)
        # if on CI, also generate the notebook
        ON_CI && Literate.notebook(src, generated_dir; mdstrings = true, preprocess = insert_setup)
    end
end

println("\n*** Generating documentation")

DocMeta.setdocmeta!(Inti, :DocTestSetup, :(using Inti); recursive = true)

modules = [Inti]
for extension in [:IntiGmshExt, :IntiHMatricesExt]
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
        # size_threshold = 2*2^20, # 2 MiB
    ),
    pages = [
        "Home" => "index.md",
        # "Meshing" => "geo_and_meshes.md",
        "Helmholtz Example" => ["examples/generated/helmholtz_scattering.md"],
        "Poisson Example" => ["examples/generated/poisson.md"],
        "Acoustic Transmission Example" => ["examples/generated/transmission.md"],
        "Lippmann Schwinger Example" => ["examples/generated/lippmann_schwinger.md"],
        "References" => "references.md",
    ],
    warnonly = ON_CI ? false : Documenter.except(:linkcheck_remotes),
    pagesonly = true,
    draft,
)

deploydocs(;
    repo = "github.com/IntegralEquations/Inti.jl",
    devbranch = "main",
    push_preview = true,
)
