using Inti
using Documenter
using Literate
# packages needed for extensions
using Gmsh
using HMatrices
using Meshes
using FMM2D
using FMM3D

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
# const examples = ["helmholtz_scattering.jl", "poisson.jl"]
# for example in examples
#     println("\n*** Generating $example example")
#     @time begin
#         src = joinpath(examples_dir, example)
#         Literate.markdown(src, generated_dir; mdstrings = true)
#         # if draft, skip creation of notebooks
#         Literate.notebook(
#             src,
#             generated_dir;
#             mdstrings = true,
#             preprocess = insert_setup,
#             # execute = !draft,
#             execute = false,
#         )
#     end
# end

println("\n*** Generating documentation")

DocMeta.setdocmeta!(Inti, :DocTestSetup, :(using Inti); recursive = true)

modules = [Inti, Meshes]
for extension in
    [:IntiGmshExt, :IntiHMatricesExt, :IntiMeshesExt, :IntiFMM2DExt, :IntiFMM3DExt]
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
        # "Meshing" => "geo_and_meshes.md",
        # "Toy example" => "examples/generated/toy_example.md",
        # "Helmholtz Example" => ["examples/generated/helmholtz_scattering.md"],
        # "Poisson Example" => ["examples/generated/poisson.md"],
        "References" => "references.md",
    ],
    # warnonly = ON_CI ? false : Documenter.except(:linkcheck_remotes),
    warnonly = true,
    pagesonly = true,
    checkdocs = :none,
    draft,
)

deploydocs(;
    repo = "github.com/IntegralEquations/Inti.jl",
    devbranch = "main",
    push_preview = true,
)
