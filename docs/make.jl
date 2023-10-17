using Inti
using Documenter
using Literate
# packages needed for extensions
using Gmsh
using WriteVTK
using CairoMakie
using HMatrices

# Generate examples using Literate
const examples_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples")
const generated_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples", "generated")
example = "mock_example.jl"
for example in ["mock_example.jl"]
    src = joinpath(examples_dir, example)
    Literate.markdown(src, generated_dir; mdstrings = true)
    Literate.notebook(src, generated_dir; mdstrings = true)
end

## setup documentation config
DocMeta.setdocmeta!(Inti, :DocTestSetup, :(using Inti); recursive = true)

modules = [Inti]
for extension in [:IntiGmshExt, :IntiMakieExt, :IntiVTKExt, :IntiHMatricesExt]
    ext = Base.get_extension(Inti, extension)
    isnothing(ext) && "error loading $ext"
    push!(modules, ext)
end

# some settings are only active on a CI build
on_CI = get(ENV, "CI", "false") == "true"

makedocs(;
    modules = modules,
    repo = "",
    sitename = "Inti.jl",
    format = Documenter.HTML(;
        prettyurls = on_CI,
        canonical = "https://IntegralEquations.github.io/Inti.jl",
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Meshing" => "geo_and_meshes.md",
        "Examples" =>[
            "examples/helmholtz_soundsoft_scattering_circle.md"
        ],
        "References" => "references.md",
    ],
    warnonly = on_CI ? false : Documenter.except(:linkcheck_remotes),
    pagesonly = true,
    draft = draft,
)

deploydocs(;
    repo = "github.com/IntegralEquations/Inti.jl",
    devbranch = "main",
    push_preview = false,
)
