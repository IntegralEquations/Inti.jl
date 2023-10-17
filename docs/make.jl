using Inti
using Documenter
using Literate
# packages needed for extensions
using Gmsh
using WriteVTK
using CairoMakie
using HMatrices

# some settings are only active on a CI build
on_CI = get(ENV, "CI", "false") == "true"

# draft mode is used to quickly test the documentation's text locally
draft = true

on_CI && (draft = false) # if on CI, never draft

# Generate examples using Literate
const examples_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples")
const generated_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples", "generated")
example = "mock_example.jl"
for example in ["mock_example.jl", "sphere_scattering.jl"]
    src = joinpath(examples_dir, example)
    Literate.markdown(src, generated_dir; mdstrings = true)
    draft || Literate.notebook(src, generated_dir; mdstrings = true)
end

## setup documentation config
DocMeta.setdocmeta!(Inti, :DocTestSetup, :(using Inti); recursive = true)

modules = [Inti]
for extension in [:IntiGmshExt, :IntiMakieExt, :IntiVTKExt, :IntiHMatricesExt]
    ext = Base.get_extension(Inti, extension)
    isnothing(ext) && "error loading $ext"
    push!(modules, ext)
end



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
        "Examples" => [
            "examples/generated/mock_example.md"
            "examples/generated/sphere_scattering.md"
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
