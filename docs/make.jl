using Inti
using Documenter

DocMeta.setdocmeta!(Inti, :DocTestSetup, :(using Inti); recursive=true)

makedocs(;
    modules=[Inti],
    authors="Luiz M. Faria",
    repo="",
    sitename="Inti.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://IntegralEquations.github.io/Inti.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Meshing" => "geo_and_meshes.md",
    ],
    pagesonly = true,
)

deploydocs(;
    repo="github.com/IntegralEquations/Inti.jl",
    devbranch="main",
)
