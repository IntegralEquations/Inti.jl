using Inti
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Pluto
using Runic
# packages needed for extensions
using Gmsh
using HMatrices
using Meshes
using GLMakie
using FMM2D
using FMM3D

const ON_CI = get(ENV, "CI", "false") == "true"

draft = false # build docs without running code. Useful for quick local testing
ON_CI && (draft = false) # always full build on CI

# from https://github.com/fonsp/Pluto.jl/pull/2471
function generate_plaintext(
        notebook,
        strmacrotrim::Union{String, Nothing} = nothing;
        header::Function = _ -> nothing,
        footer::Function = _ -> nothing,
        textcomment::Function = identity,
        codewrapper::Function,
    )
    cell_strings = String[]
    header_content = header(notebook)
    isnothing(header_content) || push!(cell_strings, header_content)
    for cell_id in notebook.cell_order
        cell = notebook.cells_dict[cell_id]
        scode = strip(cell.code)
        (raw, ltrim, rtrim) = if isnothing(strmacrotrim)
            false, 0, 0
        elseif startswith(scode, string(strmacrotrim, '"'^3))
            true, length(strmacrotrim) + 3, 3
        elseif startswith(scode, string(strmacrotrim, '"'))
            true, length(strmacrotrim) + 1, 1
        else
            false, 0, 0
        end
        push!(
            cell_strings,
            if raw
                text = strip(
                    scode[nextind(scode, 1, ltrim):prevind(scode, end, rtrim)],
                    ['\n'],
                )
                ifelse(Pluto.is_disabled(cell), textcomment, identity)(text)
            else
                codewrapper(cell, Pluto.is_disabled(cell))
            end,
        )
    end
    footer_content = footer(notebook)
    isnothing(footer_content) || push!(cell_strings, footer_content)
    return join(cell_strings, "\n\n")
end

function generate_md(input; output = replace(input, r"\.jl$" => ".md"))
    fname = basename(input)
    notebook = Pluto.load_notebook(input)
    header =
        _ ->
    "[![Pluto notebook](https://img.shields.io/badge/download-Pluto_notebook-blue)]($fname)"

    function codewrapper(cell, _)
        # 1. Strips begin/end block
        # 2. Reformats code using Runic
        # 3. Wraps all code in same ```@example``` block for documenter
        code = strip(cell.code)
        if startswith(code, "begin") && endswith(code, "end")
            code = strip(code[6:(end - 4)])  # Remove "begin" and "end" and strip spaces
            # reformat code using Runic
            code = Runic.format_string(String(code))
        end
        return if cell.code_folded
            string("```@setup $fname\n", code, "\n```")
        else
            string("```@example $fname\n", code, "\n```")
        end
    end
    textcomment(text) = string("<!-- ", text, " -->")
    str = generate_plaintext(notebook, "md"; header, codewrapper, textcomment)

    open(output, "w") do io
        return write(io, str)
    end
    return output
end

links = InterLinks(
    "Meshes" => "https://juliageometry.github.io/MeshesDocs/dev/objects.inv",
    "HMatrices" => "https://integralequations.github.io/HMatrices.jl/stable/objects.inv",
)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

println("\n*** Generating documentation")

DocMeta.setdocmeta!(Inti, :DocTestSetup, :(using Inti); recursive = true)

modules = [Inti]
for extension in
    [:IntiGmshExt, :IntiHMatricesExt, :IntiMakieExt, :IntiFMM2DExt, :IntiFMM3DExt]
    ext = Base.get_extension(Inti, extension)
    isnothing(ext) && "error loading $ext"
    push!(modules, ext)
end

size_threshold_ignore = []
notebooks = [
    "Toy example" => "toy_example.jl",
    "Helmholtz scattering" => "helmholtz_scattering.jl",
    "Poisson problem" => "poisson.jl",
]

notebook_examples = Pair{String, String}[]
for notebook in notebooks
    title, fname = notebook
    file_in = joinpath(@__DIR__, "src", "pluto-examples", fname)
    file_out = generate_md(file_in)
    push!(notebook_examples, title => joinpath("pluto-examples", basename(file_out)))
end
push!(notebook_examples, "Heat equation" => joinpath("examples", "heat_equation.md"))
push!(notebook_examples, "Stokes drag" => joinpath("examples", "stokes_drag.md"))
push!(notebook_examples, "Elastic crack" => joinpath("examples", "crack_elasticity.md"))
push!(notebook_examples, "Plasmonic eigenvalues" => joinpath("examples", "pep.md"))
size_threshold_ignore = last.(notebook_examples)

makedocs(;
    modules = modules,
    repo = "",
    sitename = "Inti.jl",
    format = Documenter.HTML(;
        prettyurls = ON_CI,
        canonical = "https://IntegralEquations.github.io/Inti.jl",
        size_threshold = 2 * 2^20, # 2 MiB
        size_threshold_warn = 1 * 2^20, # 1 MiB
        sidebar_sitename = false,
        mathengine = Documenter.KaTeX(
            Dict(
                :delimiters => [
                    Dict(:left => raw"$", :right => raw"$", display => false),
                    Dict(:left => raw"$$", :right => raw"$$", display => true),
                    Dict(:left => raw"\[", :right => raw"\]", display => true),
                ],
                :macros => Dict(
                    "\\RR" => "\\mathbb{R}",
                    "\\CC" => "\\mathbb{C}",
                    "\\bx" => "\\boldsymbol{x}",
                    "\\by" => "\\boldsymbol{y}",
                ),
            ),
        ),
        size_threshold_ignore,
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
        "Examples" => notebook_examples,
        "References" => "references.md",
        "Docstrings" => "docstrings.md",
    ],
    warnonly = ON_CI ? false : Documenter.except(:linkcheck_remotes),
    # warnonly = true,
    pagesonly = true,
    checkdocs = :none,
    clean = true,
    draft,
    plugins = [bib, links],
)

deploydocs(;
    repo = "github.com/IntegralEquations/Inti.jl",
    devbranch = "main",
    push_preview = true,
)

GLMakie.closeall()
