using Inti
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using ExampleJuggler, Literate, PlutoStaticHTML, PlutoSliderServer
# packages needed for extensions
using Gmsh
using HMatrices
using Meshes
using GLMakie
using FMM2D
using FMM3D

# Function to remove "begin #hide" and "end #hide" from a markdown file
function formatting_pluto(input_file::String, output_file::String)
    # Read the contents of the file
    file_content = read(input_file, String)

    # Replace the "begin #hide" and "end #hide" with an empty string
    cleaned_content = replace(file_content, r"\b(end #hide)\b" => "")
    cleaned_content = replace(cleaned_content, r"\b(end; #hide)\b" => "")
    cleaned_content = replace(cleaned_content, r"\b(end;#hide)\b" => "")
    cleaned_content = replace(cleaned_content, r"begin #hide\s*" => "")
    cleaned_content = replace(cleaned_content, r"let #hide\s*" => "")

    # Write the modified content back to a new file
    open(output_file, "w") do f
        return write(f, cleaned_content)
    end
end

# Function to format the terminal output for the documentation
function formatting_terminal_output(input_file::String, output_file::String)
    # Read the contents of the file
    file_content = read(input_file, String)

    # Replace the plutouiterminal in the md file by plutouiterminal with padding and background color
    cleaned_content = replace(
        file_content,
        r"\bplutouiterminal\b" => "plutouiterminal\" style=\"padding: 10px; background-color: white;",
    )

    # replace info macro (to keep?? or not use the macro)
    cleaned_content = replace(
        cleaned_content,
        r"�\[36m�\[1m\[ �\[22m�\[39m�\[36m�\[1mInfo: �\[22m�\[39m" => "[ Info: ",
    )

    # Write the modified content back to a new file
    open(output_file, "w") do f
        return write(f, cleaned_content)
    end
end

# Function to format the note sections in the markdown file
function formatting_note_tip_md(input_file::String, output_file::String)
    # Read the contents of the file
    file_content = read(input_file, String)
    
    cleaned_content =
        replace(file_content, r"\badmonition is-note\b" => "admonition is-info")
    cleaned_content =
        replace(cleaned_content, r"\badmonition is-tip\b" => "admonition is-success")

    # Write the modified content back to a new file
    open(output_file, "w") do f
        return write(f, cleaned_content)
    end
end

cleanexamples()

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

## TO REMOVE if we decide to use Pluto Notebooks to generate documentation
# Generate examples using Literate
const examples_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples")
const notebook_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "pluto-examples")
const generated_dir = joinpath(Inti.PROJECT_ROOT, "docs", "src", "examples", "generated")
const examples = ["toy_example.jl", "helmholtz_scattering.jl"]
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

size_threshold_ignore = []
notebooks = [
    "Toy example" => "toy_example.jl",
    "Helmholtz scattering" => "helmholtz_scattering.jl",
    "Poisson problem" => "poisson.jl",
]

# Generate markdown versions of the notebooks for documentation using PlutoStaticHTML.jl
notebook_examples = @docplutonotebooks(notebook_dir, notebooks, iframe = false)
size_threshold_ignore = last.(notebook_examples)

# Formatting the markdown files
for notebook in notebooks
    get_md_files = replace(notebook[2], ".jl" => ".md")
    file =
        joinpath(Inti.PROJECT_ROOT, "docs", "src", "plutostatichtml_examples", get_md_files)
    formatting_pluto(file, file)
    formatting_terminal_output(file, file)
    formatting_note_tip_md(file, file)
end

# Generate HTML versions of the notebooks using PlutoSliderServer.jl
notebook_examples_html = @docplutonotebooks(notebook_dir, notebooks, iframe = true)

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
        mathengine = MathJax3(),
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
        "Examples" => [
            "examples/generated/toy_example.md",
            "examples/generated/helmholtz_scattering.md",
            "examples/poisson.md",
            # "examples/generated/lippmann_schwinger.md",
            # "examples/generated/poisson.md",
            # "examples/generated/stokes_drag.md",
        ],
        "Notebooks" => notebook_examples,
        "References" => "references.md",
        "Docstrings" => "docstrings.md",
    ],
    warnonly = ON_CI ? false : Documenter.except(:linkcheck_remotes),
    # warnonly = true,
    pagesonly = true,
    checkdocs = :none,
    clean=false,
    draft,
    plugins = [bib, links],
)

cleanexamples()

deploydocs(;
    repo = "github.com/IntegralEquations/Inti.jl",
    devbranch = "main",
    push_preview = true,
)
