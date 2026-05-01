using Documenter
using Yelmo

DocMeta.setdocmeta!(Yelmo, :DocTestSetup, :(using Yelmo); recursive=true)

makedocs(
    sitename = "Yelmo.jl",
    modules  = [
        Yelmo,
        Yelmo.YelmoMeta,
        Yelmo.YelmoConst,
        Yelmo.YelmoPar,
        Yelmo.YelmoModelPar,
        Yelmo.YelmoCore,
        Yelmo.YelmoMirrorCore,
        Yelmo.YelmoModelTopo,
        Yelmo.YelmoModelDyn,
        Yelmo.YelmoIO,
    ],
    authors  = "Alexander Robinson <alexander.robinson@awi.de> and contributors",
    repo     = "https://github.com/fesmc/Yelmo.jl",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://fesmc.github.io/Yelmo.jl",
        assets     = String[],
        size_threshold       = 400 * 1024,
        size_threshold_warn  = 200 * 1024,
    ),
    pages = [
        "Home"            => "index.md",
        "Getting started" => "getting-started.md",
        "Concepts"        => "concepts.md",
        "Usage" => [
            "Loading a model"     => "usage/loading.md",
            "Stepping the model"  => "usage/stepping.md",
            "Output and NetCDF"   => "usage/io.md",
            "Comparing states"    => "usage/comparing.md",
        ],
        "Physics" => [
            "Overview"             => "physics/index.md",
            "Topography step"      => "physics/topography.md",
            "Advection"            => "physics/advection.md",
            "Mass balance"         => "physics/mass-balance.md",
            "Grounded fraction"    => "physics/grounded-fraction.md",
            "Relaxation"           => "physics/relaxation.md",
            "Calving"              => "physics/calving.md",
            "Dynamics"             => "physics/dynamics.md",
        ],
        "API reference" => [
            "Core model"      => "api/core.md",
            "Parameters"      => "api/parameters.md",
            "Constants"       => "api/constants.md",
            "Topography"      => "api/topography.md",
            "Dynamics"        => "api/dynamics.md",
            "Input / output"  => "api/io.md",
            "Yelmo Mirror"    => "api/mirror.md",
        ],
        "Variables"   => "variables.md",
        "References"  => "references.md",
    ],
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

# Uncomment when ready to deploy to GitHub Pages.
#
# deploydocs(
#     repo      = "github.com/fesmc/Yelmo.jl",
#     devbranch = "main",
#     push_preview = true,
# )
