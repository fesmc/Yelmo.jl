module YelmoMeta

export VariableMeta, parse_variable_table

# ---------------------------------------------------------------------------
struct VariableMeta
    id         :: Int
    set        :: String
    name       :: Symbol
    cname      :: Vector{UInt8}
    dimensions :: Tuple{Vararg{Symbol}}
    units      :: String
    long_name  :: String
end

Base.show(io::IO, v::VariableMeta) =
    print(io, "VariableMeta($(v.name) : \"$(v.long_name)\" [$(v.units)], dims=$(v.dimensions))")

# ---------------------------------------------------------------------------
"""
    parse_variable_table(filename) -> NamedTuple{names, NTuple{N,VariableMeta}}

Parse a Markdown variable table into a NamedTuple of `VariableMeta` entries
keyed by variable name (as Symbols). Expected columns:
`id | variable | dimensions | units | long_name`.
"""
function parse_variable_table(filename::AbstractString,set::AbstractString)
    markdown = read(filename, String)

    rows  = VariableMeta[]
    names = Symbol[]

    for line in eachline(IOBuffer(markdown))
        stripped = strip(line)
        isempty(stripped)               && continue
        startswith(stripped, "| id")    && continue
        startswith(stripped, "|-")      && continue
        startswith(stripped, "# ")      && continue
        startswith(stripped, '|')       || continue

        cols = strip.(split(stripped, '|'))
        filter!(!isempty, cols)
        length(cols) >= 5 || continue

        id        = parse(Int, cols[1])
        set       = string(set)
        name      = Symbol(strip(cols[2]))
        name_str  = string(name)
        cname     = Vector{UInt8}("$(set)_$(name_str)\0")
        dims      = Tuple(Symbol(strip(d)) for d in split(cols[3], ','))
        units     = strip(cols[4])
        long_name = strip(cols[5])

        push!(names, name)
        push!(rows, VariableMeta(id, set, name, cname, dims, units, long_name))
    end

    return NamedTuple{Tuple(names)}(Tuple(rows))
end

end # module YelmoMeta
