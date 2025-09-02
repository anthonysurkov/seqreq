#!/usr/bin/env julia
import JSON3
import FourLeafMLE

function run_once(vec8::Vector{Int})
    res = FourLeafMLE.fourLeafMLE(vec8)
    r = res[1]
    params = Float64[]
    desc   = ""
    for x in r
        if x isa AbstractVector{<:Real} && length(x) == 5
            params = Float64.(x)
        elseif x isa String
            desc = x
        end
    end
    return (; params, desc)
end

try
    raw = JSON3.read(read(stdin, String))     # JSON3.Array
    v   = Int.(collect(raw)) :: Vector{Int}   # convert to Vector{Int}
    out = run_once(v)
    println(JSON3.write(out))                 # ONLY JSON on stdout
catch err
    Base.print(stderr, sprint(showerror, err, catch_backtrace()))
    exit(1)
end
