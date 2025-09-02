#!/usr/bin/env julia
import JSON3
import FourLeafMLE

function infer(vec8::Vector{Int})
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

# Read newline-delimited JSON arrays; print one JSON per line
while !eof(stdin)
    line = strip(readline(stdin))
    isempty(line) && continue
    try
        raw = JSON3.read(line)         # e.g. [12,5,3,2,1,0,0,0]
        v   = Int.(collect(raw))
        out = infer(v)
        println(JSON3.write(out))
        flush(stdout)
    catch err
        Base.print(stderr, sprint(showerror, err, catch_backtrace()))
        println(JSON3.write(Dict("params"=>[], "desc"=>"error")))
        flush(stdout)
    end
end

