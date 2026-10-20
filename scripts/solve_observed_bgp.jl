# Observed-quality BGP: three states, actual quality draws, no filtering.
include(joinpath(@__DIR__,"..","src","ObservedQuality.jl"))
using .ObservedQuality, TOML, Serialization, LinearAlgebra
BLAS.set_num_threads(1)
function main(args)
    path=abspath(isempty(args) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : args[1])
    p,n=ObservedQuality.config(path)
    cache=joinpath(n.output,"observed_state.jls")
    warm=length(args)>1 ? deserialize(args[2]) : nothing
    if warm===nothing && isfile(cache)
        candidate=deserialize(cache)
        candidate.signature==ObservedQuality.signature(path) && (warm=candidate)
    end
    guess=warm===nothing ? [log(.312),.073] : warm.theta
    result=ObservedQuality.solve(p,n;guess=guess,warm=warm)
    ObservedQuality.save_result(result,p,n,path)
    result.converged || exit(2)
end
main(ARGS)
