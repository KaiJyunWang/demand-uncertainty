# Independent BGP numerical sensitivity checks; no finite horizon is required.
using DemandUncertainty, TOML
function main(args)
config=length(args)>=1 ? args[1] : joinpath(@__DIR__,"..","calibration_bgp.toml")
p,n,ini=load_config(config)
base=Dict(k=>getfield(n,k) for k in fieldnames(Numerics))
cases=[("baseline",Dict{Symbol,Any}()),
       ("time",Dict{Symbol,Any}(:dt=>n.dt/2)),
       ("price_grid",Dict{Symbol,Any}(:nx=>2*n.nx-1)),
       ("belief_grid",Dict{Symbol,Any}(:nb=>2*n.nb-1,:nv=>2*n.nv-1)),
       ("domain",Dict{Symbol,Any}(:amin=>n.amin*1.5,:amax=>n.amax*1.5,
           :bmin=>n.bmin*1.5,:bmax=>n.bmax*1.5,
           :na=>round(Int,1.5*(n.na-1))+1,:nb=>round(Int,1.5*(n.nb-1))+1)),
       ("temperature",Dict{Symbol,Any}(:price_temperature=>n.price_temperature/10))]
summaries=Dict{String,Any}(); guess=nothing
for (label,changes) in cases
    println("\nBGP sensitivity case: ",label)
    ni=Numerics(;merge(base,changes,Dict(:output=>"results_bgp_"*label))...)
    result=solve_balanced_growth(p,ni,ini;guess=guess)
    summaries[label]=write_bgp_results(result,p,ni)
    label=="baseline" && result.converged && (guess=result.unknowns)
end
open("refinement.toml","w") do io; TOML.print(io,summaries); end
if !all(s["converged"] for s in values(summaries))
    println(stderr,"At least one sensitivity root failed; inspect refinement.toml")
    exit(2)
end

end
main(ARGS)
