# Run independent numerical sensitivity experiments. Each may take several minutes.
using DemandUncertainty, TOML
config=length(ARGS)>=1 ? ARGS[1] : joinpath(@__DIR__,"..","calibration.toml")
p,n,ini=load_config(config)
base=Dict(k=>getfield(n,k) for k in fieldnames(Numerics))
cases=[("baseline",Dict{Symbol,Any}()),
       ("time",Dict{Symbol,Any}(:dt=>n.dt/2,:steps=>2*n.steps)),
       ("horizon",Dict{Symbol,Any}(:steps=>2*n.steps)),
       ("grid",Dict{Symbol,Any}(:nx=>2*n.nx-1,:na=>2*n.na-1,:nb=>2*n.nb-1)),
       ("temperature",Dict{Symbol,Any}(:price_temperature=>n.price_temperature/10)),
       ("terminal",Dict{Symbol,Any}(:terminal_value=>"zero"))]
summaries=Dict{String,Any}()
for (label,changes) in cases
    println("\nSensitivity case: ",label)
    settings=merge(base,changes,Dict(:output=>"results_"*label))
    ni=Numerics(;settings...)
    result=solve_equilibrium(p,ni,ini)
    summary=write_results(result,p,ni)
    summary["initial_consumption"]=result.moments[1].C
    summary["initial_inflation"]=result.moments[1].pi
    summaries[label]=summary
end
open("refinement.toml","w") do io; TOML.print(io,summaries); end
if !all(s["converged"] for s in values(summaries))
    println(stderr,"At least one sensitivity case failed aggregate tolerance; see refinement.toml")
    exit(2)
end
