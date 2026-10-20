# Diagnostic parameter search on smaller grids; never overwrites active results.
using DemandUncertainty,TOML,Serialization,Printf
const DU=DemandUncertainty
root=normpath(joinpath(@__DIR__,".."))
base=TOML.parsefile(joinpath(root,"calibration_bgp.toml"))
candidates=[
    (F_R=.01,lambda0=3.,premium=.1,sigma_Q=.3,K=.005),
    (F_R=.005,lambda0=3.,premium=.1,sigma_Q=.3,K=.005),
    (F_R=.015,lambda0=5.,premium=.15,sigma_Q=.4,K=.005),
    (F_R=.01,lambda0=3.,premium=.2,sigma_Q=.3,K=.01),
    (F_R=.02,lambda0=5.,premium=.2,sigma_Q=.4,K=.01),
    (F_R=.01,lambda0=5.,premium=.1,sigma_Q=.4,K=.02),
    (F_R=.01,lambda0=3.,premium=.2,sigma_Q=.5,K=.02),
    (F_R=.01,lambda0=3.,premium=.2,sigma_Q=.6,K=.03),
    (F_R=.015,lambda0=5.,premium=.3,sigma_Q=.5,K=.02)]
indices=isempty(ARGS) ? collect(eachindex(candidates)) : parse.(Int,ARGS)
for index in indices
    d=deepcopy(base);candidate=candidates[index]
    for (k,v) in pairs(candidate);d["economic"][string(k)]=v;end
    n=d["numerical"]
    merge!(n,Dict("nx"=>17,"na"=>17,"nb"=>17,"nv"=>17,
        "price_lower_tail_nodes"=>3,"quality_lower_tail_nodes"=>4,
        "xmin"=>-.35,"xmax"=>.2,"bgp_branch"=>"positive",
        "max_iterations"=>35,"output"=>joinpath("/tmp","active_bgp_search_$index")))
    mkpath(n["output"]);config=joinpath(n["output"],"calibration.toml")
    open(config,"w") do io;TOML.print(io,d);end
    println("CANDIDATE ",index," ",candidate);flush(stdout)
    try
        p,num,ini=load_config(config)
        r=solve_balanced_growth(p,num,ini;guess=[log(.33),log(.34),log(.86),.03],verbose=false)
        report=write_bgp_results(r,p,num)
        if r.converged
            serialize(joinpath(n["output"],"plot_state.jls"),
                (signature=DU.bgp_cache_signature(config,r.unknowns),V=r.workspace.V,mu=r.workspace.mu))
        end
        println("SEARCH RESULT ",index," converged=",r.converged," ",r.moments);flush(stdout)
    catch e
        println("SEARCH FAILURE ",index," ",sprint(showerror,e));flush(stdout)
    end
end
