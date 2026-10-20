using TOML
_dispatch_config=isempty(ARGS) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : ARGS[1]
if get(TOML.parsefile(_dispatch_config),"information","")=="observed_quality"
    include(joinpath(@__DIR__,"solve_observed_bgp.jl"))
else
using DemandUncertainty, TOML, Serialization
config=length(ARGS)>=1 ? ARGS[1] : joinpath(@__DIR__,"..","calibration_bgp.toml")
p,n,ini=load_config(config)
mkpath(n.output)
if abspath(config)!=abspath(joinpath(n.output,"run_calibration.toml"))
    cp(config,joinpath(n.output,"run_calibration.toml");force=true)
end
guess=length(ARGS)>=2 ? TOML.parsefile(ARGS[2])["unknowns"] : nothing
warmstart=nothing
if length(ARGS)>=3
    include(joinpath(@__DIR__,"price_regrid.jl"))
    warmstart=price_regrid(ARGS[3],config,p,n,guess)
end
result=solve_balanced_growth(p,n,ini;guess=guess,warmstart=warmstart)
report=write_bgp_results(result,p,n)
if hasproperty(result,:large)
    open(joinpath(n.output,"large_progress.toml"),"w") do io
        TOML.print(io,Dict("iteration"=>length(result.history),"unknowns"=>result.unknowns,
            "residuals"=>result.residual,"equilibrium_error"=>report["max_equilibrium_residual"],
            "firm_error"=>report["bellman_residual"],"distribution_error"=>report["distribution_residual"],
            "F_R"=>p.F_R,"equilibrium_selection"=>report["equilibrium_selection"]))
    end
end
println(report)
if hasproperty(result,:large) && result.converged
    signature=DemandUncertainty.bgp_cache_signature(config,result.unknowns)
    serialize(joinpath(n.output,"plot_state.jls"),
        (signature=signature,V=result.workspace.V,mu=result.workspace.mu))
end
if (n.nx,n.na,n.nb,n.nv)==(51,51,51,51)
    cp(joinpath(n.output,"solution.csv"),joinpath(@__DIR__,"..","bgp_solution_51.csv");force=true)
end
if !result.converged
    println(stderr,"BGP equilibrium not converged. Outputs are candidate values.")
    exit(2)
end
if result.moments.clipping>0.01
    println(stderr,"BGP root converged, but boundary clipping requires grid/domain refinement.")
end

end # information-structure dispatch
