using DemandUncertainty, TOML
config=length(ARGS)>=1 ? ARGS[1] : joinpath(@__DIR__,"..","calibration.toml")
p,n,ini=load_config(config)
println("Solving finite-horizon menu-cost / learning / R&D equilibrium")
guess=length(ARGS)>=2 ? TOML.parsefile(ARGS[2])["unknowns"] : nothing
result=solve_equilibrium(p,n,ini;guess=guess)
summary=write_results(result,p,n)
println(summary)
if !result.converged
    println(stderr,"Equilibrium did not meet tolerance. Inspect diagnostics and refine/restart; outputs are candidate paths.")
    exit(2)
end
if summary["max_transition_clipping"]>0.01 || !summary["terminal_value_converged"]
    println(stderr,"Numerical diagnostics need attention; do not interpret this coarse run as a validated calibration.")
end
