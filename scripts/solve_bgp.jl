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
println(report)
if hasproperty(result,:large) && result.converged
    signature=(read(config,String),result.unknowns,string(VERSION),
        read(joinpath(@__DIR__,"..","src","LargeBGP.jl"),String))
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
