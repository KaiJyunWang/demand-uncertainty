using TOML
_dispatch_config=isempty(ARGS) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : ARGS[1]
if get(TOML.parsefile(_dispatch_config),"information","")=="observed_quality"
    root=normpath(joinpath(@__DIR__,".."))
    cmd(script)=`julia --threads=$(Threads.nthreads()) --project=$root $(joinpath(@__DIR__,script)) $_dispatch_config`
    "--figures-only" in ARGS || run(cmd("solve_observed_bgp.jl"))
    run(cmd("check_observed_bgp.jl"))
    run(cmd("plot_observed_bgp.jl"))
else
# Re-solve and regenerate all current outputs using juliaup's default Julia.
# julia --threads=16 --project=. scripts/recompute_bgp.jl [config]
using DemandUncertainty, TOML
root=normpath(joinpath(@__DIR__,".."))
config=abspath(isempty(ARGS) ? joinpath(root,"calibration_bgp.toml") : ARGS[1])
p,n,ini=load_config(config)
mkpath(n.output)
function run_logged(script,args,logfile)
    command=Cmd(vcat(["julia","--threads=$(Threads.nthreads())","--project=$root",
        joinpath(root,"scripts",script)],String.(args)))
    println("Running ",script,"; log: ",logfile);flush(stdout)
    open(logfile,"w") do io
        run(pipeline(command;stdout=io,stderr=io))
    end
end
if !("--figures-only" in ARGS)
mktempdir() do temporary
    reportfile=joinpath(n.output,"balanced_growth.toml")
    if isfile(reportfile)
        baseline=TOML.parsefile(reportfile)
        if p.replacement_entry && !get(baseline,"replacement_entry",false)
            open(joinpath(n.output,"no_turnover_benchmark.toml"),"w") do io;TOML.print(io,baseline);end
        end
        if get(baseline,"F_R",0.) != p.F_R
            open(joinpath(n.output,"fixed_cost_benchmark.toml"),"w") do io
                TOML.print(io,baseline)
            end
        end
        if !get(baseline,"rd_cashflow_constraint",false)
            open(joinpath(n.output,"financing_benchmark.toml"),"w") do io
                TOML.print(io,baseline)
            end
        end
    end
    solve_args=[config]
    checkpoint=joinpath(n.output,"checkpoint.toml")
    cache=joinpath(n.output,"plot_state.jls")
    if isfile(checkpoint)
        cp(checkpoint,joinpath(temporary,"checkpoint.toml"))
        push!(solve_args,joinpath(temporary,"checkpoint.toml"))
        if isfile(cache)
            cp(cache,joinpath(temporary,"state.jls"))
            push!(solve_args,joinpath(temporary,"state.jls"))
        end
    end
    run_logged("solve_bgp.jl",solve_args,joinpath(n.output,"solve_51.log"))
end
end
report=TOML.parsefile(joinpath(n.output,"balanced_growth.toml"))
report["converged"] && report["rd_cashflow_constraint"] || error("Constrained equilibrium was not obtained")
report["rd_max_cashflow_violation"]<=1e-12 || error("Equilibrium violates the financing constraint")
if report["G"]>0 && p.replacement_entry
    run_logged("check_active_bgp.jl",[config],joinpath(n.output,"active_equilibrium_checks.log"))
elseif report["G"]==0 && p.F_R>0
    run_logged("check_fixed_cost_bgp.jl",[config],joinpath(n.output,"fixed_cost_checks.log"))
end
run_logged("plot_bgp.jl",[config],joinpath(n.output,"plot_adaptive.log"))
run_logged("plot_bgp.jl",[config,"30","2026",joinpath(n.output,"figures","30_year")],joinpath(n.output,"plot_30_year.log"))
run_logged("innovation_response.jl",[config],joinpath(n.output,"innovation_response.log"))
baseline_file=joinpath(n.output,"financing_benchmark.toml")
if isfile(baseline_file) && n.bgp_branch=="auto"
    baseline=TOML.parsefile(baseline_file)
    metrics=("C","Y","w","N","G","consumption_growth_pct_per_year",
        "price_frequency","research","innovation_frequency","mean_variance")
    open(joinpath(n.output,"financing_comparison.csv"),"w") do io
        println(io,"metric,unconstrained,constrained,change,relative_change_pct")
        for key in metrics
            before,after=baseline[key],report[key]
            println(io,join((key,before,after,after-before,before==0 ? NaN : 100*(after/before-1)),","))
        end
    end
end
# These old grid-comparison artifacts referred to the superseded unconstrained run.
for file in ("price_grid_comparison.csv","compare_price_grids.log",
        joinpath("figures","price_grid_comparison.png"),joinpath("figures","price_grid_comparison.pdf"))
    path=joinpath(n.output,file)
    isfile(path) && rm(path)
end
println("Constrained equilibrium, 300/30-year simulations, and innovation levels regenerated.")
fixed_benchmark=joinpath(n.output,"fixed_cost_benchmark.toml")
if isfile(fixed_benchmark) && n.bgp_branch=="auto"
    baseline=TOML.parsefile(fixed_benchmark)
    open(joinpath(n.output,"fixed_cost_comparison.csv"),"w") do io
        println(io,"metric,previous,new,change,relative_change_pct")
        for key in ("F_R","C","Y","w","N","G","consumption_growth_pct_per_year",
                "price_frequency","research","variable_research","fixed_research",
                "rd_active_mass","innovation_frequency","mean_variance")
            before=get(baseline,key,key=="F_R" || key=="fixed_research" ? 0. : NaN)
            key=="variable_research" && !haskey(baseline,key) && (before=baseline["research"])
            after=report[key]
            println(io,join((key,before,after,after-before,before==0 ? NaN : 100*(after/before-1)),","))
        end
    end
end

benchmark=joinpath(n.output,"no_turnover_benchmark.toml")
if p.replacement_entry && isfile(benchmark) && n.bgp_branch=="auto"
    before=TOML.parsefile(benchmark)
    open(joinpath(n.output,"turnover_comparison.csv"),"w") do io
        println(io,"metric,no_turnover,replacement_entry,change")
        for key in ("C","Y","w","G","research","price_frequency","exit_rate","entry_rate")
            b=get(before,key,0.);a=report[key];println(io,join((key,b,a,a-b),","))
        end
    end
end
if n.bgp_branch=="auto"
    include(joinpath(@__DIR__,"compare_innovation_scale.jl"))
else
    include(joinpath(@__DIR__,"compare_active_equilibrium.jl"))
end

end # information-structure dispatch
