# Explicit multi-parameter comparison with the earlier inactive equilibrium.
using TOML
config=isempty(ARGS) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : ARGS[1]
d=TOML.parsefile(config);out=d["numerical"]["output"]
old_report=joinpath(out,"inactive_benchmark.toml")
old_config=joinpath(out,"inactive_benchmark_calibration.toml")
if isfile(old_report) && isfile(old_config)
    before=TOML.parsefile(old_report);after=TOML.parsefile(joinpath(out,"balanced_growth.toml"))
    old=TOML.parsefile(old_config)["economic"];new=d["economic"]
    open(joinpath(out,"active_parameter_changes.csv"),"w") do io
        println(io,"parameter,inactive_benchmark,active_calibration,change")
        for key in ("F_R","lambda0","premium","sigma_Q","K","chi","sigma_a","rho","epsilon")
            println(io,join((key,old[key],new[key],new[key]-old[key]),","))
        end
    end
    open(joinpath(out,"active_equilibrium_comparison.csv"),"w") do io
        println(io,"metric,inactive_benchmark,active_equilibrium,change")
        for key in ("C","Y","w","N","G","consumption_growth_pct_per_year","research",
                "variable_research","fixed_research","rd_active_mass","innovation_frequency",
                "exit_rate","entry_rate","price_frequency","mean_variance","rd_binding_mass")
            println(io,join((key,before[key],after[key],after[key]-before[key]),","))
        end
    end
end
