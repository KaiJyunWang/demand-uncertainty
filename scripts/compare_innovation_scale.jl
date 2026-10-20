# Compare current aggregates with the saved lambda0=0.5 replacement-entry run.
using TOML
config=isempty(ARGS) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : ARGS[1]
calibration=TOML.parsefile(config);out=calibration["numerical"]["output"]
benchmark=joinpath(out,"lambda0_0p5_benchmark.toml")
if isfile(benchmark)
    before=TOML.parsefile(benchmark);after=TOML.parsefile(joinpath(out,"balanced_growth.toml"))
    before["lambda0"]=get(before,"lambda0",0.5)
    @assert after["lambda0"]==calibration["economic"]["lambda0"]
    open(joinpath(out,"innovation_scale_comparison.csv"),"w") do io
        println(io,"metric,lambda0_0p5,current,change")
        for key in ("lambda0","C","Y","w","N","G","consumption_growth_pct_per_year",
                "research","variable_research","fixed_research","rd_active_mass","innovation_frequency",
                "price_frequency","exit_rate","entry_rate","entry_value","mean_variance")
            a,b=before[key],after[key]
            println(io,join((key,a,b,b-a),","))
        end
    end
end
