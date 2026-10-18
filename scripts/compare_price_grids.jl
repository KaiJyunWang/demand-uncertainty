using TOML, CairoMakie
old_dir=length(ARGS)>0 ? ARGS[1] : "results_bgp_wide_price_grid"
new_dir=length(ARGS)>1 ? ARGS[2] : "results_bgp"
function read_marginal(dir)
    data=[parse.(Float64,split(line,",")) for line in readlines(joinpath(dir,"figures","marginals.csv"))[2:end]]
    price=filter(row->row[1]==1,data)
    [row[2] for row in price],[row[3] for row in price]
end
old=TOML.parsefile(joinpath(old_dir,"balanced_growth.toml"))
new=TOML.parsefile(joinpath(new_dir,"balanced_growth.toml"))
ox,om=read_marginal(old_dir);xx,mm=read_marginal(new_dir)
old["price_grid_spacing"]=ox[2]-ox[1]
old["price_effective_nodes"]=1/sum(abs2,om)
old["price_nodes_mass_above_1e6"]=count(>(1e-6),om)
old["price_boundary_mass"]=first(om)+last(om)
metrics=["price_grid_spacing","price_effective_nodes","price_nodes_mass_above_1e6",
    "price_boundary_mass","C","Y","w","N","G","consumption_growth_pct_per_year",
    "price_frequency","research","innovation_frequency","clipping",
    "max_equilibrium_residual","bellman_residual","distribution_residual"]
open(joinpath(new_dir,"price_grid_comparison.csv"),"w") do io
    println(io,"metric,wide_grid,tight_grid,absolute_change,relative_change_pct")
    for key in metrics
        a,b=old[key],new[key]
        println(io,join((key,a,b,b-a,a==0 ? NaN : 100*(b/a-1)),","))
    end
end
f=Figure(size=(1150,480),fontsize=16)
for (j,(x,m,name)) in enumerate(((ox,om,"Previous: price gap ±0.45"),(xx,mm,"Recomputed: price gap ±0.075")))
    active=count(>(1e-6),m)
    ax=Axis(f[1,j],xlabel="Detrended log relative price",ylabel="Probability mass",
        title="$name\n$active of 51 nodes have mass > 10⁻⁶")
    barplot!(ax,x,m;width=.9*(x[2]-x[1]),color=:steelblue)
end
Label(f[0,:],"Price-grid allocation before and after refinement",fontsize=22)
for ext in ("png","pdf")
    save(joinpath(new_dir,"figures","price_grid_comparison.$ext"),f)
end
println("Saved grid comparison to ",new_dir)
