using TOML
_dispatch_config=isempty(ARGS) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : ARGS[1]
if get(TOML.parsefile(_dispatch_config),"information","")=="observed_quality"
    include(joinpath(@__DIR__,"plot_observed_bgp.jl"))
else
# Run: julia --threads=16 --project=. scripts/plot_bgp.jl [config] [years] [seed] [figure_directory]
using DemandUncertainty, TOML, Serialization, Random, SparseArrays
const DU = DemandUncertainty
config = isempty(ARGS) ? joinpath(@__DIR__, "..", "calibration_bgp.toml") : ARGS[1]
years = length(ARGS)>1 ? parse(Float64,ARGS[2]) : 300.0
seed = length(ARGS)>2 ? parse(Int,ARGS[3]) : 2026
years>0 || error("Simulation horizon must be positive")
p,n,ini = load_config(config)
z = TOML.parsefile(joinpath(n.output,"checkpoint.toml"))["unknowns"]
cache = joinpath(n.output,"plot_state.jls")
# Cache includes its complete inputs: changing calibration or aggregate root invalidates it.
signature = DU.bgp_cache_signature(config,z)
ws = DU.tensor_workspace(p,n,ini,z[4])
if isfile(cache)
    saved = deserialize(cache)
    saved.signature == signature || error("Stale plot cache: remove $cache and rerun")
    ws.V .= saved.V; ws.mu .= saved.mu
    DU.tensor_bellman!(ws.work[5],ws.V,ws,p,n,exp(z[2]),exp(z[3]),z[4])
else
    println("Reconstructing full-grid policies and invariant distribution at saved aggregates."); flush(stdout)
    result = DU.tensor_residual!(ws,z,p,n;verbose=true,update=false)
    maximum(abs,result.residual)<n.tolerance || error("Saved aggregates fail equilibrium check")
    result.distribution_error<1e-8 || error("Invariant distribution did not converge")
    serialize(cache,(signature=signature,V=ws.V,mu=ws.mu))
    println("Saved full-grid plot cache: ",cache); flush(stdout)
end

# Pre-decision invariant masses; shift the internal price coordinate to actual x-tilde.
center = log(p.epsilon/(p.epsilon-1)*exp(z[3])/p.A)
axes = (ws.g.axes[1].+center,ws.g.axes[2],ws.g.axes[3],ws.g.axes[4])
mu = reshape(ws.mu,ws.g.dims)
marginals = [vec(sum(mu;dims=Tuple(j for j in 1:4 if j!=d))) for d in 1:4]
all(m->isapprox(sum(m),1;atol=1e-9),marginals) || error("Marginal mass check failed")
out = length(ARGS)>3 ? ARGS[4] : joinpath(n.output,"figures"); mkpath(out)
open(joinpath(out,"marginals.csv"),"w") do io
    println(io,"state,node,probability_mass")
    for d in 1:4, j in eachindex(axes[d])
        println(io,join((d,axes[d][j],marginals[d][j]),","))
    end
end

include(joinpath(@__DIR__,"bgp_diagnostics.jl"))
policies=continuous_policies(ws,p,n,z,axes)
rows,diagnostics,events=simulate_firm(ws,p,n,z,axes,years,seed;policies=policies)
for (name,table) in (("firm_path.csv",rows),("firm_events.csv",events))
    open(joinpath(out,name),"w") do io
        if isempty(table)
            println(io,"time,kind,before,after")
        else
            println(io,join(keys(first(table)),","))
            for r in table; println(io,join(values(r),",")); end
        end
    end
end
price_mass=marginals[1]
reset_indices=[argmax(view(ws.policy.reset,:,q)) for q in Base.axes(ws.policy.reset,2)]
quality_mass=vec(sum(ws.mu;dims=1))
merge!(diagnostics,Dict("price_boundary_mass"=>first(price_mass)+last(price_mass),
    "price_effective_nodes"=>1/sum(abs2,price_mass),
    "price_nodes_mass_above_1e6"=>count(>(1e-6),price_mass),
    "reset_target_min"=>minimum(axes[1][i] for i in reset_indices),
    "reset_target_max"=>maximum(axes[1][i] for i in reset_indices),
    "reset_target_boundary_quality_mass"=>sum(quality_mass[q] for q in eachindex(reset_indices)
        if reset_indices[q]==1 || reset_indices[q]==length(axes[1]);init=0.)))
merge!(diagnostics,Dict("seed"=>seed,"years"=>years,"policy_solution_dt"=>n.dt,
    "grid_dimensions"=>collect(ws.g.dims),"initialization"=>"pre-decision finite-grid invariant distribution",
    "marginal_mass_errors"=>[abs(sum(m)-1) for m in marginals]))
open(joinpath(out,"diagnostics.toml"),"w") do io
    TOML.print(io,diagnostics)
end

using CairoMakie
CairoMakie.activate!()
set_theme!(Theme(fontsize=16))
labels=["Detrended log relative price", "Taste gap: a − M", "Belief gap: b − M", "Posterior variance: v"]
f=Figure(size=(1100,750))
for d in 1:4
    ax=Axis(f[cld(d,2),mod1(d,2)],xlabel=labels[d],ylabel="Probability mass")
    gaps=diff(axes[d])
    widths=0.9 .* vcat(first(gaps),min.(gaps[1:end-1],gaps[2:end]),last(gaps))
    barplot!(ax,axes[d],marginals[d];color=:steelblue,width=widths)
end
Label(f[0,:],"Stationary equilibrium — pre-decision marginals ($(join(ws.g.dims," × ")))",fontsize=21)
for ext in ("png","pdf"); save(joinpath(out,"equilibrium_marginals.$ext"),f); end

t=[r.time for r in rows]
series(key)=[getproperty(r,key) for r in rows]
f=Figure(size=(1300,1050))
a1=Axis(f[1,1],ylabel="Detrended log relative price")
lines!(a1,t,series(:price),color=:steelblue)
price_events=filter(e->e.kind=="price",events)
scatter!(a1,[e.time for e in price_events],[e.after for e in price_events],color=:firebrick,markersize=5,label="Price adjustment")
axislegend(a1;position=:rt)
a2=Axis(f[1,2],ylabel="Taste / belief gap")
b=series(:belief); sd=sqrt.(series(:variance))
band!(a2,t,b.-1.96sd,b.+1.96sd,color=(:orange,.2),label="95% posterior band for latent quality")
lines!(a2,t,series(:taste),label="Observed taste",color=:steelblue)
lines!(a2,t,b,label="Posterior mean",color=:darkorange)
axislegend(a2;position=:rt,labelsize=12)
a3=Axis(f[2,1],ylabel="Posterior variance")
lines!(a3,t,series(:variance),color=:purple)
a4=Axis(f[2,2],ylabel="Detrended R&D expenditure")
lines!(a4,t,series(:research_expenditure),label="Total",color=:darkgreen)
lines!(a4,t,series(:research),label="Variable effort",color=:orange)
axislegend(a4;position=:rt,labelsize=12)
a5=Axis(f[3,1],xlabel="Years",ylabel="Physical sales")
lines!(a5,t,series(:sales),color=:steelblue)
a6=Axis(f[3,2],xlabel="Years",ylabel="Detrended profit flow")
lines!(a6,t,series(:operating_profit),label="Operating",color=:steelblue)
lines!(a6,t,series(:net_flow),label="After R&D (menu costs are impulses)",color=:firebrick)
axislegend(a6;position=:rt,labelsize=12)
innovations=[e.time for e in events if e.kind=="innovation"]
replacements=[e.time for e in events if e.kind=="entry"]
for ax in (a1,a2,a3,a4,a5,a6)
    isempty(replacements) || vlines!(ax,replacements;color=(:purple,.5),linestyle=:dot,linewidth=1)
    isempty(innovations) || vlines!(ax,innovations;color=(:black,.4),linestyle=:dash,linewidth=1)
end
linkxaxes!(a1,a2,a3,a4,a5,a6)
Label(f[0,:],(p.replacement_entry ? "Product slot (successive firms)" : "One firm")*" — seed $seed; dashed lines mark innovation successes",fontsize=21)
Label(f[4,:],"Adaptive continuous-time SDE; linearly interpolated policies; menu payments recorded as cash jumps.",fontsize=14)
for ext in ("png","pdf"); save(joinpath(out,"firm_paths.$ext"),f); end
# Reconstruct levels with P(0)=1 and M(0)=0 (hence Q(0)=1).
# The firm's initial nominal price remains its equilibrium draw, not necessarily 1.
f=Figure(size=(1300,850))
l1=Axis(f[1,1],ylabel="Nominal price level",yscale=log10)
lines!(l1,t,series(:nominal_price),label="Firm price pᵢₜ",color=:steelblue)
lines!(l1,t,series(:aggregate_price),label="Aggregate Pₜ",color=:darkorange,linestyle=:dash)
axislegend(l1;position=:lt,labelsize=13)
l2=Axis(f[1,2],ylabel="Relative price level: pᵢₜ / Pₜ")
lines!(l2,t,series(:relative_price),color=:steelblue)
l3=Axis(f[2,1],xlabel="Years",ylabel="Undetrended taste state: aᵢₜ")
bl=series(:belief_level)
band!(l3,t,bl.-1.96sd,bl.+1.96sd,color=(:orange,.2))
lines!(l3,t,series(:taste_level),label="Taste aᵢₜ",color=:steelblue)
lines!(l3,t,bl,label="Posterior mean bᵢₜ",color=:darkorange)
lines!(l3,t,series(:quality_mean),label="Market mean Mₜ",color=:black,linestyle=:dash)
axislegend(l3;position=:lt,labelsize=12)
l4=Axis(f[2,2],xlabel="Years",ylabel="Taste multiplier: exp(aᵢₜ)")
lines!(l4,t,series(:taste_multiplier),label="Firm demand multiplier",color=:steelblue)
lines!(l4,t,exp.(series(:quality_mean)),label="exp(Mₜ)",color=:black,linestyle=:dash)
axislegend(l4;position=:lt,labelsize=12)
for ax in (l1,l2,l3,l4)
    isempty(innovations) || vlines!(ax,innovations;color=(:black,.3),linestyle=:dot,linewidth=1)
end
linkxaxes!(l1,l2,l3,l4)
Label(f[0,:],(p.replacement_entry ? "Product slot in levels" : "One firm in levels")*" — P₀ = 1, M₀ = 0; seed $seed",fontsize=22)
Label(f[3,:],"Nominal price panel uses a logarithmic axis; dotted vertical lines mark innovation successes.",fontsize=14)
for ext in ("png","pdf"); save(joinpath(out,"firm_levels.$ext"),f); end
if p.replacement_entry
    bgrid=axes[3]
    decisions=[policies.survival(x,b,b,0.)<=0 ? 2. : policies.surplus(x,b,b,0.)>0 ? 1. : 0.
        for x in axes[1],b in bgrid]
    f=Figure(size=(1100,750),fontsize=18)
    ax=Axis(f[1,1],xlabel="Detrended log relative price",ylabel="Relative posterior mean beta",
        title="Policy slice: taste = posterior mean, variance = 0")
    hm=heatmap!(ax,axes[1],bgrid,decisions;colormap=[:steelblue,:orange,:firebrick],colorrange=(0.,2.))
    Colorbar(f[1,2],hm;ticks=([0,1,2],["Continue","Reprice","Exit"]))
    scatter!(ax,[policies.entry_price],[p.premium];color=:black,marker=:star5,markersize=18)
    Label(f[2,:],"Star: entrant price and mean quality (entry variance is sigma_Q^2). Exit pays zero.",fontsize=14)
    for ext in ("png","pdf");save(joinpath(out,"exit_policy.$ext"),f);end
end
println("Figures and underlying CSV data saved in ",abspath(out))

end # information-structure dispatch
