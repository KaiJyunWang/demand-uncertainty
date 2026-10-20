# Verify strict nonparticipation throughout the selected zero-growth class.
using DemandUncertainty,Serialization,TOML
const DU=DemandUncertainty
config=isempty(ARGS) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : ARGS[1]
p,n,ini=load_config(config)
saved=deserialize(joinpath(n.output,"plot_state.jls"));z=saved.signature[2]
@assert saved.signature==DU.bgp_cache_signature(config,z)
ws=DU.tensor_workspace(p,n,ini,z[4]);ws.V .= saved.V;ws.mu .= saved.mu
DU.tensor_bellman!(ws.work[5],ws.V,ws,p,n,exp(z[2]),exp(z[3]),z[4])
qs=findall(q->q[2]==0 && q[3]==0,ws.g.qstates)
p0=Parameters(; (name=>getfield(p,name) for name in fieldnames(Parameters) if name!=:F_R)...,F_R=0.)
disc=exp(-p.rho*n.dt);max_active_surplus=-Inf;maxgain=-Inf;feasible=0
for q in qs,x in axes(ws.V,1)
    gain=disc*(ws.work[3][x,q]-ws.work[2][x,q]);global maxgain=max(maxgain,gain)
    cap=max(ws.policy.profit[x,q],0.)
    if cap>p.F_R && gain>0
        candidate=DU.tensor_rd(gain,p0,n.dt;cap=cap-p.F_R)
        global max_active_surplus=max(max_active_surplus,DU.step_rd_surplus(candidate,gain,p,n.dt))
        global feasible+=1
    end
end
@assert maximum(ws.policy.k[:,qs])==0
@assert max_active_surplus<=0
report=TOML.parsefile(joinpath(n.output,"balanced_growth.toml"))
@assert report["rd_active_mass"]==0 && report["F_R"]==p.F_R
checks=Dict("lambda0"=>p.lambda0,"F_R"=>p.F_R,"support_price_taste_nodes"=>length(qs)*n.nx,
    "positive_gain_financeable_nodes"=>feasible,"maximum_discounted_innovation_gain"=>maxgain,
    "maximum_active_branch_surplus"=>max_active_surplus,"maximum_support_effort"=>0.,
    "nonzero_research_nodes_off_support"=>count(>(0),ws.policy.k),
    "closed_class_selection"=>report["equilibrium_selection"])
open(joinpath(n.output,"fixed_cost_checks.toml"),"w") do io;TOML.print(io,checks);end
println(checks)

if p.replacement_entry
    @assert minimum(ws.V)>=-1e-10
    @assert maximum(abs,ws.V.*ws.policy.exit)<=1e-10
    @assert maximum(ws.policy.exit[:,qs])==0
    @assert sum(ws.entry)≈1
    @assert sum(ws.entry.*ws.V)>0
    @assert sum(ws.entry.*ws.policy.exit)==0
    turnover_checks=Dict("lambda0"=>p.lambda0,"replacement_entry"=>true,"exit_policy_nodes"=>count(>(0),ws.policy.exit),
        "exit_max_value"=>maximum(ws.V.*ws.policy.exit),"entry_mass"=>sum(ws.entry),
        "entry_value"=>sum(ws.entry.*ws.V),"entrant_immediate_exit_mass"=>sum(ws.entry.*ws.policy.exit),
        "equilibrium_exit_rate"=>report["exit_rate"],"equilibrium_entry_rate"=>report["entry_rate"],
        "support_exit_nodes"=>count(>(0),ws.policy.exit[:,qs]))
    open(joinpath(n.output,"replacement_entry_checks.toml"),"w") do io;TOML.print(io,turnover_checks);end
end
