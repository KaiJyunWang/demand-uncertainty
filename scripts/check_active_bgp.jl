# Full-grid verification for an equilibrium with endogenous R&D and turnover.
using DemandUncertainty,Serialization,TOML,LinearAlgebra
const DU=DemandUncertainty
config=isempty(ARGS) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : ARGS[1]
p,n,ini=load_config(config)
saved=deserialize(joinpath(n.output,"plot_state.jls"));z=saved.signature[2]
@assert saved.signature==DU.bgp_cache_signature(config,z)
ws=DU.tensor_workspace(p,n,ini,z[4]);ws.V .=saved.V;ws.mu .=saved.mu
DU.tensor_bellman!(ws.work[5],ws.V,ws,p,n,exp(z[2]),exp(z[3]),z[4])
bellman=maximum(abs,ws.work[5].-ws.V)
@assert bellman<1e-9
@assert minimum(ws.V)>=-1e-9
@assert maximum(abs,ws.V.*ws.policy.exit)<1e-9
@assert abs(sum(ws.entry)-1)<1e-12 && sum(ws.entry.*ws.V)>0
@assert sum(ws.entry.*ws.policy.exit)<1e-12
DU.tensor_forward!(ws.work[5],ws.mu,ws)
distribution_error=sum(abs,ws.work[5].-ws.mu)
@assert distribution_error<1e-8
@assert abs(sum(ws.mu)-1)<1e-12 && minimum(ws.mu)>=0
residual,m=DU.tensor_moments(ws,p,n,exp(z[1]),exp(z[2]),exp(z[3]),z[4])
@assert maximum(abs,residual)<n.tolerance
@assert m.rd_max_cashflow_violation<1e-12
@assert m.rd_max_kkt_residual<1e-8
@assert m.research>0 && m.rd_active_mass>0 && m.G>0 && m.exit_rate>0
@assert m.exit_rate==m.entry_rate
@assert abs(m.split_growth_error)<max(1e-5,10*m.clipping)
@assert abs(m.split_taste_mean_error)<max(1e-5,10*m.clipping)
@assert abs(m.research-m.variable_research-m.fixed_research)<1e-12
post=ws.work[5];DU.tensor_operating_forward!(post,ws.mu,ws)
@assert abs(sum(post)-sum(ws.mu))<1e-12
shape=ws.g.dims;mass=reshape(ws.mu,shape)
boundaries=Dict{String,Any}()
for (dim,name) in enumerate(("price","taste","belief","variance"))
    dim==4 && continue
    boundaries[name*"_lower_mass"]=sum(selectdim(mass,dim,1))
    boundaries[name*"_upper_mass"]=sum(selectdim(mass,dim,shape[dim]))
end
exitmass=reshape(ws.mu.*ws.policy.exit,shape)
boundaries["exit_mass_at_lower_belief_boundary"]=sum(selectdim(exitmass,3,1))
boundaries["exit_mass_at_lower_price_boundary"]=sum(selectdim(exitmass,1,1))
# A tiny boundary tail need not vanish exactly. Compare it with actual exit
# mass: at least 99.99% of exits must occur strictly inside each lower bound.
exit_per_step=m.exit_rate*n.dt
boundaries["exit_lower_belief_boundary_fraction"]=boundaries["exit_mass_at_lower_belief_boundary"]/exit_per_step
boundaries["exit_lower_price_boundary_fraction"]=boundaries["exit_mass_at_lower_price_boundary"]/exit_per_step
@assert boundaries["exit_lower_belief_boundary_fraction"]<1e-4
@assert boundaries["exit_lower_price_boundary_fraction"]<1e-4
@assert boundaries["belief_lower_mass"]<1e-6
@assert boundaries["belief_upper_mass"]<1e-6
@assert m.clipping<1e-5
unaffordable=sum(post[j] for j in eachindex(post) if max(ws.policy.profit[j],0.)<=p.F_R)
affordable_inactive=sum(post[j] for j in eachindex(post) if max(ws.policy.profit[j],0.)>p.F_R && ws.policy.k[j]==0)
checks=merge(boundaries,Dict("rd_cashflow_exclusion_mass"=>unaffordable,
    "rd_affordable_inactive_mass"=>affordable_inactive,"lambda0"=>p.lambda0,"F_R"=>p.F_R,
    "positive_branch"=>true,"bellman_residual"=>bellman,"distribution_residual"=>distribution_error,
    "equilibrium_residual"=>maximum(abs,residual),"research"=>m.research,
    "rd_active_mass"=>m.rd_active_mass,"exit_rate"=>m.exit_rate,"entry_rate"=>m.entry_rate,
    "G"=>m.G,"entry_value"=>sum(ws.entry.*ws.V),"exit_policy_nodes"=>count(>(0),ws.policy.exit),
    "quality_clipping"=>m.clipping,"split_growth_error"=>m.split_growth_error,
    "split_taste_mean_error"=>m.split_taste_mean_error,"rd_max_cashflow_violation"=>m.rd_max_cashflow_violation,
    "scope"=>"finite grid and finite time step; boundary-exit checks are not a continuum convergence proof"))
open(joinpath(n.output,"active_equilibrium_checks.toml"),"w") do io;TOML.print(io,checks);end
println(checks)
