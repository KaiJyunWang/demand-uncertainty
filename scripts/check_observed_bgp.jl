include(joinpath(@__DIR__,"..","src","ObservedQuality.jl"))
using .ObservedQuality, Serialization, TOML, LinearAlgebra
BLAS.set_num_threads(1)
const OQ=ObservedQuality
path=isempty(ARGS) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : ARGS[1]
p,n=OQ.config(path);saved=deserialize(joinpath(n.output,"observed_state.jls"));r=saved.report
@assert saved.signature==OQ.signature(path) "Stale cache"
@assert r["information"]=="observed_quality" && r["state_dimensions"]==3
@assert r["grid_counts"]==[n.nx,n.nz,n.nq]
@assert r["converged"] && r["max_equilibrium_residual"]<n.tolerance
for f in fieldnames(OQ.Parameters)
    @assert r["parameter_"*string(f)]==getfield(p,f)
end
ws=OQ.workspace(p,n,saved.theta[2]);ws.V.=saved.V;ws.mu.=saved.mu;ws.policy=saved.policy;ws.entry.=saved.entry
OQ.tensor_bellman!(ws.work[5],ws.V,ws,p,n,r["Y"],r["w"],r["G"])
bellman=maximum(abs,ws.work[5].-ws.V)
OQ.tensor_forward!(ws.work[5],ws.mu,ws)
distribution=sum(abs,ws.work[5].-ws.mu)
residual,checked=OQ.moments(ws,p,n,r["C"],r["Y"],r["w"],r["G"])
@assert bellman<1e-9 && distribution<1e-8 && norm(residual,Inf)<n.tolerance
@assert checked["rd_max_cashflow_violation"]<1e-12 && checked["rd_max_kkt_residual"]<1e-7
@assert checked["research"]>0 && checked["rd_active_mass"]>0 && checked["G"]>0
@assert checked["entry_rate"]==checked["exit_rate"]>0
@assert checked["entry_value"]>0 && checked["exit_max_value"]==0
@assert isapprox(checked["variable_research"]+checked["fixed_research"],checked["research"];atol=1e-12)
@assert abs(sum(ws.mu)-1)<1e-12 && minimum(ws.mu)>=0
# Boundary diagnostics are reported explicitly; domain refinement is required if these fail.
@assert checked["clipping"]<1e-5
@assert checked["price_drift_clipping"]<1e-7
@assert abs(checked["split_growth_error"])<1e-5
@assert abs(checked["split_taste_error"])<1e-5
nx,nz,nq=ws.g.dims
out=reshape(ws.mu.*ws.policy.exit,nx,nz,nq)
total=sum(out)
checks=Dict{String,Any}("passed"=>true,"information"=>"observed_quality",
    "bellman_residual"=>bellman,"distribution_residual"=>distribution,
    "max_equilibrium_residual"=>norm(residual,Inf),
    "quality_lower_boundary_exit_fraction"=>sum(out[:,:,1])/total,
    "price_lower_boundary_exit_fraction"=>sum(out[1,:,:])/total)
merge!(checks,Dict(k=>v for (k,v) in checked if occursin("error",k)||occursin("violation",k)||occursin("clipping",k)))
open(joinpath(n.output,"checks.toml"),"w") do io;TOML.print(io,checks);end
println(checks)
