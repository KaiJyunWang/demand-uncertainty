using DemandUncertainty, Test, Random
include(joinpath(@__DIR__,"..","scripts","bgp_diagnostics.jl"))
const DU=DemandUncertainty
p=Parameters(gamma=1.0)
n=Numerics(nx=5,na=5,nb=5,nv=5)
ws=DU.tensor_workspace(p,n,Initial(),.003)
z=[log(1/3),log(.335),log(.86),.003]
axes=ws.g.axes
pol=continuous_policies(ws,p,n,z,axes)
@test pol.research(axes[1][3],axes[2][3],axes[3][3],axes[4][3])≈ws.policy.k[3,LinearIndices(ws.g.dims[2:4])[3,3,3]]
# Deterministic impulse boundary and zero R&D isolate the diffusion/filter tests.
linear(ax,values)=extrapolate(interpolate(ax,values,Gridded(Linear())),Flat())
policy=(research=linear(axes,zeros(ws.g.dims)),
    surplus=linear(axes,[-x-.10 for x in axes[1],a in axes[2],b in axes[3],v in axes[4]]),
    target=linear(axes[2:4],zeros(ws.g.dims[2:4])))
ws.mu .=0
ws.mu[3,LinearIndices(ws.g.dims[2:4])[3,3,5]]=1.
r,d,e=simulate_firm(ws,p,n,z,axes,10.,2026;policies=policy)
r2,d2,e2=simulate_firm(ws,p,n,z,axes,10.,2026;policies=policy)
@test r==r2 && e==e2
@test d["adaptive"] && d["maximum_saved_step"]>d["minimum_saved_step"]
@test last(r).time≈10.
@test d["innovations"]==0
@test d["price_adjustments"]>=1
@test all(x->x.variance>=0 && x.research>=0 && x.sales>0,r)
@test all(x->isapprox(x.revenue,exp(x.price)*x.sales),r)
@test last(r).cumulative_menu_cost≈p.K*d["price_adjustments"]
# Nominal price remains constant outside price impulses (no grid rounding).
for j in 2:length(r)
    if r[j].cumulative_menu_cost==r[j-1].cumulative_menu_cost
        @test r[j].log_nominal_price≈r[j-1].log_nominal_price atol=1e-10
    end
end
@test last(r).variance≈p.sigma_Q^2/(1+p.sigma_Q^2*10/p.sigma_a^2) rtol=2e-3
# Force an observable innovation clock to validate callback resets and counts.
policy2=merge(policy,(research=linear(axes,fill(4.,ws.g.dims)),))
r3,d3,e3=simulate_firm(ws,p,n,z,axes,3.,2026;policies=policy2)
@test d3["innovations"]>0
@test all(e->e.after==p.premium,filter(e->e.kind=="innovation",e3))
@test all(e->any(r->r.time==e.time && r.variance==p.sigma_Q^2,r3),filter(e->e.kind=="innovation",e3))
println("Adaptive simulation diagnostics checks passed")
