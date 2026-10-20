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
pjump=Parameters(gamma=1.0,lambda0=20.)
r3,d3,e3=simulate_firm(ws,pjump,n,z,axes,3.,2026;policies=policy2)
@test d3["innovations"]>0
@test all(e->e.after==p.premium,filter(e->e.kind=="innovation",e3))
@test all(e->any(r->r.time==e.time && r.variance==p.sigma_Q^2,r3),filter(e->e.kind=="innovation",e3))

@test all(r->r.research<=r.rd_cashflow_cap+1e-14,r3)
@test d3["rd_max_cashflow_violation"]==0

println("Adaptive simulation and cash-flow feasibility checks passed")

@testset "Fixed-cost participation in interpolated simulations" begin
    fixed=Parameters(F_R=0.04,gamma=1.)
    high=merge(policy,(research_gain=linear(axes,fill(10.,ws.g.dims)),dt=n.dt))
    low=merge(high,(research_gain=linear(axes,fill(0.1,ws.g.dims)),))
    for x in (-0.5,-0.2,0.,0.1,0.8), a in (-0.31,0.,0.23)
        u=[x,a,0.,0.015]
        k=interpolated_research(high,u,fixed,z)
        cap=DU.rd_cashflow_cap(x,a,exp(z[2]),exp(z[3]),fixed)
        @test DU.rd_expense(k,fixed)<=cap+1e-12
        @test interpolated_research(low,u,fixed,z)==0
        @test k≈DU.tensor_rd(10.,fixed,n.dt;cap=cap)
    end
    rs,ds,es=simulate_firm(ws,fixed,n,z,axes,1.,2026;policies=high)
    @test ds["F_R"]==0.04
    @test ds["rd_max_cashflow_violation"]<=1e-12
    @test all(r->r.research_expenditure==DU.rd_expense(r.research,fixed),rs)
    @test all(r->r.net_flow≈r.operating_profit-r.research_expenditure,rs)
end

@testset "Adaptive exit and replacement callbacks" begin
    turn=Parameters(replacement_entry=true,F_R=.04,gamma=1.)
    turnover_policy=merge(policy,(
        research_gain=linear(axes,fill(-1.,ws.g.dims)),dt=n.dt,
        survival=linear(axes,[x+.08 for x in axes[1],a in axes[2],b in axes[3],v in axes[4]]),
        entry_price=0.))
    ws.mu .=0
    ws.mu[3,LinearIndices(ws.g.dims[2:4])[3,3,1]]=1.
    rows,diagnostics,events=simulate_firm(ws,turn,n,z,axes,10.,2026;policies=turnover_policy)
    @test diagnostics["exits"]>=1
    @test diagnostics["entries"]==diagnostics["exits"]
    @test diagnostics["price_adjustments"]==0
    @test diagnostics["innovations"]==0
    @test last(rows).firm_id==1+diagnostics["exits"]
    @test last(rows).cumulative_menu_cost==0
    @test diagnostics["rd_max_cashflow_violation"]==0
    for event in filter(e->e.kind=="entry",events)
        @test any(r->r.time==event.time && r.belief==turn.premium && r.variance==turn.sigma_Q^2,rows)
    end
end
