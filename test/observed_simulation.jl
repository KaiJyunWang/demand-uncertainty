# Integration tests for observed-quality adaptive SDE and impulse callbacks.
const OBSERVED_PLOT_LIBRARY_ONLY=true
include(joinpath(@__DIR__,"..","scripts","plot_observed_bgp.jl"))
using Test
@testset "Observed-quality adaptive simulation and replacement" begin
    p,_=OQ.config(joinpath(@__DIR__,"..","calibration_bgp.toml"))
    n=OQ.Settings(nx=31,nz=31,nq=31)
    G=.0765536970790363
    ws=OQ.workspace(p,n,G)
    result=OQ.evaluate!(ws,p,n,log(.3007360211353526),G)
    @test maximum(abs,result.full_residual)<1e-5
    r=result.report;pol=policies(ws,p,n,r)
    rows,events,diag=simulate(ws,p,n,r,pol,.25,2026;initial_state=[-.35,-5.,-5.,0.,1.])
    @test count(e->e.kind=="exit",events)>=1
    @test count(e->e.kind=="entry",events)==count(e->e.kind=="exit",events)
    @test diag["rd_max_cashflow_violation"]<1e-12
    @test any(row->row.firm_id==2,rows)
    entry=first(filter(e->e.kind=="entry",events))
    @test entry.after!=p.premium # realization is observed; it is not reset to its mean
    rows,events,diag=simulate(ws,p,n,r,pol,2.,2026;noise=false,innovation_at_zero=true)
    @test diag["innovations"]==0
    @test maximum(abs(row.log_quality-p.premium) for row in rows)<1e-10
    @test maximum(abs(row.log_taste-p.premium*(-expm1(-row.time))) for row in rows)<1e-5
    @test diag["rd_max_cashflow_violation"]<1e-12
end
