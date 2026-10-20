using Test, LinearAlgebra, SparseArrays, Random, TOML
include(joinpath(@__DIR__,"..","src","ObservedQuality.jl"))
const OQ=ObservedQuality
@testset "Observed-quality primitives and stochastic operators" begin
    p,_=OQ.config(joinpath(@__DIR__,"..","calibration_bgp.toml"))
    n=OQ.Settings(nx=17,nz=21,nq=21,dt=.125)
    nodes,weights=OQ.normal_quadrature(n.draw_nodes)
    @test sum(weights)≈1
    @test dot(nodes,weights)≈0 atol=1e-14
    @test dot(nodes.^2,weights)≈1
    G=.07;ws=OQ.workspace(p,n,G);g=ws.g
    @test length(g.axes)==3
    @test length(ws.V)==17*21*21
    z=first.(g.qstates);q=last.(g.qstates)
    for kernel in (ws.K0,ws.K1)
        @test minimum(kernel.outgoing.nzval)>=0
        @test vec(sum(kernel.outgoing,dims=1))≈ones(length(q))
        rng=Xoshiro(123);v=rand(rng,size(ws.V)...);mu=rand(rng,size(ws.mu)...)
        back=similar(v);forward=similar(v)
        OQ.quality_apply!(back,v,kernel.outgoing)
        OQ.quality_apply!(forward,mu,kernel.incoming)
        @test dot(back,mu)≈dot(v,forward)
    end
    rng=Xoshiro(42);v=rand(rng,size(ws.V)...)
    e0=similar(v);e1=similar(v);direct=similar(v);shifted=similar(v)
    OQ.continuation!(e0,e1,v,ws)
    OQ.price_value!(shifted,v,ws.drift)
    OQ.quality_apply!(direct,shifted,ws.K1.outgoing)
    @test e1≈direct atol=1e-12
    OQ.quality_apply!(direct,v,ws.K1.incoming)
    OQ.quality_apply!(shifted,v,ws.K0.incoming)
    OQ.redraw_forward!(e1,shifted,ws)
    @test e1≈direct atol=1e-12
    j=argmin(z.^2+q.^2)
    @test dot(z,ws.K0.outgoing[:,j])≈q[j]+(z[j]-q[j])*exp(-n.dt)-G*n.dt atol=1e-12
    @test dot(q,ws.K0.outgoing[:,j])≈q[j]-G*n.dt atol=1e-12
    @test dot(q,ws.K1.outgoing[:,j])≈p.premium atol=1e-9
    @test dot(z,ws.K1.outgoing[:,j])≈dot(z,ws.K0.outgoing[:,j]) atol=1e-12
    # Observing draws requires integrating nonlinear value, rather than evaluating
    # value at the mean draw. Grid interpolation may add, but not remove, variance.
    @test dot(q.^2,ws.K1.outgoing[:,j])>p.premium^2+.9*p.sigma_Q^2
    # Price optima vary with quality. Entry optimizes separately for each draw.
    for (j,(_,qj)) in enumerate(g.qstates),x in axes(ws.V,1)
        ws.V[x,j]=2-(g.axes[1][x]-.04*qj)^2
    end
    OQ.update_entry!(ws,p)
    @test sum(ws.entry)≈1
    @test dot(vec(sum(ws.entry,dims=1)),q)≈p.premium atol=1e-9
    @test count(>(1e-10),vec(sum(ws.entry,dims=2)))>1
    # The incumbent's killed value operator must never include successor value.
    fill!(ws.policy.exit,1.)
    OQ.tensor_policy_value!(ws.work[5],ws.V,ws)
    @test maximum(abs,ws.work[5])==0
    OQ.tensor_forward!(ws.work[5],ws.mu,ws)
    @test sum(ws.work[5])≈sum(ws.mu)
    @test OQ.tensor_rd(0.,p,n.dt;cap=1.)==0
    @test OQ.tensor_rd(10.,p,n.dt;cap=p.F_R)==0
    for gain in (.01,.2,1.),cap in (.005,.02,.1)
        k=OQ.tensor_rd(gain,p,n.dt;cap=cap)
        @test OQ.rd_expense(k,p)<=cap+1e-14
        if k>0
            @test OQ.step_rd_surplus(k,gain,p,n.dt)>0
        end
    end
end
