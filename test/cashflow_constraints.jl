using Test, DemandUncertainty, LinearAlgebra
const CFDU=DemandUncertainty
@testset "cash-flow constrained R&D optimization" begin
    p=Parameters(gamma=1.)
    for gain in (-1.,0.,.1,10.),cap in (0.,.0001,.1,10.)
        kd=CFDU.step_rd(gain,p,.25;cap=cap)
        kt=CFDU.tensor_rd(gain,p,.25;cap=cap)
        @test 0<=kt<=cap
        @test kt≈kd atol=1e-11 rtol=1e-10
        objective(k)=-k+(-expm1(-p.lambda0*k^p.omega*.25))*gain/.25
        @test objective(kt)+1e-10>=maximum(objective,range(0,cap;length=2001))
        @test rd_policy(gain,p;cap=cap)<=cap
        if kt>0
            marginal=p.lambda0*p.omega*kt^(p.omega-1)*gain*exp(-p.lambda0*kt^p.omega*.25)
            @test kt==cap ? marginal>=1-1e-9 : isapprox(marginal,1;atol=1e-9)
        end
    end
    @test CFDU.rd_cashflow_cap(log(.8),0.,1.,1.,p)==0
    @test CFDU.rd_cashflow_cap(log(1.2),0.,1.,1.,p)>0
    n=Numerics(nx=5,na=5,nb=5,nv=3)
    g=make_grid(p,n);G=.006;Y=.338;w=.87;center=log(7/6*w)
    ws=CFDU.tensor_workspace(p,n,Initial(),G)
    V=reshape([10s[3] for s in g.states],n.nx,:)
    output=similar(V)
    CFDU.tensor_bellman!(output,V,ws,p,n,Y,w,G)
    cap=max.(ws.policy.profit,0.)
    @test maximum(ws.policy.k-cap)<=0
    @test all(ws.policy.k[cap.==0].==0)
    gain=exp(-p.rho*n.dt).*(ws.work[3]-ws.work[2])
    stats=CFDU.rd_financing_statistics(vec(ws.policy.k),vec(ws.policy.profit),vec(gain),vec(ws.mu),p,n.dt)
    @test stats.rd_binding_mass>0
    @test stats.rd_cashflow_shortfall>0
    @test stats.rd_max_cashflow_violation==0
    @test stats.rd_max_kkt_residual<1e-8
end
