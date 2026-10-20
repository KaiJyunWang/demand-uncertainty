@testset "Fixed R&D operating cost and participation" begin
    p=Parameters(F_R=0.04)
    for gain in (-1.,0.,0.1,0.8,1.,2.,10.), cap in (0.,0.02,0.04,0.041,0.08,1.,Inf), dt in (0.01,0.25)
        k=DemandUncertainty.step_rd(gain,p,dt;cap=cap)
        kt=DemandUncertainty.tensor_rd(gain,p,dt;cap=cap)
        @test kt≈k atol=1e-11 rtol=1e-10
        @test DemandUncertainty.rd_expense(k,p)<=cap+1e-12
        upper=isfinite(cap) ? max(cap-p.F_R,0.) : max(1.,k*2)
        best=maximum(DemandUncertainty.step_rd_surplus(x,gain,p,dt) for x in range(0,upper,length=2001))
        @test DemandUncertainty.step_rd_surplus(k,gain,p,dt)>=best-1e-10
        @test cap<=p.F_R ? k==0 : true
    end
    @test DemandUncertainty.rd_policy(0.79,p)==0
    @test DemandUncertainty.rd_policy(0.81,p)>0
end
@testset "Positive fixed cost sparse/tensor accounting" begin
    p=Parameters(F_R=0.04,gamma=1.,chi=(6/7)*3^6)
    n=Numerics(nx=5,na=5,nb=5,nv=3,terminal_tolerance=1e-11)
    g=make_grid(p,n); G=.006; Y=.338; w=.87
    ws=DU.tensor_workspace(p,n,Initial(),G)
    z=[log(.335),log(Y),log(w),G]
    large=DU.tensor_residual!(ws,z,p,n)
    small=bgp_residual(z,g,p,n;details=true)
    @test norm(large.residual-small.residual,Inf)<1e-7
    @test maximum(abs,vec(ws.V)-small.firm.value)<1e-9
    for m in (large.moments,small.moments)
        @test m.research≈m.variable_research+p.F_R*m.rd_active_mass atol=1e-12
        @test m.fixed_research≈p.F_R*m.rd_active_mass
        @test m.rd_max_cashflow_violation<1e-12
        @test m.rd_max_kkt_residual<1e-8
    end
    @test large.moments.research≈small.moments.research atol=1e-9
end
@testset "Zero-growth equilibrium and closed-class stationarity" begin
    p=Parameters(F_R=0.04,gamma=1.)
    n=Numerics(nx=5,na=5,nb=5,nv=3,xmin=-.075,xmax=.075,terminal_tolerance=1e-10)
    result=DU.solve_zero_research_bgp(p,n,Initial();verbose=false)
    @test result!==nothing
    @test result.converged
    @test result.unknowns[4]==0
    @test result.moments.rd_active_mass==0
    @test result.moments.research==0
    @test result.moments.mean_b==0
    @test result.moments.mean_variance==0
    @test result.distribution_error<1e-9
    @test sum(result.workspace.mu)≈1
    qs=findall(q->q[2]!=0 || q[3]!=0,result.workspace.g.qstates)
    @test sum(result.workspace.mu[:,qs])==0
end
