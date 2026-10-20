@testset "lambda0=3 finite-step participation" begin
 p=Parameters(lambda0=3.,F_R=.04)
 for gain in (-1.,0.,.05,.1,.15,.3,1.,10.), cap in (0.,.04,.041,.05,.1,1.)
  k=DU.tensor_rd(gain,p,.25;cap=cap)
  @test k≈DU.step_rd(gain,p,.25;cap=cap) atol=1e-10 rtol=1e-9
  @test DU.rd_expense(k,p)<=cap+1e-12
  best=maximum(DU.step_rd_surplus(x,gain,p,.25) for x in range(0,max(cap-p.F_R,0),length=2001))
  @test DU.step_rd_surplus(k,gain,p,.25)>=best-1e-10
 end
end
@testset "Reject zero-growth warm start when research is profitable" begin
 p=Parameters(lambda0=3.,F_R=.0001,gamma=1.,replacement_entry=true,chi=(6/7)*3^6)
 n=Numerics(nx=5,na=5,nb=5,nv=3,terminal_tolerance=1e-10,max_iterations=20,output=mktempdir())
 r=DU.solve_large_bgp(p,n,Initial();guess=[log(.336),log(.338),log(.87),0.],verbose=false)
 @test r.converged
 @test r.unknowns[4]>0
 @test r.moments.research>0
 @test r.distribution_error<1e-8
end

@testset "Profit-scale invariance and labour-scale calibration" begin
 p=Parameters(lambda0=3.,F_R=.0001,gamma=1.,replacement_entry=true,chi=(6/7)*3^6)
 n=Numerics(nx=5,na=5,nb=5,nv=3,terminal_tolerance=1e-10,max_iterations=20,output=mktempdir(),bgp_branch="positive")
 r=DU.solve_large_bgp(p,n,Initial();guess=[log(.336),log(.338),log(.87),.04],verbose=false)
 @test r.converged
 z=r.unknowns;G=z[4]
 center=log(p.epsilon/(p.epsilon-1)*exp(z[3])/p.A)
 F=exp(z[2]+(1-p.epsilon)*center);w0=p.A*(p.epsilon-1)/p.epsilon
 canonical=DU.tensor_residual!(r.workspace,[log(.33),log(F),log(w0),G],p,n)
 S=exp(canonical.residual[1]);L=canonical.moments.N*p.A/F
 c0=log(S)/(p.epsilon-1);Y=F*S
 C=Y-p.K*canonical.moments.price_frequency-canonical.moments.research
 w=w0*exp(c0);N=F*exp(-c0)*L/p.A
 chi=w/(N^p.varphi*C^p.gamma)
 changed=Parameters(; (field=>(field==:chi ? chi : getfield(p,field)) for field in fieldnames(Parameters))...)
 V=copy(r.workspace.V)
 checked=DU.tensor_residual!(r.workspace,[log(C),log(Y),log(w),G],changed,n)
 @test maximum(abs,checked.residual[1:3])<1e-9
 @test abs(checked.residual[4])<n.tolerance
 @test maximum(abs,r.workspace.V.-V)<1e-8
 @test checked.moments.research≈canonical.moments.research atol=1e-10
 @test checked.moments.exit_rate≈canonical.moments.exit_rate atol=1e-10
 @test checked.moments.split_growth_error≈canonical.moments.split_growth_error atol=1e-10
 @test checked.moments.split_taste_mean_error≈canonical.moments.split_taste_mean_error atol=1e-10
end

@testset "Nonuniform grids retain central resolution and both tails" begin
 n=Numerics(nx=51,na=51,nb=51,nv=51,price_lower_tail_nodes=3,price_upper_tail_nodes=3,
    quality_lower_tail_nodes=10,quality_upper_tail_nodes=6,price_core_min=-.18,price_core_max=.12,
    xmin=-.35,xmax=.3,amin=-9.,amax=3.9,bmin=-8.,bmax=3.6,bgp_branch="positive")
 g=DU.tensor_grid(Parameters(sigma_Q=.5),n)
 @test g.dims==(51,51,51,51)
 @test all(a->all(diff(a).>0),g.axes)
 @test first(g.axes[1])==-.35 && last(g.axes[1])==.3
 @test first(g.axes[2])==-9 && last(g.axes[2])==3.9
 @test first(g.axes[3])==-8 && last(g.axes[3])==3.6
 @test minimum(diff(g.axes[1]))<.007
 @test 0. in g.axes[3]
end
