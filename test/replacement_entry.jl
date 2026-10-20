@testset "Exit value and replacement population accounting" begin
    p=Parameters(replacement_entry=true,F_R=.04,gamma=1.)
    n=Numerics(nx=7,na=7,nb=7,nv=3,xmin=-.3,xmax=.075,amin=-9.,amax=1.5,
        bmin=-8.,bmax=1.2,price_lower_tail_nodes=2,quality_lower_tail_nodes=2)
    ws=DU.tensor_workspace(p,n,Initial(),0.)
    @test all(a->length(a)==7,ws.g.axes[1:3])
    @test all(a->all(diff(a).>0),ws.g.axes)
    ws.V .=1
    DU.update_entry!(ws,p)
    @test sum(ws.entry)≈1
    @test sum(ws.entry.*reshape([q[1] for q in ws.g.qstates],1,:))≈p.premium
    @test sum(ws.entry.*reshape([q[2] for q in ws.g.qstates],1,:))≈p.premium
    @test sum(ws.entry.*reshape([q[3] for q in ws.g.qstates],1,:))≈p.sigma_Q^2
    ws.policy.exit .=1
    ws.policy.keep .=1
    ws.policy.probability .=0
    out=similar(ws.V)
    DU.tensor_forward!(out,ws.mu,ws)
    @test sum(out)≈1
    @test minimum(out)>=0
    DU.tensor_policy_value!(out,ws.V,ws)
    @test maximum(abs,out)==0 # successors never accrue to exiting incumbents
    ws.policy.profit .=1
    DU.tensor_reward!(out,ws,p,n)
    @test maximum(abs,out)==0
    ws.V .=0
    err,it=DU.tensor_firm!(ws,p,n,.334,.858,0.)
    @test err<1e-9
    @test minimum(ws.V)>=0
    @test count(>(0),ws.policy.exit)>0
    @test maximum(ws.V.*ws.policy.exit)==0
    @test sum(ws.entry.*ws.V)>0
    @test sum(ws.entry.*ws.policy.exit)==0
    DU.tensor_forward!(out,ws.mu,ws)
    @test sum(out)≈sum(ws.mu) atol=1e-12
    @test minimum(out)>=0
    result=DU.solve_zero_research_bgp(p,n,Initial();verbose=false)
    @test result!==nothing && result.converged
    @test result.moments.exit_rate==result.moments.entry_rate
    @test result.moments.exit_max_value==0
    @test result.moments.entry_value>0
end

@testset "Turnover-adjusted discrete quality and taste moments" begin
 p=Parameters(replacement_entry=true,premium=.2,sigma_Q=.5)
 n=Numerics(nx=9,na=9,nb=9,nv=9,price_lower_tail_nodes=1,price_upper_tail_nodes=1,
    quality_lower_tail_nodes=1,quality_upper_tail_nodes=1,price_core_min=-.18,price_core_max=.12,
    xmin=-.35,xmax=.3,amin=-9.,amax=3.9,bmin=-8.,bmax=3.6)
 G=.04;ws=DU.tensor_workspace(p,n,Initial(),G)
 for q in axes(ws.V,2),x in axes(ws.V,1);ws.V[x,q]=1-ws.g.axes[1][x]^2;end
 DU.update_entry!(ws,p)
 ia=findfirst(==(.5),ws.g.axes[2]);ib=findfirst(==(0.),ws.g.axes[3]);ix=argmin(abs.(ws.g.axes[1]))
 q=LinearIndices(ws.g.dims[2:4])[ia,ib,1]
 fill!(ws.mu,0.);ws.mu[ix,q]=1
 ws.policy.keep .=1;ws.policy.exit .=0;ws.policy.exit[ix,q]=.3
 out=similar(ws.mu);DU.tensor_forward!(out,ws.mu,ws)
 b=reshape([s[2] for s in ws.g.qstates],1,:);a=reshape([s[1] for s in ws.g.qstates],1,:)
 h=n.dt;turn_b=.3*p.premium/h;turn_a=.3*(p.premium-.5)/h
 expected_b=h*turn_b-G*h
 expected_a=exp(-h)*(.5+h*turn_a)+(1-exp(-h))*h*turn_b-G*h
 @test sum(out)≈1 atol=1e-13
 @test sum(out.*b)≈expected_b atol=1e-12
 @test sum(out.*a)≈expected_a atol=1e-12
 @test sum(ws.entry.*b)≈p.premium atol=1e-13
 @test sum(ws.entry.*a)≈p.premium atol=1e-13
end
