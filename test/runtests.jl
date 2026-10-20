using Test, LinearAlgebra, SparseArrays, DemandUncertainty
const DU=DemandUncertainty

@testset "parameters and provisional calibration" begin
    p,n,ini=load_config(joinpath(@__DIR__,"fixtures","transition.toml"))
    @test p.chi ≈ (6/7)*3^7
    @test p.pi_target ≈ log(1.02)
    @test p.omega==0.5
    @test_throws AssertionError DU.validate(Parameters(omega=1.0),n,ini)
    g=make_grid(p,n,ini)
    @test minimum(diff(g.axes[4]))>1e-10
end

@testset "Gaussian filtering, Inada policy, and time-step limit" begin
    p=Parameters()
    v=.04; h=.1; sigma=.2
    f=filter_moments(.1,-.1,v,h,sigma)
    @test f.variance ≈ inv(inv(v)+h/sigma^2)
    @test f.var_b+f.variance ≈ v
    @test f.cov_ab ≈ v*(1-exp(-h))
    @test f.var_a*f.var_b-f.cov_ab^2 >= -1e-14
    @test filter_moments(0.,0.,0.,h,sigma).cov_ab==0
    @test filter_moments(0.,0.,0.,h,sigma).variance==0
    @test rd_policy(-1.,p)==0
    for gain in (1e-4,.1,1.,5.)
        k=rd_policy(gain,p)
        @test k>0
        @test p.lambda0*p.omega*k^(p.omega-1)*gain ≈ 1
        kd=DU.step_rd(gain,p,1e-6)
        @test kd ≈ k rtol=1e-5
        @test DU.step_rd(gain,p,.25)<=k
        @test -kd+p.lambda0*kd^p.omega*gain >= 0
    end
end

@testset "positive transitions and Bellman/forward duality" begin
    p=Parameters(); n=Numerics(nx=5,na=5,nb=5,nv=3,steps=2,dt=.1,price_temperature=0.0)
    ini=Initial(); g=make_grid(p,n,ini)
    P,c=transition(g,p,n,p.pi_target,0.)
    R,cr=transition(g,p,n,p.pi_target,0.;reset=true)
    @test minimum(nonzeros(P))>=0
    @test maximum(abs.(sum(P,dims=2).-1))<1e-12
    @test maximum(abs.(sum(R,dims=2).-1))<1e-12
    # Reset must put exactly the prescribed b and variance into the distribution.
    b=[s[3] for s in g.states]; v=[s[4] for s in g.states]
    @test maximum(abs.(R*b.-p.premium))<1e-12
    @test maximum(abs.(R*v.-p.sigma_Q^2))<1e-12
    mu=initial_distribution(g,ini,0.)
    @test sum(mu)≈1
    next=[sin(s[1])+s[2]^2+2s[3]+s[4] for s in g.states]
    V,policy=bellman_step(next,g,p,n,1/3,6/7,p.pi_target,0.,p.rho)
    f=forward_step(mu,policy,n)
    @test minimum(f.next)>=0
    @test sum(f.next)≈1 atol=1e-12
    @test 0<=f.innovation_probability<=1
    @test f.research>=0
    # All terms in the backward objective equal the forward expected cash flow.
    reward=0.0
    for j in eachindex(mu)
        k=policy.target[j]; x,a,_,_=g.states[k]
        profit=(exp(x)-(6/7)/p.A)*exp(a-p.epsilon*x)/3
        reward+=mu[j]*(n.dt*(profit-policy.k[k])-p.K*(j!=k))
    end
    @test dot(mu,V) ≈ reward+exp(-p.rho*n.dt)*dot(f.next,next) atol=1e-11
end

@testset "equilibrium residual accounting and terminal policy evaluation" begin
    p=Parameters(); n=Numerics(nx=5,na=5,nb=3,nv=3,steps=2,terminal_iterations=50)
    ini=Initial(); g=make_grid(p,n,ini); z=baseline_guess(p,n,ini)
    result=equilibrium_residual(z,g,p,n,ini;details=true)
    @test length(result.residual)==length(z)
    @test all(isfinite,result.residual)
    @test result.mass_error<1e-12
    @test result.terminal.converged
    @test result.terminal.residual<n.terminal_tolerance
    for row in result.moments
        @test row.budget_error ≈ row.C*result.residual[n.steps+1+Int(round(row.time/n.dt))+1] atol=1e-10
    end
end

@testset "price smoothing bounds and exact forward accounting" begin
    p=Parameters(); ini=Initial()
    n=Numerics(nx=5,na=3,nb=3,nv=3,steps=1,price_temperature=1e-4)
    n0=Numerics(nx=5,na=3,nb=3,nv=3,steps=1,price_temperature=0.0)
    g=make_grid(p,n,ini)
    next=[cos(s[1])+s[3] for s in g.states]
    V,policy=bellman_step(next,g,p,n,1/3,6/7,p.pi_target,0.,p.rho)
    V0,_=bellman_step(next,g,p,n0,1/3,6/7,p.pi_target,0.,p.rho)
    @test minimum(V-V0)>=-1e-12
    @test maximum(V-V0)<=n.price_temperature*log(n.nx)+1e-12
    @test maximum(abs.(sum(policy.pricing,dims=2).-1))<1e-12
    mu=initial_distribution(g,ini,0.)
    f=forward_step(mu,policy,n)
    profit=[(exp(x)-(6/7)/p.A)*exp(a-p.epsilon*x)/3 for (x,a,b,v) in g.states]
    rhs=n.dt*dot(f.post,profit-policy.k)-p.K*n.dt*f.frequency+
        exp(-p.rho*n.dt)*dot(f.next,next)+dot(mu,policy.entropy)
    @test dot(mu,V)≈rhs atol=1e-11
    @test sum(f.next)≈1 atol=1e-12
end

@testset "balanced growth transformations and invariant distribution" begin
    p=Parameters(gamma=1.0,chi=(6/7)*3^6)
    rates=bgp_growth_rates(.012,p)
    @test rates.consumption ≈ .012/(p.epsilon-1)
    @test rates.nominal_price ≈ rates.quality_adjusted_inflation+rates.consumption
    @test rates.interest ≈ p.rho+p.gamma*rates.consumption+p.pi_target
    K=sparse([.9 .1; .2 .8])
    mu,err=invariant_distribution(K)
    @test mu≈[2/3,1/3]
    @test err<1e-12
    n=Numerics(nx=5,na=5,nb=5,nv=3,steps=1,dt=.1,
               amin=-2.,amax=2.,bmin=-2.,bmax=2.)
    ini=Initial(); g=make_grid(p,n,ini)
    growth=.03
    P,_=transition(g,p,n,p.pi_target+growth/(p.epsilon-1),0.;quality_growth=growth)
    j=LinearIndices(g.dims)[3,3,3,3]
    beta=[s[3] for s in g.states]
    taste=[s[2] for s in g.states]
    @test (P*beta)[j] ≈ -growth*n.dt atol=1e-12
    @test (P*taste)[j] ≈ -growth*n.dt atol=1e-12
    @test_throws ArgumentError solve_balanced_growth(Parameters(),n,ini;verbose=false)
end

@testset "stationary balanced-growth equilibrium end to end" begin
    p=Parameters(gamma=1.0,chi=(6/7)*3^6)
    n=Numerics(nx=5,na=5,nb=5,nv=3,max_iterations=15,tolerance=1e-5)
    result=solve_balanced_growth(p,n;verbose=false)
    @test result.converged
    @test maximum(abs.(result.residual))<n.tolerance
    @test result.moments.G>0
    @test result.moments.C>0
    @test result.moments.N>0
    @test result.firm.error<n.terminal_tolerance
    @test result.distribution_error<1e-9
    @test sum(result.distribution)≈1
    @test minimum(result.distribution)>=0
    @test abs(result.moments.budget_error)<1e-5
    mktempdir() do directory
        d=write_bgp_results(result,p,n;directory=directory)
        @test d["converged"]
        @test isfile(joinpath(directory,"distribution.csv"))
        @test isfile(joinpath(directory,"balanced_growth.toml"))
    end
end

@testset "tensor backend equals full sparse BGP discretization" begin
    p=Parameters(gamma=1.0,chi=(6/7)*3^6)
    n=Numerics(nx=5,na=5,nb=5,nv=3,terminal_tolerance=1e-11)
    g=make_grid(p,n);G=.006;Y=.338;w=.87
    ws=DU.tensor_workspace(p,n,Initial(),G)
    V=reshape([sin(s[1])+s[2]^2+2s[3]+s[4] for s in g.states],n.nx,:)
    P0,c0=transition(g,p,n,p.pi_target+G/6,0.;quality_growth=G)
    P1,c1=transition(g,p,n,p.pi_target+G/6,0.;reset=true,quality_growth=G)
    center=log(7/6*w)
    direct,policy=bellman_step(vec(V),g,p,n,Y,w,p.pi_target+G/6,0.,p.rho;
        matrices=(P0,c0,P1,c1),center=center,cashflow_constraint=true)
    output=similar(V)
    DU.tensor_bellman!(output,V,ws,p,n,Y,w,G)
    @test vec(output)≈direct atol=1e-11
    @test vec(ws.policy.k)≈policy.k atol=1e-10
    @test vec(ws.policy.keep.+ws.policy.mix)≈ones(length(V)) atol=1e-14
    mu=copy(ws.mu)
    flow=forward_step(vec(mu),policy,n)
    DU.tensor_forward!(output,mu,ws)
    @test vec(output)≈flow.next atol=1e-13
    @test sum(output)≈1 atol=1e-13
    @test minimum(output)>=0
    @test DU.tensor_rd(.5,p,n.dt)≈DU.step_rd(.5,p,n.dt) rtol=1e-10
    copyto!(ws.V,V)
    aggregate=[log(.335),log(Y),log(w),G]
    large=DU.tensor_residual!(ws,aggregate,p,n)
    small=bgp_residual(aggregate,g,p,n;details=true)
    @test norm(large.residual-small.residual,Inf)<1e-7
    @test maximum(abs,vec(ws.V)-small.firm.value)<1e-9
    @test large.moments.mean_variance≈small.moments.mean_variance atol=1e-9
    for field in (:rd_binding_mass,:rd_zero_cashflow_mass,:rd_cashflow_shortfall,:rd_mean_financing_multiplier)
        @test getproperty(large.moments,field)≈getproperty(small.moments,field) atol=1e-9
    end
    @test large.moments.rd_max_cashflow_violation==0
    @test large.moments.rd_max_kkt_residual<1e-8
    @test large.distribution_error<1e-9
    @test prod(ws.g.dims)==length(g.states)
    mktempdir() do folder
        DU.write_solution_csv(Dict("G"=>G,"nx"=>51,"converged"=>true),joinpath(folder,"solution.csv"))
        lines=readlines(joinpath(folder,"solution.csv"))
        @test lines==["G,converged,nx","0.006,true,51"]
    end
end

@testset "requested 51-point grids and Julia default compatibility" begin
    p,n,ini=load_config(joinpath(@__DIR__,"fixtures","learning.toml"))
    @test (n.nx,n.na,n.nb,n.nv)==(51,51,51,51)
    defaults=Numerics()
    @test (defaults.nx,defaults.na,defaults.nb,defaults.nv)==(51,51,51,51)
    tg=DU.tensor_grid(p,n,ini)
    @test tg.dims==(51,51,51,51)
    @test prod(tg.dims)==6_765_201
    @test VERSION>=v"1.13"
end

@testset "matrix-free aggregate root and CSV report" begin
    p=Parameters(gamma=1.0,chi=(6/7)*3^6)
    n=Numerics(nx=5,na=5,nb=5,nv=3,terminal_tolerance=1e-10,
               max_iterations=12,tolerance=1e-5,output=mktempdir())
    result=DU.solve_large_bgp(p,n,Initial();verbose=false)
    @test result.converged
    @test norm(result.residual,Inf)<n.tolerance
    @test result.distribution_error<1e-8
    report=write_bgp_results(result,p,n)
    @test report["rd_cashflow_constraint"]
    @test report["rd_max_cashflow_violation"]==0
    @test report["rd_max_kkt_residual"]<1e-8
    @test report["nx"]==5
    @test report["state_count"]==375
    @test report["julia_version"]==string(VERSION)
    @test report["consumption_growth_pct_per_year"]≈100*expm1(result.moments.growth)
    @test report["menu_expenditure_share"]≈p.K*result.moments.price_frequency/result.moments.Y
    lines=readlines(joinpath(n.output,"solution.csv"))
    @test length(lines)==2
    @test occursin("max_equilibrium_residual",lines[1])
end

include("cashflow_constraints.jl")

include("fixed_rd_cost.jl")

include("replacement_entry.jl")

include("innovation_scale.jl")

include("observed_quality.jl")
