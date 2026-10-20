# Continuous-time simulation of the filtered firm state under interpolated policies.
using Random, Interpolations, StochasticDiffEq, SciMLBase

function draw(rng,weights)
    u=rand(rng)*sum(weights); total=0.0
    for j in eachindex(weights)
        total+=weights[j]
        u<=total && return j
    end
    lastindex(weights)
end

"""Interpolate R&D, the intervention surplus, and the optimal reset price.
The impulse rule uses the unsmoothed continuation values underlying the numerical
softmax, rather than interpreting a per-quarter choice probability as a hazard.
Flat extrapolation applies only to policy evaluation; states are never clipped.
"""
function continuous_policies(ws,p,n,z,axes)
    DemandUncertainty.tensor_bellman!(ws.work[5],ws.V,ws,p,n,exp(z[2]),exp(z[3]),z[4])
    # tensor_bellman! leaves no-reset/reset expectations in work[2:3].
    disc=exp(-(p.rho+(p.gamma-1)*z[4]/(p.epsilon-1))*n.dt)
    prob=ws.policy.probability
    continuation=n.dt.*(ws.policy.profit.-DemandUncertainty.rd_expense.(ws.policy.k,Ref(p))).+
        disc.*((1 .-prob).*ws.work[2].+prob.*ws.work[3])
    best=vec(maximum(continuation;dims=1))
    target=[axes[1][argmax(view(continuation,:,q))] for q in eachindex(best)]
    surplus=reshape(reshape(best,1,:).-p.K.-continuation,ws.g.dims)
    linear(ax,values)=extrapolate(interpolate(ax,values,Gridded(Linear())),Flat())
    (research=linear(axes,reshape(ws.policy.k,ws.g.dims)),
     survival=linear(axes,reshape(max.(continuation,reshape(best,1,:).-p.K),ws.g.dims)),
     entry_price=sum(ws.entry.*reshape(axes[1],:,1)),
     research_gain=linear(axes,reshape(disc.*(ws.work[3].-ws.work[2]),ws.g.dims)),
     surplus=linear(axes,surplus),
     target=linear(axes[2:4],reshape(target,ws.g.dims[2:4])),axes=axes,dt=n.dt)
end

# Multilinear interpolation need not preserve a nonlinear cash-flow bound.
# Enforce the exact bound at every ODE/SDE stage and observation, including
# evaluations outside the policy grid. Equilibrium policies are already constrained.
function interpolated_research(policies,u,p,z)
    requested=max(0.,policies.research(u[1],u[2],u[3],max(u[4],0.)))
    cap=DemandUncertainty.rd_cashflow_cap(u[1],u[2],exp(z[2]),exp(z[3]),p)
    if hasproperty(policies,:research_gain)
        gain=policies.research_gain(u[1],u[2],u[3],max(u[4],0.))
        return DemandUncertainty.tensor_rd(gain,p,policies.dt;cap=cap)
    end
    p.F_R==0 || error("Positive fixed costs require interpolated innovation gains")
    min(requested,cap)
end

function simulate_firm(ws,p,n,z,axes,years,seed;abstol=1e-7,reltol=1e-5,
        policies=continuous_policies(ws,p,n,z,axes))
    rng=Xoshiro(seed)
    flat=draw(rng,vec(ws.mu)); nx=ws.g.dims[1]
    x=mod1(flat,nx); q=cld(flat,nx)
    a,b,v=ws.g.qstates[q]
    # State: x-tilde, a-M, b-M, v, integrated innovation hazard,
    # slot cumulative menu expenditure, cumulative net cash, firm identity.
    u0=[axes[1][x],a,b,v,0.,0.,0.,1.]
    events=NamedTuple{(:time,:kind,:before,:after),Tuple{Float64,String,Float64,Float64}}[]
    life_times=Float64[]
    threshold=Ref(randexp(rng))
    noise_seed=Int(rand(rng,UInt32))
    G=z[4]; price_trend=p.pi_target+G/(p.epsilon-1)
    rd(u)=interpolated_research(policies,u,p,z)
    surplus(u)=policies.surplus(u[1],u[2],u[3],max(u[4],0.))
    survival(u)=!p.replacement_entry ? Inf : policies.survival(u[1],u[2],u[3],max(u[4],0.))
    function price_reset!(u,t)
        before=u[1]
        u[1]=policies.target(u[2],u[3],max(u[4],0.))
        u[6]+=p.K; u[7]-=p.K
        push!(events,(time=Float64(t),kind="price",before=before,after=u[1]))
    end
    function drift!(du,u,_,t)
        k=rd(u); vv=max(u[4],0.)
        du[1]=-price_trend
        du[2]=u[3]-u[2]-G
        du[3]=-G
        du[4]=-vv^2/p.sigma_a^2
        du[5]=p.lambda0*k^p.omega
        du[6]=0.;du[8]=0.
        sales=exp(z[2]+u[2]-p.epsilon*u[1])
        du[7]=DemandUncertainty.operating_profit(u[1],u[2],exp(z[2]),exp(z[3]),p)-DemandUncertainty.rd_expense(k,p)
    end
    function diffusion!(du,u,_,t)
        fill!(du,0.)
        # Exactly ONE Brownian innovation drives both taste and posterior mean.
        du[2,1]=p.sigma_a
        du[3,1]=max(u[4],0.)/p.sigma_a
    end
    function innovate!(integrator)
        u=integrator.u
        before=u[3];u[3]=p.premium;u[4]=p.sigma_Q^2
        push!(events,(time=Float64(integrator.t),kind="innovation",before=before,after=u[3]))
        # Drawing exponential thresholds implements a state-dependent Poisson clock.
        threshold[]+=randexp(rng)
        surplus(u)>0 && price_reset!(u,integrator.t)
    end
    function replace_state!(u,t)
        t=Float64(t)
        push!(events,(time=t,kind="exit",before=u[3],after=0.))
        push!(life_times,t);u[8]+=1
        u[1]=policies.entry_price;u[2]=p.premium;u[3]=p.premium;u[4]=p.sigma_Q^2
        threshold[]=u[5]+randexp(rng)
        push!(events,(time=t,kind="entry",before=0.,after=u[3]))
        survival(u)>0 || error("Entrant has nonpositive continuation value")
    end
    replace!(integrator)=replace_state!(integrator.u,integrator.t)
    exit_cb=ContinuousCallback((u,t,i)->survival(u),nothing,replace!;
        save_positions=(true,true))
    price_cb=ContinuousCallback((u,t,i)->surplus(u),
        i->price_reset!(i.u,i.t),nothing;save_positions=(true,true))
    innovation_cb=ContinuousCallback((u,t,i)->u[5]-threshold[],
        innovate!,nothing;save_positions=(true,true))
    if p.replacement_entry && survival(u0)<=0
        replace_state!(u0,0.)
    else
        surplus(u0)>0 && price_reset!(u0,0.)
    end
    problem=SDEProblem(drift!,diffusion!,u0,(0.,Float64(years));
        noise_rate_prototype=zeros(8,1))
    # No dt, dtmax, or saveat: the adaptive solver chooses the integration grid.
    solution=solve(problem,RKMilCommute();adaptive=true,abstol=abstol,reltol=reltol,
        seed=noise_seed,callback=p.replacement_entry ? CallbackSet(exit_cb,price_cb,innovation_cb) : CallbackSet(price_cb,innovation_cb),save_everystep=true,
        isoutofdomain=(u,p,t)->u[4]<0,maxiters=10^7)
    SciMLBase.successful_retcode(solution) || error("Firm SDE failed: $(solution.retcode), t=$(last(solution.t)), stats=$(solution.stats), events=$(length(events))")
    rows=NamedTuple[]
    for (t,u) in zip(solution.t,solution.u)
        k=rd(u); sales=exp(z[2]+u[2]-p.epsilon*u[1])
        operating=DemandUncertainty.operating_profit(u[1],u[2],exp(z[2]),exp(z[3]),p)
        push!(rows,(time=t,firm_id=round(Int,u[8]),price=u[1],log_nominal_price=u[1]+price_trend*t,
            aggregate_price=exp(p.pi_target*t),
            nominal_price=exp(u[1]+price_trend*t),
            relative_price=exp(u[1]+(G/(p.epsilon-1))*t),
            quality_mean=G*t,taste_level=u[2]+G*t,
            taste_multiplier=exp(u[2]+G*t),belief_level=u[3]+G*t,
            taste=u[2],belief=u[3],variance=max(u[4],0.),research=k,
            research_expenditure=DemandUncertainty.rd_expense(k,p),rd_active=Int(k>0),
            intensity=p.lambda0*k^p.omega,sales=sales,revenue=exp(u[1])*sales,
            operating_profit=operating,rd_cashflow_cap=max(operating,0.),
            rd_cap_binding=Int(k>0 && DemandUncertainty.rd_expense(k,p)>=operating*(1-1e-9)),net_flow=operating-DemandUncertainty.rd_expense(k,p),
            cumulative_menu_cost=u[6],cumulative_net_cash=u[7]))
    end
    outside=count(u->any(d->u[d]<axes[d][1] || u[d]>axes[d][end],1:4),solution.u)
    increments=filter(>(0.),diff(solution.t))
    diagnostics=Dict("simulation"=>"adaptive continuous-time filtered SDE with impulse callbacks",
        "replacement_entry"=>p.replacement_entry,"exits"=>length(life_times),"entries"=>length(life_times),
        "path_unit"=>"product slot; firm_id changes at replacement",
        "lambda0"=>p.lambda0,"F_R"=>p.F_R,"algorithm"=>"RKMilCommute","rd_cashflow_constraint"=>true,
        "rd_max_cashflow_violation"=>maximum(max(r.research_expenditure-r.rd_cashflow_cap,0.) for r in rows),
        "rd_binding_saved_points"=>sum(r.rd_cap_binding for r in rows),"adaptive"=>true,"abstol"=>abstol,"reltol"=>reltol,
        "saved_points"=>length(rows),"accepted_steps"=>solution.stats.naccept,
        "rejected_steps"=>solution.stats.nreject,"minimum_saved_step"=>minimum(increments),
        "maximum_saved_step"=>maximum(increments),"saved_points_outside_policy_domain"=>outside,
        "policy_extrapolation"=>"flat; continuous states are not clipped",
        "price_rule"=>"interpolated unsmoothed intervention surplus crosses zero",
        "innovation_timing"=>"root of integrated intensity minus exponential threshold",
        "price_adjustments"=>count(e->e.kind=="price",events),
        "innovations"=>count(e->e.kind=="innovation",events))
    rows,diagnostics,events
end
