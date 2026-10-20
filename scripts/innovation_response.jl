# julia --threads=16 --project=. scripts/innovation_response.jl [config] [years]
using DemandUncertainty, Serialization, TOML, SciMLBase, OrdinaryDiffEqTsit5, CairoMakie
include(joinpath(@__DIR__,"bgp_diagnostics.jl"))

"""Zero-innovation-noise skeleton, optionally forcing a single success at t=0.
The innovation draw equals its prior mean. Existing stochastic-equilibrium
policies remain fixed; Brownian shocks and all subsequent arrivals are absent.
"""
function deterministic_firm(p,z,policies,initial,years;innovation=false)
    u=copy(initial) # x-tilde, taste gap, belief gap, variance, menu costs, net cash
    if innovation
        u[3]=p.premium; u[4]=p.sigma_Q^2
    end
    impact=copy(u)
    events=NamedTuple{(:time,:before,:after),Tuple{Float64,Float64,Float64}}[]
    surplus(u)=policies.surplus(u[1],u[2],u[3],max(u[4],0.))
    function reset!(u,t)
        old=u[1];u[1]=policies.target(u[2],u[3],max(u[4],0.))
        u[5]+=p.K;u[6]-=p.K
        push!(events,(time=Float64(t),before=old,after=u[1]))
    end
    surplus(u)>0 && reset!(u,0.)
    G=z[4]; trend=p.pi_target+G/(p.epsilon-1)
    function drift!(du,u,_,t)
        v=max(u[4],0.)
        k=interpolated_research(policies,u,p,z)
        du[1]=-trend;du[2]=u[3]-u[2]-G;du[3]=-G
        du[4]=-v^2/p.sigma_a^2;du[5]=0.
        sales=exp(z[2]+u[2]-p.epsilon*u[1])
        du[6]=DemandUncertainty.operating_profit(u[1],u[2],exp(z[2]),exp(z[3]),p)-DemandUncertainty.rd_expense(k,p)
    end
    price_callback=ContinuousCallback((u,t,i)->surplus(u),i->reset!(i.u,i.t),nothing;
        save_positions=(true,true))
    exit_callback=ContinuousCallback((u,t,i)->policies.survival(u[1],u[2],u[3],max(u[4],0.)),nothing,i->terminate!(i);save_positions=(true,true))
    callback=p.replacement_entry ? CallbackSet(exit_callback,price_callback) : price_callback
    problem=ODEProblem(drift!,u,(0.,Float64(years)))
    solution=solve(problem,Tsit5();callback=callback,abstol=1e-10,reltol=1e-8,
        save_everystep=true,isoutofdomain=(u,p,t)->u[4]<0)
    SciMLBase.successful_retcode(solution) || error("ODE failed: $(solution.retcode)")
    (solution=solution,events=events,impact=impact)
end

function innovation_response(config,years)
    years>0 || error("Horizon must be positive")
    p,n,ini=load_config(config)
    z=TOML.parsefile(joinpath(n.output,"checkpoint.toml"))["unknowns"]
    cache=joinpath(n.output,"plot_state.jls")
    isfile(cache) || error("Run scripts/plot_bgp.jl to reconstruct the equilibrium cache first")
    saved=deserialize(cache)
    signature=DemandUncertainty.bgp_cache_signature(config,z)
    saved.signature==signature || error("Equilibrium cache is stale")
    ws=DemandUncertainty.tensor_workspace(p,n,ini,z[4]);ws.V.=saved.V;ws.mu.=saved.mu
    center=log(p.epsilon/(p.epsilon-1)*exp(z[3])/p.A)
    axes=(ws.g.axes[1].+center,ws.g.axes[2],ws.g.axes[3],ws.g.axes[4])
    policies=continuous_policies(ws,p,n,z,axes)
    v0=sum(sum(view(ws.mu,:,q))*ws.g.qstates[q][3] for q in eachindex(ws.g.qstates))
    initial=[policies.target(0.,0.,v0),0.,0.,v0,0.,0.]
    baseline=deterministic_firm(p,z,policies,initial,years)
    success=deterministic_firm(p,z,policies,initial,years;innovation=true)
    last(baseline.solution.t)>=years && last(success.solution.t)>=years || error("A deterministic firm exited; shorten the comparison horizon")
    # Observation times do not restrict the adaptive ODE integration steps.
    event_times=Set(e.time for path in (baseline,success) for e in path.events)
    times=sort!(unique(vcat(collect(range(0.,years;length=1001)),collect(event_times))))
    function quantities(u,t)
        k=interpolated_research(policies,u,p,z)
        sales=exp(z[2]+u[2]-p.epsilon*u[1])
        (taste=u[2]+z[4]*t,belief=u[3]+z[4]*t,variance=u[4],
         nominal_price=exp(u[1]+(p.pi_target+z[4]/(p.epsilon-1))*t),
         research=k,research_expenditure=DemandUncertainty.rd_expense(k,p),cashflow_cap=DemandUncertainty.rd_cashflow_cap(u[1],u[2],exp(z[2]),exp(z[3]),p),
         sales=sales,intensity=p.lambda0*k^p.omega,net_cash=u[6])
    end
    rows=NamedTuple[]
    function record(t,u0,u1,side)
        b=quantities(u0,t);s=quantities(u1,t)
        push!(rows,(time=t,side=side,
            baseline_taste=b.taste,innovation_taste=s.taste,taste_response=s.taste-b.taste,
            baseline_belief=b.belief,innovation_belief=s.belief,belief_response=s.belief-b.belief,
            baseline_variance=b.variance,innovation_variance=s.variance,variance_response=s.variance-b.variance,
            baseline_price=b.nominal_price,innovation_price=s.nominal_price,price_response_pct=100*(s.nominal_price/b.nominal_price-1),
            baseline_research=b.research,innovation_research=s.research,
            baseline_research_expenditure=b.research_expenditure,innovation_research_expenditure=s.research_expenditure,research_response=s.research-b.research,
            baseline_cashflow_cap=b.cashflow_cap,innovation_cashflow_cap=s.cashflow_cap,
            baseline_sales=b.sales,innovation_sales=s.sales,sales_response_pct=100*(s.sales/b.sales-1),
            intensity_response=s.intensity-b.intensity,net_cash_response=s.net_cash-b.net_cash))
    end
    record(0.,initial,initial,"before_impact")
    for t in times
        if t>0 && t in event_times
            record(t,baseline.solution(t;continuity=:left),success.solution(t;continuity=:left),"left")
        end
        record(t,baseline.solution(t;continuity=:right),success.solution(t;continuity=:right),"right")
    end
    # Exact zero-noise identities independently verify filtering and jump timing.
    analytic_taste_error=maximum(abs(r.taste_response-p.premium*(-expm1(-r.time))) for r in rows)
    analytic_belief_error=maximum(abs(r.belief_response-p.premium) for r in rows if r.side!="before_impact")
    analytic_variance_error=maximum(abs(r.innovation_variance-p.sigma_Q^2/(1+p.sigma_Q^2*r.time/p.sigma_a^2)) for r in rows if r.side!="before_impact")
    max_cashflow_violation=maximum(max(r.baseline_research_expenditure-r.baseline_cashflow_cap,
        r.innovation_research_expenditure-r.innovation_cashflow_cap,0.) for r in rows)
    @assert max_cashflow_violation<=1e-12
    @assert analytic_taste_error<1e-7
    @assert analytic_belief_error<1e-8
    @assert analytic_variance_error<1e-8
    @assert success.impact[2]==initial[2] && success.impact[3]==p.premium && success.impact[4]==p.sigma_Q^2
    outside=Dict(name=>count(u->any(d->u[d]<axes[d][1] || u[d]>axes[d][end],1:4),path.solution.u)
        for (name,path) in (("baseline",baseline),("innovation",success)))
    out=joinpath(n.output,"figures","innovation_response");mkpath(out)
    open(joinpath(out,"innovation_response.csv"),"w") do io
        println(io,join(keys(first(rows)),","))
        for row in rows;println(io,join(values(row),","));end
    end
    open(joinpath(out,"diagnostics.toml"),"w") do io
        TOML.print(io,Dict("lambda0"=>p.lambda0,"replacement_entry"=>p.replacement_entry,"F_R"=>p.F_R,"years"=>years,"algorithm"=>"adaptive Tsit5",
            "initial_state"=>initial,"innovation_premium"=>p.premium,
            "innovation_time"=>0.,"innovation_draw"=>"prior mean; no idiosyncratic draw",
            "brownian_noise"=>false,"subsequent_innovations"=>false,
            "policies"=>"fixed cash-flow-constrained stochastic-equilibrium policies",
            "rd_cashflow_constraint"=>true,"rd_max_cashflow_violation"=>max_cashflow_violation,
            "baseline_price_adjustments"=>length(baseline.events),"innovation_price_adjustments"=>length(success.events),
            "baseline_accepted_steps"=>baseline.solution.stats.naccept,"innovation_accepted_steps"=>success.solution.stats.naccept,
            "outside_policy_domain"=>outside,"taste_identity_error"=>analytic_taste_error,
            "belief_identity_error"=>analytic_belief_error,"variance_identity_error"=>analytic_variance_error))
    end
    rows,out
end

function plot_innovation_response(rows,out)
    CairoMakie.activate!()
    t=[r.time for r in rows];series(key)=[getproperty(r,key) for r in rows]
    f=Figure(size=(1200,1000),fontsize=16)
    specifications=((:innovation_taste,"Undetrended taste aᵢₜ"),
        (:innovation_belief,"Posterior mean bᵢₜ"),
        (:innovation_variance,"Posterior variance"),
        (:innovation_price,"Nominal price (P₀ = 1)"),
        (:innovation_research_expenditure,"Detrended total R&D expenditure (level)"),
        (:innovation_sales,"Physical sales (level)"))
    for (j,(key,label)) in enumerate(specifications)
        ax=Axis(f[cld(j,2),mod1(j,2)],ylabel=label,xlabel=j>=5 ? "Years after success" : "")
        lines!(ax,t,series(key);color=:steelblue,linewidth=2)
    end
    Label(f[0,:],"One successful innovation at t = 0 — absolute firm levels",fontsize=21)
    Label(f[4,:],"No Brownian shocks or later arrivals; mean innovation draw; equilibrium policies held fixed.",fontsize=14)
    for ext in ("png","pdf")
        save(joinpath(out,"innovation_levels.$ext"),f)
    end
    f=Figure(size=(1200,1000),fontsize=16)
    specs=((:taste,"Undetrended taste aᵢₜ"),(:belief,"Posterior mean bᵢₜ"),
        (:variance,"Posterior variance"),(:price,"Nominal price (P₀ = 1)"),
        (:research_expenditure,"Detrended total R&D expenditure"),(:sales,"Physical sales"))
    for (j,(key,label)) in enumerate(specs)
        ax=Axis(f[cld(j,2),mod1(j,2)],ylabel=label,xlabel=j>=5 ? "Years after success" : "")
        lines!(ax,t,series(Symbol("baseline_",key));color=:darkorange,linestyle=:dash,linewidth=2,label="No innovation")
        lines!(ax,t,series(Symbol("innovation_",key));color=:steelblue,linewidth=2,label="Success at t = 0")
        j==1 && axislegend(ax;position=:rb,labelsize=12)
    end
    Label(f[0,:],"Deterministic firm paths with and without a successful innovation",fontsize=21)
    for ext in ("png","pdf");save(joinpath(out,"innovation_paths.$ext"),f);end
end

if abspath(PROGRAM_FILE)==@__FILE__
    config=length(ARGS)>0 ? ARGS[1] : joinpath(@__DIR__,"..","calibration_bgp.toml")
    years=length(ARGS)>1 ? parse(Float64,ARGS[2]) : 10.
    rows,out=innovation_response(config,years)
    plot_innovation_response(rows,out)
    println("Saved innovation response figures and verified analytic identities: ",abspath(out))
end
