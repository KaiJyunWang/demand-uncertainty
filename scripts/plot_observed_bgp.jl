# Three-state observed-quality figures and adaptive SciML firm simulations.
include(joinpath(@__DIR__,"..","src","ObservedQuality.jl"))
using .ObservedQuality, TOML, Serialization, Random, Interpolations
using CairoMakie, StochasticDiffEq, SciMLBase, LinearAlgebra
using OrdinaryDiffEqTsit5: Tsit5
BLAS.set_num_threads(1)
const OQ=ObservedQuality
linear(ax,values)=extrapolate(interpolate(ax,values,Gridded(Linear())),Flat())
function write_csv(path,rows)
    isempty(rows) && return
    open(path,"w") do io
        println(io,join(string.(keys(first(rows))),","))
        for row in rows;println(io,join(values(row),","));end
    end
end
function policies(ws,p,n,r)
    OQ.continuation!(ws.work[2],ws.work[3],ws.V,ws)
    prob=ws.policy.probability;disc=exp(-p.rho*n.dt)
    stay=n.dt.*(ws.policy.profit.-OQ.rd_expense.(ws.policy.k,Ref(p))).+
        disc.*((1 .-prob).*ws.work[2].+prob.*ws.work[3])
    best=vec(maximum(stay,dims=1));target=[ws.g.axes[1][argmax(view(stay,:,j))] for j in eachindex(best)]
    entrytarget=[ws.g.axes[1][argmax(view(ws.V,:,j))] for j in eachindex(best)]
    ax=ws.g.axes;dims=ws.g.dims
    (gain=linear(ax,reshape(disc.*(ws.work[3].-ws.work[2]),dims)),
        surplus=linear(ax,reshape(reshape(best,1,:).-p.K.-stay,dims)),
        survival=linear(ax,reshape(max.(stay,reshape(best,1,:).-p.K),dims)),
        target=linear(ax[2:3],reshape(target,dims[2:3])),
        entry=linear(ax[2:3],reshape(entrytarget,dims[2:3])),axes=ax,
        center=log(p.epsilon/(p.epsilon-1)*r["w"]/p.A))
end
function simulate(ws,p,n,r,policy,years,seed;noise=true,innovation_at_zero=false,initial_state=nothing)
    rng=Xoshiro(seed);G=r["G"];g=r["g"];trend=p.pi_target+g
    weights=vec(ws.mu);j=searchsortedfirst(cumsum(weights),rand(rng));j=min(j,length(weights))
    x=mod1(j,n.nx);block=cld(j,n.nx);z,q=ws.g.qstates[block]
    # Price gap, taste gap, known quality gap, integrated hazard, firm identity.
    u0=[ws.g.axes[1][x],z,q,0.,1.]
    if !noise
        u0=[policy.entry(0.,0.),0.,0.,0.,1.]
        innovation_at_zero && (u0[3]=p.premium)
    end
    initial_state===nothing || (u0=Float64.(initial_state))
    events=NamedTuple{(:time,:kind,:before,:after),Tuple{Float64,String,Float64,Float64}}[]
    threshold=Ref(randexp(rng));noise_seed=Int(rand(rng,UInt32))
    profit(u)=OQ.operating_profit(u[1]+policy.center,u[2],r["Y"],r["w"],p)
    rd(u)=OQ.tensor_rd(policy.gain(u[1],u[2],u[3]),p,n.dt;cap=max(profit(u),0.))
    surplus(u)=policy.surplus(u[1],u[2],u[3])
    survival(u)=policy.survival(u[1],u[2],u[3])
    function price!(u,t)
        before=u[1];u[1]=policy.target(u[2],u[3])
        push!(events,(time=Float64(t),kind="price",before=before,after=u[1]))
    end
    function replace!(u,t)
        push!(events,(time=Float64(t),kind="exit",before=u[3],after=0.))
        if !noise
            error("Deterministic original-firm path reached exit at $t; shorten horizon")
        end
        u[2]=p.premium;u[3]=p.premium+p.sigma_Q*randn(rng)
        u[1]=policy.entry(u[2],u[3]);u[5]+=1
        threshold[]=u[4]+randexp(rng)
        push!(events,(time=Float64(t),kind="entry",before=0.,after=u[3]))
        survival(u)>0 || error("Nonpositive entry value")
    end
    function drift!(du,u,_,t)
        du[1]=-trend;du[2]=u[3]-u[2]-G;du[3]=-G
        du[4]=noise ? p.lambda0*rd(u)^p.omega : 0.;du[5]=0.
    end
    function diffusion!(du,u,_,t)
        fill!(du,0.);du[2,1]=noise ? p.sigma_a : 0.
    end
    function innovate!(integrator)
        u=integrator.u;before=u[3];u[3]=p.premium+p.sigma_Q*randn(rng)
        push!(events,(time=Float64(integrator.t),kind="innovation",before=before,after=u[3]))
        threshold[]+=randexp(rng)
        if survival(u)<=0;replace!(u,integrator.t)
        elseif surplus(u)>0;price!(u,integrator.t);end
    end
    pricecb=ContinuousCallback((u,t,i)->surplus(u),i->price!(i.u,i.t),nothing;save_positions=(true,true))
    exitcb=ContinuousCallback((u,t,i)->survival(u),nothing,i->replace!(i.u,i.t);save_positions=(true,true))
    innovcb=ContinuousCallback((u,t,i)->u[4]-threshold[],innovate!,nothing;save_positions=(true,true))
    if survival(u0)<=0;replace!(u0,0.)
    elseif surplus(u0)>0;price!(u0,0.);end
    callbacks=CallbackSet(exitcb,pricecb,innovcb)
    if noise
        problem=SDEProblem(drift!,diffusion!,u0,(0.,Float64(years));noise_rate_prototype=zeros(5,1))
        solution=solve(problem,RKMilCommute();adaptive=true,abstol=1e-7,reltol=1e-5,
            seed=noise_seed,callback=callbacks,save_everystep=true,maxiters=10^7)
    else
        problem=ODEProblem(drift!,u0,(0.,Float64(years)))
        solution=solve(problem,Tsit5();abstol=1e-9,reltol=1e-7,
            callback=callbacks,save_everystep=true,maxiters=10^7)
    end
    SciMLBase.successful_retcode(solution) || error("Simulation failed: $(solution.retcode)")
    rows=NamedTuple[]
    for (t,u) in zip(solution.t,solution.u)
        k=rd(u);pi=profit(u);xactual=u[1]+policy.center
        push!(rows,(time=t,firm_id=round(Int,u[5]),price_gap=u[1],detrended_log_price=xactual,
            taste_gap=u[2],known_quality_gap=u[3],aggregate_price=exp(p.pi_target*t),
            nominal_price=exp(xactual+trend*t),taste_level=exp(u[2]+G*t),
            quality_level=exp(u[3]+G*t),log_taste=u[2]+G*t,log_quality=u[3]+G*t,
            variable_rd=k,total_rd=OQ.rd_expense(k,p),intensity=p.lambda0*k^p.omega,
            physical_sales=r["Y"]*exp(u[2]-p.epsilon*xactual),
            operating_profit=pi,profit_after_rd=pi-OQ.rd_expense(k,p)))
    end
    violation=maximum(max(row.total_rd-max(row.operating_profit,0.),0.) for row in rows)
    @assert violation<1e-12
    diag=Dict("information"=>"observed_quality","years"=>years,"seed"=>seed,
        "algorithm"=>(noise ? "adaptive RKMilCommute" : "adaptive Tsit5 (noise off)"),"accepted_steps"=>solution.stats.naccept,
        "rejected_steps"=>solution.stats.nreject,"saved_points"=>length(rows),
        "innovations"=>count(e->e.kind=="innovation",events),"exits"=>count(e->e.kind=="exit",events),
        "entries"=>count(e->e.kind=="entry",events),"rd_max_cashflow_violation"=>violation,
        "saved_points_outside_policy_domain"=>count(u->any(d->u[d]<first(policy.axes[d]) || u[d]>last(policy.axes[d]),1:3),solution.u),
        "policy_interpolation"=>"Interpolations.jl gridded linear, flat extrapolation; financing reoptimized at every solver stage",
        "noise"=>noise,"initial_innovation"=>innovation_at_zero,"policy_solution_dt"=>n.dt)
    rows,events,diag
end
function plot_path(rows,events,out;title="Observed quality — successive firms")
    mkpath(out);write_csv(joinpath(out,"firm_path.csv"),rows);write_csv(joinpath(out,"firm_events.csv"),events)
    t=getproperty.(rows,:time)
    f=Figure(size=(1600,1100));Label(f[0,1:2],title)
    for (index,(fields,label)) in enumerate((([:detrended_log_price],"Detrended log relative price"),
            ([:taste_gap,:known_quality_gap],"Taste and known quality gaps"),
            ([:variable_rd,:total_rd],"Quality-normalized R&D flow"),
            ([:operating_profit,:profit_after_rd],"Quality-normalized profit flow")))
        ax=Axis(f[cld(index,2),mod1(index,2)],xlabel="Years",ylabel=label)
        for field in fields;lines!(ax,t,getproperty.(rows,field);label=replace(string(field),"_"=>" "));end
        for kind in ("innovation","entry")
            ts=[e.time for e in events if e.kind==kind]
            isempty(ts) || vlines!(ax,ts;color=(kind=="entry" ? :purple : :gray,.45),linestyle=:dash)
        end
        length(fields)>1 && axislegend(ax;position=:rt)
    end
    save(joinpath(out,"firm_paths.png"),f);save(joinpath(out,"firm_paths.pdf"),f)
    f=Figure(size=(1500,900));Label(f[0,1:2],title*" — levels (P₀ = 1)")
    for (j,(fields,label)) in enumerate((([:nominal_price,:aggregate_price],"Nominal prices"),
            ([:taste_level,:quality_level],"Taste and long-run quality levels"),
            ([:log_taste,:log_quality],"Log taste and known log quality"),
            ([:total_rd],"Quality-normalized total R&D flow")))
        ax=Axis(f[cld(j,2),mod1(j,2)],xlabel="Years",ylabel=label)
        for field in fields;lines!(ax,t,getproperty.(rows,field);label=replace(string(field),"_"=>" "));end
        length(fields)>1 && axislegend(ax;position=:lt)
    end
    save(joinpath(out,"firm_levels.png"),f);save(joinpath(out,"firm_levels.pdf"),f)
end
function main(args)
    path=abspath(isempty(args) ? joinpath(@__DIR__,"..","calibration_bgp.toml") : args[1])
    p,n=OQ.config(path);saved=deserialize(joinpath(n.output,"observed_state.jls"))
    saved.signature==OQ.signature(path) || error("Stale observed-quality cache")
    saved.report["converged"] || error("Cannot plot unconverged equilibrium")
    ws=OQ.workspace(p,n,saved.theta[2]);ws.V.=saved.V;ws.mu.=saved.mu;ws.policy=saved.policy;ws.entry.=saved.entry
    r=saved.report;out=length(args)>3 ? args[4] : joinpath(n.output,"figures");mkpath(out)
    center=log(p.epsilon/(p.epsilon-1)*r["w"]/p.A)
    f=Figure(size=(1800,550));Label(f[0,1:3],"Observed-quality stationary equilibrium — $(n.nx) × $(n.nz) × $(n.nq)")
    marginalrows=NamedTuple[]
    for d in 1:3
        others=Tuple(setdiff(1:3,[d]));mass=vec(dropdims(sum(reshape(ws.mu,ws.g.dims),dims=others),dims=others))
        axis=ws.g.axes[d].+(d==1 ? center : 0.)
        gaps=diff(axis);widths=.9 .*vcat(first(gaps),min.(gaps[1:end-1],gaps[2:end]),last(gaps))
        ax=Axis(f[1,d],xlabel=("Detrended log relative price","Taste gap a − M","Known quality gap θ − M")[d],ylabel="Probability mass")
        barplot!(ax,axis,mass;width=widths)
        for j in eachindex(axis);push!(marginalrows,(state=d,node=j,value=axis[j],mass=mass[j]));end
    end
    save(joinpath(out,"equilibrium_marginals.png"),f);save(joinpath(out,"equilibrium_marginals.pdf"),f)
    write_csv(joinpath(out,"marginals.csv"),marginalrows)
    policy=policies(ws,p,n,r)
    fpolicy=Figure(size=(1500,600));Label(fpolicy[0,1:2],"Policy slice: current taste equals known long-run quality (z = q)")
    statecodes=zeros(n.nx,n.nq);research=zeros(n.nx,n.nq)
    for (i,x) in enumerate(ws.g.axes[1]),(j,q) in enumerate(ws.g.axes[3])
        statecodes[i,j]=policy.survival(x,q,q)<=0 ? 2 : policy.surplus(x,q,q)>0 ? 1 : 0
        cap=max(OQ.operating_profit(x+center,q,r["Y"],r["w"],p),0.)
        k=OQ.tensor_rd(policy.gain(x,q,q),p,n.dt;cap=cap)
        research[i,j]=statecodes[i,j]==2 ? 0. : OQ.rd_expense(k,p)
    end
    ax=Axis(fpolicy[1,1],xlabel="Price gap from flexible-price center",ylabel="Known quality gap")
    hm=heatmap!(ax,ws.g.axes[1],ws.g.axes[3],statecodes;colorrange=(0,2),colormap=[:white,:orange,:purple])
    Colorbar(fpolicy[1,2],hm;ticks=([0,1,2],["Continue","Reprice","Exit"]))
    ax2=Axis(fpolicy[1,3],xlabel="Price gap from flexible-price center",ylabel="Known quality gap")
    hm2=heatmap!(ax2,ws.g.axes[1],ws.g.axes[3],research)
    Colorbar(fpolicy[1,4],hm2;label="Total normalized R&D flow")
    save(joinpath(out,"exit_policy.png"),fpolicy);save(joinpath(out,"exit_policy.pdf"),fpolicy)
    horizons=length(args)>1 ? [parse(Float64,args[2])] : [300.,30.]
    seed=length(args)>2 ? parse(Int,args[3]) : 2026
    for years in horizons
        dest=years==300 || length(args)>3 ? out : joinpath(out,"30_year")
        rows,events,diag=simulate(ws,p,n,r,policy,years,seed)
        plot_path(rows,events,dest)
        open(joinpath(dest,"diagnostics.toml"),"w") do io;TOML.print(io,diag);end
        println("Simulation ",years," years: ",diag);flush(stdout)
    end
    rows,events,diag=simulate(ws,p,n,r,policy,10.,seed;noise=false,innovation_at_zero=true)
    dest=joinpath(out,"innovation_levels");plot_path(rows,events,dest;title="One observed successful innovation — subsequent noise shut down")
    open(joinpath(dest,"diagnostics.toml"),"w") do io;TOML.print(io,diag);end
end
if !isdefined(Main,:OBSERVED_PLOT_LIBRARY_ONLY)
    main(ARGS)
end
