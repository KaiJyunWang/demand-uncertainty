module DemandUncertainty

using LinearAlgebra, SparseArrays, TOML, Printf
export Parameters, Numerics, Initial, Grid, load_config, make_grid, filter_moments,
       rd_policy, initial_distribution, transition, bellman_step, forward_step,
       solve_equilibrium, write_results, equilibrium_residual, firm_path, baseline_guess

Base.@kwdef struct Parameters
    rho::Float64 = 0.04
    gamma::Float64 = 2.0
    epsilon::Float64 = 7.0
    varphi::Float64 = 5.0
    A::Float64 = 1.0
    chi::Float64 = (6/7)*3^7
    K::Float64 = 0.0015
    sigma_a::Float64 = 0.20
    lambda0::Float64 = 0.50
    omega::Float64 = 0.50
    premium::Float64 = 0.05
    sigma_Q::Float64 = 0.20
    pi_target::Float64 = log(1.02)
    phi_pi::Float64 = 1.5
end
Base.@kwdef struct Initial
    mean_a::Float64 = -0.03
    sd_a::Float64 = 0.12
    mean_b::Float64 = -0.03
    variance::Float64 = 0.04
    log_nominal_price::Float64 = 0.0
end
Base.@kwdef struct Numerics
    dt::Float64 = 0.25
    steps::Int = 8
    nx::Int = 51
    na::Int = 51
    nb::Int = 51
    nv::Int = 51
    xmin::Float64 = -0.45
    xmax::Float64 = 0.45
    amin::Float64 = -0.90
    amax::Float64 = 0.90
    bmin::Float64 = -0.80
    bmax::Float64 = 0.80
    price_temperature::Float64 = 1e-5 # numerical log-sum-exp smoothing, not a preference parameter
    terminal_consumption::Float64 = 1/3
    terminal_value::String = "frozen"
    terminal_iterations::Int = 500
    terminal_tolerance::Float64 = 1e-7
    max_iterations::Int = 35
    tolerance::Float64 = 1e-5
    jacobian_refresh::Int = 8
    output::String = "results"
end

function load_config(path)
    d = TOML.parsefile(path)
    p = Parameters(; (Symbol(k)=>v for (k,v) in d["economic"])...)
    n = Numerics(; (Symbol(k)=>v for (k,v) in d["numerical"])...)
    ini = Initial(; (Symbol(k)=>v for (k,v) in d["initial"])...)
    validate(p,n,ini)
    p,n,ini
end
function validate(p,n,ini)
    @assert p.rho > 0 && p.gamma > 0 && p.epsilon > 1 && p.varphi > 0
    @assert p.A > 0 && p.chi > 0 && p.K > 0 && p.sigma_a > 0
    @assert p.lambda0 >= 0 && 0 < p.omega < 1 && p.sigma_Q > 0
    @assert p.premium >= 0 && p.phi_pi > 1
    @assert n.dt > 0 && n.steps >= 1 && min(n.nx,n.na,n.nb,n.nv) >= 2
    @assert n.xmin < n.xmax && n.amin < n.amax && n.bmin < n.bmax
    @assert n.price_temperature >= 0
    @assert n.terminal_consumption > 0 && ini.variance > 0 && ini.sd_a >= 0
    @assert n.terminal_value in ("frozen", "zero")
    @assert n.max_iterations > 0 && n.jacobian_refresh > 0
end

struct Grid
    axes::NTuple{4,Vector{Float64}}
    dims::NTuple{4,Int}
    states::Vector{NTuple{4,Float64}}
end
function make_grid(p,n,ini=Initial())
    ax=(collect(range(n.xmin,n.xmax,length=n.nx)),
        collect(range(n.amin,n.amax,length=n.na)),
        collect(range(n.bmin,n.bmax,length=n.nb)),
        # Exact reset and initial variance nodes; a nonuniform variance grid.
        sort!(unique!(round.(vcat(collect(range(0,max(ini.variance,p.sigma_Q^2),length=n.nv)),
                            [ini.variance,p.sigma_Q^2]); digits=14))))
    dims=length.(ax)
    states=[Tuple(ax[d][I[d]] for d in 1:4) for I in CartesianIndices(dims)]
    Grid(ax,dims,vec(states))
end

# Nonnegative multilinear interpolation. Outside points are clamped and diagnosed.
function bracket(axis, z)
    j=clamp(searchsortedlast(axis,z),1,length(axis)-1)
    weight=clamp((z-axis[j])/(axis[j+1]-axis[j]),0.0,1.0)
    j,weight
end
function deposit!(indices, weights, g, state, scale=1.0)
    bw=ntuple(d->bracket(g.axes[d],state[d]),4)
    linear=LinearIndices(g.dims)
    for mask in 0:15
        ind=ntuple(d->bw[d][1]+((mask>>(d-1))&1),4)
        w=scale
        for d in 1:4
            w*=((mask>>(d-1))&1)==1 ? bw[d][2] : 1-bw[d][2]
        end
        if w>0
            push!(indices,linear[ind...]); push!(weights,w)
        end
    end
end
const NODES=(-sqrt(3.0),0.0,sqrt(3.0))
const WEIGHTS=(1/6,2/3,1/6)

"""Exact conditional moments of (a',b',v') absent a quality reset."""
function filter_moments(a,b,v,dt,sigma)
    e=exp(-dt)
    vn=v/(1+v*dt/sigma^2)
    va=v*(1-e)^2+sigma^2*(-expm1(-2dt))/2
    vb=max(v-vn,0.0)
    cab=v*(1-e)
    (mean_a=e*a+(1-e)*b, mean_b=b, variance=vn,
     var_a=va, var_b=vb, cov_ab=cab)
end

"""Row-stochastic, positive quadrature approximation; reset occurs at step end."""
function transition(g,p,n,inflation,market; reset=false, quality_growth=0.0)
    rows=Int[]; cols=Int[]; vals=Float64[]
    m=length(g.states)
    sizehint!(rows,60m); sizehint!(cols,60m); sizehint!(vals,60m)
    outside=zeros(m)
    for (j,(x,a,b,v)) in enumerate(g.states)
        mom=filter_moments(a,b,v,n.dt,p.sigma_a)
        la=sqrt(mom.var_a)
        lb=mom.cov_ab/la
        lc=sqrt(max(mom.var_b-lb^2,0.0))
        inds=Int[]; ws=Float64[]
        for i in 1:3, k in 1:3
            an=mom.mean_a-quality_growth*n.dt+la*NODES[i]
            bn=reset ? market+p.premium : b-quality_growth*n.dt+lb*NODES[i]+lc*NODES[k]
            vn=reset ? p.sigma_Q^2 : mom.variance
            point=(x-inflation*n.dt,an,bn,vn)
            w=WEIGHTS[i]*WEIGHTS[k]
            # Variance is bounded by design. Diagnose price/taste/belief clipping.
            if any(point[d]<first(g.axes[d])-1e-12 || point[d]>last(g.axes[d])+1e-12 for d in 1:3)
                outside[j]+=w
            end
            deposit!(inds,ws,g,point,w)
        end
        append!(rows,fill(j,length(inds))); append!(cols,inds); append!(vals,ws)
    end
    sparse(rows,cols,vals,m,m),outside
end

"""Continuous-time optimal R&D, conditional on innovation value gain G."""
rd_policy(G,p)=G<=0 || p.lambda0==0 ? 0.0 : (p.lambda0*p.omega*G)^(1/(1-p.omega))

# Exact optimizer for the discrete step's probability 1-exp(-lambda(k)*dt).
# It converges to rd_policy as dt -> 0 and never silently caps R&D or intensities.
function step_rd(G,p,dt)
    (G<=0 || p.lambda0==0) && return 0.0
    hi=rd_policy(G,p); lo=0.0
    for _ in 1:45
        mid=(lo+hi)/2
        marginal=p.lambda0*p.omega*mid^(p.omega-1)*G*exp(-p.lambda0*mid^p.omega*dt)
        marginal>1 ? (lo=mid) : (hi=mid)
    end
    (lo+hi)/2
end

struct Policy
    target::Vector{Int}
    pricing::SparseMatrixCSC{Float64,Int}
    entropy::Vector{Float64}
    k::Vector{Float64} # before any price change; forward pass gathers at target
    probability::Vector{Float64}
    P0::SparseMatrixCSC{Float64,Int}
    P1::SparseMatrixCSC{Float64,Int}
    clip0::Vector{Float64}
    clip1::Vector{Float64}
end

function bellman_step(next,g,p,n,Y,w,inflation,market,r; matrices=nothing, center=0.0)
    abs(r)<100 || throw(DomainError(r,"trial discount rate exceeds numerical search range"))
    P0,c0,P1,c1=if matrices===nothing
        a,b=transition(g,p,n,inflation,market)
        c,d=transition(g,p,n,inflation,market;reset=true)
        (a,b,c,d)
    else
        matrices
    end
    e0=P0*next; e1=P1*next
    disc=exp(-r*n.dt)
    m=length(next); stay=zeros(m); k=zeros(m); prob=zeros(m)
    for j in 1:m
        x,a,_,_=g.states[j]
        k[j]=step_rd(disc*(e1[j]-e0[j]),p,n.dt)
        prob[j]=-expm1(-p.lambda0*k[j]^p.omega*n.dt)
        actual_x=x+center
        profit=(exp(actual_x)-w/p.A)*exp(a-p.epsilon*actual_x)*Y
        stay[j]=n.dt*(profit-k[j])+disc*((1-prob[j])*e0[j]+prob[j]*e1[j])
    end
    value=copy(stay); target=collect(1:m)
    rows=Int[]; cols=Int[]; weights=Float64[]; entropy=zeros(m)
    # A tiny log-sum-exp regularization resolves finite-grid indifference masses.
    # Setting temperature=0 recovers the deterministic max exactly.
    tau=n.price_temperature
    for jv in 1:g.dims[4], jb in 1:g.dims[3], ja in 1:g.dims[2]
        offset=g.dims[1]*((ja-1)+g.dims[2]*((jb-1)+g.dims[3]*(jv-1)))
        block=offset+1:offset+g.dims[1]
        best=first(block)+argmax(@view stay[block])-1
        for j in block
            q=[stay[k]-p.K*(k!=j) for k in block]
            qmax=maximum(q)
            if tau==0
                dest=stay[best]-p.K>stay[j]+1e-12 ? best : j
                value[j]=stay[dest]-p.K*(dest!=j); target[j]=dest
                push!(rows,j); push!(cols,dest); push!(weights,1.0)
            else
                wts=exp.((q.-qmax)./tau); wts./=sum(wts)
                value[j]=qmax+tau*log(sum(exp.((q.-qmax)./tau)))
                target[j]=first(block)+argmax(q)-1
                for (k,wk) in zip(block,wts)
                    if wk>0
                        push!(rows,j); push!(cols,k); push!(weights,wk)
                        entropy[j]-=tau*wk*log(wk)
                    end
                end
            end
        end
    end
    S=sparse(rows,cols,weights,m,m)
    value,Policy(target,S,entropy,k,prob,P0,P1,c0,c1)
end

function initial_distribution(g,ini,logP; center=0.0)
    mu=zeros(length(g.states))
    for j in 1:3
        inds=Int[]; ws=Float64[]
        deposit!(inds,ws,g,(ini.log_nominal_price-logP-center,
            ini.mean_a+ini.sd_a*NODES[j],ini.mean_b,ini.variance),WEIGHTS[j])
        for (i,w) in zip(inds,ws); mu[i]+=w; end
    end
    mu
end

"""Move mass through paid price resets and the same Markov kernel as the Bellman step."""
function forward_step(mu,policy,n)
    post=transpose(policy.pricing)*mu
    count=dot(mu,1 .-diag(policy.pricing))
    prob=policy.probability
    nxt=transpose(policy.P0)*(post.*(1 .-prob))+transpose(policy.P1)*(post.*prob)
    clip=dot(post,(1 .-prob).*policy.clip0.+prob.*policy.clip1)
    (next=nxt,post=post,frequency=count/n.dt,
     research=dot(post,policy.k),innovation_probability=dot(post,prob),
     clip=clip)
end

# A frozen-aggregate, single-firm continuation approximation; no stationary GE assumed.
function terminal_value(g,p,n,Y,w,market)
    n.terminal_value=="zero" && return zeros(length(g.states)),0.0,true
    P0,c0=transition(g,p,n,p.pi_target,market)
    P1,c1=transition(g,p,n,p.pi_target,market;reset=true)
    center=log(p.epsilon/(p.epsilon-1)*w/p.A)
    V=[max((exp(x+center)-w/p.A)*exp(a-p.epsilon*(x+center))*Y,0.0)/p.rho
       for (x,a,b,v) in g.states]
    err=Inf
    matrices=(P0,c0,P1,c1)
    for _ in 1:n.terminal_iterations
        _,policy=bellman_step(V,g,p,n,Y,w,p.pi_target,market,p.rho;matrices=matrices,center=center)
        mix=spdiagm(0=>1 .-policy.probability)*P0+spdiagm(0=>policy.probability)*P1
        kernel=policy.pricing*mix
        flow=[n.dt*((exp(x+center)-w/p.A)*exp(a-p.epsilon*(x+center))*Y-policy.k[j])
              for (j,(x,a,b,v)) in enumerate(g.states)]
        reward=policy.pricing*flow.-p.K.*(1 .-diag(policy.pricing)).+policy.entropy
        V=(spdiagm(0=>ones(length(V)))-exp(-p.rho*n.dt)*kernel)\Vector(reward)
        check,_=bellman_step(V,g,p,n,Y,w,p.pi_target,market,p.rho;matrices=matrices,center=center)
        err=maximum(abs.(check.-V))
        err<n.terminal_tolerance && return V,err,true
    end
    V,err,false
end

# Unknowns include endpoint C,Y,w,M; pi has one entry per transition.
# Endpoint allocation clears markets with a flat-consumption tail condition.
function unpack(z,n)
    T=n.steps; j=0
    C=exp.(z[j+1:j+T+1]); j+=T+1
    Y=exp.(z[j+1:j+T+1]); j+=T+1
    w=exp.(z[j+1:j+T+1]); j+=T+1
    pi=z[j+1:j+T]; j+=T
    M=z[j+1:j+T+1]; j+=T+1
    (C=C,Y=Y,w=w,pi=pi,M=M,logP0=z[j+1])
end
function baseline_guess(p,n,ini)
    C=fill(n.terminal_consumption,n.steps+1)
    w=fill(p.A*(p.epsilon-1)/p.epsilon,n.steps+1)
    Y=fill(n.terminal_consumption*1.01,n.steps+1)
    M=fill(ini.mean_b,n.steps+1)
    # CES identity for the three-point initial taste quadrature.
    avg=sum(WEIGHTS[j]*exp(ini.mean_a+ini.sd_a*NODES[j]) for j in 1:3)
    logP=ini.log_nominal_price+log(avg)/(1-p.epsilon)
    vcat(log.(C),log.(Y),log.(w),fill(p.pi_target,n.steps),M,[logP])
end
function firm_path(paths,g,p,n)
    T=n.steps
    value,terr,tok=terminal_value(g,p,n,paths.Y[end],paths.w[end],paths.M[end])
    policies=Vector{Policy}(undef,T+1)
    center=log(p.epsilon/(p.epsilon-1)*paths.w[end]/p.A)
    _,policies[end]=bellman_step(value,g,p,n,paths.Y[end],paths.w[end],
        p.pi_target,paths.M[end],p.rho;center=center)
    for t in T:-1:1
        growth=log(paths.C[t+1]/paths.C[t])/n.dt
        r=p.rho+p.gamma*growth
        center=log(p.epsilon/(p.epsilon-1)*paths.w[t]/p.A)
        nextcenter=log(p.epsilon/(p.epsilon-1)*paths.w[t+1]/p.A)
        effective_inflation=paths.pi[t]+(nextcenter-center)/n.dt
        value,policies[t]=bellman_step(value,g,p,n,paths.Y[t],paths.w[t],
                                     effective_inflation,paths.M[t+1],r;center=center)
    end
    policies,(residual=terr,converged=tok)
end
function equilibrium_residual(z,g,p,n,ini; details=false)
    paths=unpack(z,n); T=n.steps
    policies,terminal=firm_path(paths,g,p,n)
    centers=log.(p.epsilon/(p.epsilon-1).*paths.w./p.A)
    mu=initial_distribution(g,ini,paths.logP0;center=centers[1])
    index=[exp(a+(1-p.epsilon)*x) for (x,a,b,v) in g.states]
    labor=[exp(a-p.epsilon*x) for (x,a,b,v) in g.states]
    beliefs=[b for (x,a,b,v) in g.states]
    price=zeros(T+1); resource=zeros(T+1); wage=zeros(T+1); euler=zeros(T)
    quality=zeros(T+1); moments=NamedTuple[]
    initial_clip=ini.log_nominal_price-paths.logP0-centers[1]<g.axes[1][1] ||
                 ini.log_nominal_price-paths.logP0-centers[1]>g.axes[1][end]
    mass_error=abs(sum(mu)-1)
    for t in 1:T+1
        quality[t]=paths.M[t]-dot(mu,beliefs)
        f=forward_step(mu,policies[t],n)
        price[t]=log(dot(f.post,index))+(1-p.epsilon)*centers[t]
        N=paths.Y[t]*exp(-p.epsilon*centers[t])*dot(f.post,labor)/p.A
        costs=p.K*f.frequency+f.research
        resource[t]=(paths.Y[t]-paths.C[t]-costs)/paths.C[t]
        wage[t]=log(paths.w[t])-(log(p.chi)+p.varphi*log(N)+p.gamma*log(paths.C[t]))
        if t<=T
            growth=log(paths.C[t+1]/paths.C[t])/n.dt
            euler[t]=p.gamma*growth-(p.phi_pi-1)*(paths.pi[t]-p.pi_target)
        end
        div=paths.Y[t]-paths.w[t]*N-costs
        push!(moments,(time=(t-1)*n.dt,C=paths.C[t],Y=paths.Y[t],w=paths.w[t],
            N=N,pi=(t<=T ? paths.pi[t] : p.pi_target),M=paths.M[t],price_frequency=f.frequency,
            research=f.research,innovation_probability=f.innovation_probability,
            clipping=f.clip,budget_error=paths.w[t]*N+div-paths.C[t],
            mean_variance=dot(f.post,[s[4] for s in g.states])))
        if t<=T
            mu=f.next
            mass_error=max(mass_error,abs(sum(mu)-1))
        end
    end
    # Flat-consumption tail, with endpoint consumption determined by market clearing.
    tail=log(paths.C[end]/paths.C[end-1])
    residual=vcat(price,resource,wage,euler,quality,[tail])
    details ? (residual=residual,paths=paths,moments=moments,distribution=mu,
        terminal=terminal,mass_error=mass_error,initial_clipping=initial_clip) : residual
end

function numerical_jacobian(f,z,r)
    J=zeros(length(r),length(z))
    for j in eachindex(z)
        h=1e-5*max(1,abs(z[j])); zp=copy(z); zp[j]+=h
        J[:,j]=(f(zp).-r)./h
    end
    J
end

"""Damped least-squares/Broyden shooting. Returns convergence honestly; never assumes it."""
function solve_equilibrium(p,n,ini; verbose=true,guess=nothing)
    validate(p,n,ini)
    g=make_grid(p,n,ini)
    z=guess===nothing ? baseline_guess(p,n,ini) : copy(guess)
    f=z->equilibrium_residual(z,g,p,n,ini)
    r=f(z); J=nothing; history=Float64[]; converged=false
    damping=1e-3
    for iteration in 1:n.max_iterations
        err=maximum(abs.(r)); push!(history,err)
        verbose && @printf("iteration %3d | equilibrium residual %.4e\n",iteration,err)
        if err<n.tolerance; converged=true; break; end
        if J===nothing || (iteration-1)%n.jacobian_refresh==0
            J=numerical_jacobian(f,z,r)
        end
        accepted=false
        for _ in 1:8
            step=-(transpose(J)*J+damping*I)\(transpose(J)*r)
            # A bounded trust step keeps discounting and exponential levels sensible.
            step*=min(1.0,0.25/max(norm(step,Inf),eps()))
            for alpha in (1.0,0.5,0.25,0.125,0.0625)
                candidate=z.+alpha.*step
                trial=try
                    f(candidate)
                catch e
                    e isa DomainError ? fill(Inf,length(r)) : rethrow()
                end
                if all(isfinite,trial) && sum(abs2,trial)<sum(abs2,r)
                    delta=candidate-z; diff=trial-r
                    J .+= ((diff-J*delta)*transpose(delta))/(dot(delta,delta)+eps())
                    z=candidate; r=trial; accepted=true
                    damping=max(damping/3,1e-9)
                    break
                end
            end
            accepted && break
            damping*=10
        end
        if !accepted
            J=nothing
            damping=min(damping,1e3)
        end
    end
    result=equilibrium_residual(z,g,p,n,ini;details=true)
    converged=maximum(abs.(result.residual))<n.tolerance
    merge(result,(converged=converged,history=history,unknowns=z,grid=g))
end

function write_results(result,p,n; directory=n.output)
    mkpath(directory)
    open(joinpath(directory,"aggregates.csv"),"w") do io
        rows=result.moments
        println(io,join(string.(keys(first(rows))),","))
        for row in rows; println(io,join(values(row),",")); end
    end
    summary=Dict("converged"=>result.converged,
        "max_equilibrium_residual"=>maximum(abs.(result.residual)),
        "mass_error"=>result.mass_error,
        "terminal_value_converged"=>result.terminal.converged,
        "terminal_value_residual"=>result.terminal.residual,
        "max_transition_clipping"=>maximum(m.clipping for m in result.moments),
        "initial_price_clipping"=>result.initial_clipping,
        "terminal_allocation"=>"endogenous market clearing with zero final consumption growth",
        "steps"=>n.steps,"dt"=>n.dt,"state_count"=>length(result.grid.states),
        "terminal_closure"=>n.terminal_value,
        "price_temperature"=>n.price_temperature,
        "one_step_smoothing_bound"=>n.price_temperature*log(n.nx),
        "scope"=>"finite-horizon approximation; requires domain, grid, time and horizon convergence")
    open(joinpath(directory,"diagnostics.toml"),"w") do io; TOML.print(io,summary); end
    open(joinpath(directory,"checkpoint.toml"),"w") do io
        TOML.print(io,Dict("unknowns"=>result.unknowns,"residuals"=>result.residual,
                          "residual_history"=>result.history))
    end
    summary
end
include("BalancedGrowth.jl")
include("LargeBGP.jl")
end # module
