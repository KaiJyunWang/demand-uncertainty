# Balanced growth: a and b denote deviations from mean log quality M.
# x grid denotes detrended relative price, further centered on flexible price.
export solve_balanced_growth, bgp_residual, write_bgp_results, stationary_firm,
       invariant_distribution, bgp_growth_rates

bgp_growth_rates(G,p)=(quality=G,consumption=G/(p.epsilon-1),
    quality_adjusted_inflation=p.pi_target,
    nominal_price=p.pi_target+G/(p.epsilon-1),
    interest=p.rho+p.gamma*G/(p.epsilon-1)+p.pi_target)

function stationary_firm(g,p,n,Y,w,G)
    p.replacement_entry && error("Use the tensor backend for exit and replacement entry")
    growth=G/(p.epsilon-1)
    discount=p.rho+(p.gamma-1)*growth
    discount>0 || throw(DomainError(discount,"nonpositive detrended discount rate"))
    drift=p.pi_target+growth
    P0,c0=transition(g,p,n,drift,0.;quality_growth=G)
    P1,c1=transition(g,p,n,drift,0.;reset=true,quality_growth=G)
    center=log(p.epsilon/(p.epsilon-1)*w/p.A)
    V=[max((exp(y+center)-w/p.A)*exp(z-p.epsilon*(y+center))*Y,0.)/discount
       for (y,z,b,v) in g.states]
    matrices=(P0,c0,P1,c1); err=Inf
    policy=nothing
    for iteration in 1:n.terminal_iterations
        _,policy=bellman_step(V,g,p,n,Y,w,drift,0.,discount;
                             matrices=matrices,center=center,cashflow_constraint=true)
        mix=spdiagm(0=>1 .-policy.probability)*P0+spdiagm(0=>policy.probability)*P1
        kernel=policy.pricing*mix
        flow=[n.dt*((exp(y+center)-w/p.A)*exp(z-p.epsilon*(y+center))*Y-rd_expense(policy.k[j],p))
              for (j,(y,z,b,v)) in enumerate(g.states)]
        reward=policy.pricing*flow.-p.K.*(1 .-diag(policy.pricing)).+policy.entropy
        V=(spdiagm(0=>ones(length(V)))-exp(-discount*n.dt)*kernel)\Vector(reward)
        check,newpolicy=bellman_step(V,g,p,n,Y,w,drift,0.,discount;
                                     matrices=matrices,center=center,cashflow_constraint=true)
        err=maximum(abs.(check.-V))
        if err<n.terminal_tolerance
            # Use the policy evaluated by the linear system. The residual
            # measures the difference between this policy and reoptimization.
            return (value=V,policy=policy,kernel=kernel,error=err,iterations=iteration)
        end
    end
    throw(ErrorException("stationary firm policy iteration failed: Bellman residual $err"))
end

function invariant_distribution(kernel; tolerance=1e-9)
    m=size(kernel,1)
    # Normalize at the linear-equation level; do not renormalize quality states.
    A=spdiagm(0=>ones(m))-sparse(transpose(kernel))
    A[m,:]=ones(m)
    rhs=zeros(m); rhs[m]=1
    mu=A\rhs
    minimum(mu)>=-tolerance || throw(DomainError(minimum(mu),"negative invariant mass"))
    mu=max.(mu,0.0); mu./=sum(mu)
    residual=norm(transpose(kernel)*mu-mu,1)
    residual<tolerance || throw(ErrorException("invariant distribution failed: $residual"))
    mu,residual
end

"""Unknowns are log detrended C,Y,w, and quality growth G (level, per year)."""
function bgp_residual(z,g,p,n; details=false)
    C,Y,w=exp.(z[1:3]); G=z[4]
    firm=stationary_firm(g,p,n,Y,w,G)
    mu,distribution_error=invariant_distribution(firm.kernel)
    f=forward_step(mu,firm.policy,n)
    center=log(p.epsilon/(p.epsilon-1)*w/p.A)
    price=dot(f.post,[exp(a+(1-p.epsilon)*(y+center)) for (y,a,b,v) in g.states])
    dispersion=dot(f.post,[exp(a-p.epsilon*(y+center)) for (y,a,b,v) in g.states])
    N=Y*dispersion/p.A
    mean_b=dot(mu,[s[3] for s in g.states])
    variable_research=f.research
    active_mass=dot(f.post,Float64.(firm.policy.k.>0))
    research=variable_research+p.F_R*active_mass
    resource=(Y-C-p.K*f.frequency-research)/C
    labor=log(w)-log(p.chi)-p.varphi*log(N)-p.gamma*log(C)
    residual=[log(price),resource,labor,mean_b]
    lambda=[p.lambda0*k^p.omega for k in firm.policy.k]
    flux=dot(f.post,lambda.*[p.premium-s[3] for s in g.states])
    # The exact discrete split mean equation must also vanish off boundaries.
    split_flux=dot(f.post,firm.policy.probability.*[p.premium-s[3] for s in g.states])/n.dt
    split_drift=G*dot(f.post,1 .-firm.policy.probability)
    budget=Y-w*N-p.K*f.frequency-research+w*N-C
    moments=(C=C,Y=Y,w=w,N=N,G=G,growth=G/(p.epsilon-1),
        price_frequency=f.frequency,research=research,variable_research=variable_research,
        rd_active_mass=active_mass,fixed_research=p.F_R*active_mass,
        innovation_frequency=dot(f.post,lambda),clipping=f.clip,
        mean_b=mean_b,mean_a=dot(mu,[s[2] for s in g.states]),
        mean_variance=dot(mu,[s[4] for s in g.states]),
        quality_flux=flux,continuous_growth_error=G-flux,
        split_growth_error=split_drift-split_flux,budget_error=budget)
    profits=[operating_profit(x+center,a,Y,w,p) for (x,a,b,v) in g.states]
    gain=exp(-(p.rho+(p.gamma-1)*G/(p.epsilon-1))*n.dt).*((firm.policy.P1-firm.policy.P0)*firm.value)
    moments=merge(moments,rd_financing_statistics(firm.policy.k,profits,gain,f.post,p,n.dt))
    details ? (residual=residual,firm=firm,distribution=mu,post_distribution=f.post,
               distribution_error=distribution_error,moments=moments,
               rates=bgp_growth_rates(G,p)) : residual
end

function solve_balanced_growth(p,n,ini=Initial();guess=nothing,verbose=true,warmstart=nothing)
    validate(p,n,ini)
    p.gamma==1 || throw(ArgumentError("with separable, constant-chi preferences, constant-hours balanced growth requires gamma=1; use calibration_bgp.toml"))
    if p.replacement_entry || n.nx*n.na*n.nb*n.nv >= 100_000
        return solve_large_bgp(p,n,ini;guess=guess,verbose=verbose,warmstart=warmstart)
    end
    g=make_grid(p,n,ini)
    z=guess===nothing ? [log(1/3),log(0.34),log(p.A*(p.epsilon-1)/p.epsilon),0.01] : copy(guess)
    length(z)==4 || throw(ArgumentError("BGP guess must have four entries"))
    f=z->bgp_residual(z,g,p,n)
    r=f(z); history=Float64[]
    for iteration in 1:n.max_iterations
        err=norm(r,Inf); push!(history,err)
        verbose && @printf("BGP iteration %3d | residual %.4e | quality growth %.6f\n",iteration,err,z[4])
        err<n.tolerance && break
        J=numerical_jacobian(f,z,r)
        accepted=false
        for damping in (0.,1e-7,1e-5,1e-3,0.1,1.,100.)
            step=-(J'*J+damping*I)\(J'*r)
            step*=min(1.,0.25/max(norm(step,Inf),eps()))
            for alpha in (1.,0.5,0.25,0.1,0.025,0.005)
                trial=z+alpha*step
                # Search for the positive-growth equilibrium requested here.
                trial[4]>0 || continue
                rt=try
                    f(trial)
                catch e
                    if e isa DomainError || e isa SingularException
                        fill(Inf,4)
                    else
                        rethrow()
                    end
                end
                if all(isfinite,rt) && norm(rt)<norm(r)
                    z=trial; r=rt; accepted=true; break
                end
            end
            accepted && break
        end
        if !accepted
            verbose && println("No improving BGP step; returning unconverged candidate.")
            break
        end
    end
    result=bgp_residual(z,g,p,n;details=true)
    merge(result,(converged=norm(result.residual,Inf)<n.tolerance,
                  unknowns=z,history=history,grid=g))
end

function write_bgp_results(result,p,n;directory=n.output)
    hasproperty(result,:large) && return write_large_bgp_results(result,p,n;directory=directory)
    mkpath(directory)
    data=Dict{String,Any}(string(k)=>v for (k,v) in pairs(result.moments))
    data["rd_cashflow_constraint"]=true
    data["F_R"]=p.F_R
    data["lambda0"]=p.lambda0
    data["converged"]=result.converged
    data["max_equilibrium_residual"]=norm(result.residual,Inf)
    data["bellman_residual"]=result.firm.error
    data["distribution_residual"]=result.distribution_error
    data["mass_error"]=abs(sum(result.distribution)-1)
    data["dt"]=n.dt
    data["state_count"]=length(result.grid.states)
    data["price_temperature"]=n.price_temperature
    data["scope"]="stationary detrended balanced-growth equilibrium on a finite grid"
    for (key,value) in pairs(result.rates); data[string(key)]=value; end
    data["menu_expenditure"]=p.K*result.moments.price_frequency
    enhance_report!(data)
    open(joinpath(directory,"balanced_growth.toml"),"w") do io; TOML.print(io,data); end
    write_solution_csv(data,joinpath(directory,"solution.csv"))
    open(joinpath(directory,"checkpoint.toml"),"w") do io
        TOML.print(io,Dict("unknowns"=>result.unknowns,"residuals"=>result.residual,"history"=>result.history))
    end
    open(joinpath(directory,"distribution.csv"),"w") do io
        println(io,"price_gap,taste_gap,belief_gap,variance,mass,post_price_mass,rd,price_adjustment_probability")
        for (j,s) in enumerate(result.grid.states)
            println(io,join((s...,result.distribution[j],result.post_distribution[j],
                result.firm.policy.k[j],1-result.firm.policy.pricing[j,j]),","))
        end
    end
    data
end
