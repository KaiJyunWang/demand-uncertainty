# A valid zero-growth equilibrium may coexist with other equilibria when research
# has an operating fee. Select the closed class beta=v=0 only after verifying
# that optimal R&D is zero at EVERY price/taste node in that class. Firm values
# and innovation deviations are still solved on the full four-dimensional grid.
function zero_research_invariant!(ws,n)
    qs=findall(q->q[2]==0. && q[3]==0.,ws.g.qstates)
    length(qs)==ws.g.dims[2] || return nothing
    any(>(0.),view(ws.policy.k,:,qs)) && return nothing
    any(>(0.),view(ws.policy.exit,:,qs)) && return nothing
    nx=ws.g.dims[1]; rows=Int[];cols=Int[];vals=Float64[]
    for (a,q) in enumerate(qs), j in 1:nx, i in 1:nx
        prob=(i==j ? ws.policy.keep[j,q] : 0.)+ws.policy.reset[i,q]*ws.policy.mix[j,q]
        prob==0 && continue
        push!(rows,(a-1)*nx+i);push!(cols,(a-1)*nx+j);push!(vals,prob)
    end
    S=sparse(rows,cols,vals,nx*length(qs),nx*length(qs))
    T=kron(ws.K0.outgoing[qs,qs],sparse(transpose(ws.drift.incoming)))*S
    maximum(abs,vec(sum(T;dims=1)).-1)<1e-12 || error("Zero-research class is not closed")
    mass,err=invariant_distribution(sparse(transpose(T)))
    fill!(ws.mu,0.);ws.mu[:,qs].=reshape(mass,nx,:)
    tensor_forward!(ws.work[5],ws.mu,ws)
    full_error=sum(abs,ws.work[5].-ws.mu)
    full_error<1e-9 || error("Full-grid zero-research stationarity failed: $full_error")
    full_error
end

function zero_research_residual!(ws,z,p,n;verbose=false)
    C,Y,w=exp.(z[1:3])
    ferr,fit=tensor_firm!(ws,p,n,Y,w,0.;verbose=verbose)
    derr=zero_research_invariant!(ws,n)
    derr===nothing && return nothing
    residual,moments=tensor_moments(ws,p,n,C,Y,w,0.)
    (residual=residual,moments=moments,firm_error=ferr,firm_iterations=fit,
     distribution_error=derr,negative_mass=0.,linear=(converged=true,))
end

function solve_zero_research_bgp(p,n,ini;guess=nothing,verbose=true,warmstart=nothing,workspace_sink=nothing)
    z=guess===nothing ? [log(.333),log(.334),log(.86),0.] : [guess[1:3];0.]
    ws=tensor_workspace(p,n,ini,0.)
    workspace_sink!==nothing && (workspace_sink[]=ws)
    warmstart!==nothing && (ws.V .= warmstart.V)
    result=zero_research_residual!(ws,z,p,n;verbose=verbose)
    result===nothing && return nothing
    history=Float64[]
    for iteration in 1:n.max_iterations
        err=norm(result.residual,Inf);push!(history,err)
        println("ZERO-RESEARCH BGP iteration ",iteration," residual ",err);flush(stdout)
        err<n.tolerance && break
        J=zeros(3,3)
        J[:,1]=[0.,-(1+result.residual[2]),-p.gamma]
        trial=copy(z);h=1e-5;trial[2]+=h
        rt=zero_research_residual!(ws,trial,p,n;verbose=verbose)
        rt===nothing && return nothing
        J[:,2]=(rt.residual[1:3]-result.residual[1:3])/h
        J[:,3]=(1-p.epsilon).*J[:,2]
        J[1,3]+=1-p.epsilon
        J[2,3]-=(1-p.epsilon)*exp(z[2]-z[1])
        J[3,3]+=1+p.varphi
        step=-(J\result.residual[1:3]);step*=min(1.,.2/max(norm(step,Inf),eps()))
        accepted=false
        for alpha in (1.,.5,.1,.025)
            trial=[z[1:3]+alpha*step;0.]
            rt=zero_research_residual!(ws,trial,p,n;verbose=verbose)
            if rt!==nothing && norm(rt.residual)<norm(result.residual)
                z=trial;result=rt;accepted=true;break
            end
        end
        accepted || return nothing
    end
    result=zero_research_residual!(ws,z,p,n;verbose=verbose)
    result===nothing && return nothing
    merge(result,(large=true,converged=norm(result.residual,Inf)<n.tolerance,
        unknowns=z,history=history,workspace=ws,rates=bgp_growth_rates(0.,p),
        equilibrium_selection=p.replacement_entry ? "zero growth; known common quality; research and exit inactive on invariant support" : "zero growth; known common quality; optimal research inactive on invariant support"))
end
