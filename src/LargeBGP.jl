# Tensor-factorized equivalent of the direct discrete BGP model.
# Full 51^4 matrices are never assembled. Price drift and (a,b,v) diffusion
# commute, and price-choice softmax is diagonal plus a rank-one block.
using Base.Threads

struct TensorGrid
    axes::NTuple{4,Vector{Float64}}
    dims::NTuple{4,Int}
    qstates::Vector{NTuple{3,Float64}}
end
function tensor_grid(p,n,ini=Initial())
    ax=(price_axis(n),
        taste_axis(n),
        belief_axis(n),
        collect(range(0,p.sigma_Q^2,length=n.nv)))
    dims=length.(ax)
    qstates=vec([(ax[2][I[1]],ax[3][I[2]],ax[4][I[3]])
                 for I in CartesianIndices(dims[2:4])])
    TensorGrid(ax,dims,qstates)
end

struct TensorKernel
    outgoing::SparseMatrixCSC{Float64,Int} # dest quality state x source
    incoming::SparseMatrixCSC{Float64,Int}
    clipping::Vector{Float64}
end
function quality_kernel(g,p,n,G;reset=false)
    rows=Int[]; cols=Int[]; vals=Float64[]
    nq=length(g.qstates)
    sizehint!(rows,50nq); sizehint!(cols,50nq); sizehint!(vals,50nq)
    clipping=zeros(nq)
    linear=LinearIndices(g.dims[2:4])
    for (q,(a,b,v)) in enumerate(g.qstates)
        mom=filter_moments(a,b,v,n.dt,p.sigma_a)
        la=sqrt(mom.var_a); lb=mom.cov_ab/la
        lc=sqrt(max(mom.var_b-lb^2,0.))
        for i in 1:3,j in 1:3
            an=mom.mean_a-G*n.dt+la*NODES[i]
            bn=reset ? p.premium : b-G*n.dt+lb*NODES[i]+lc*NODES[j]
            vn=reset ? p.sigma_Q^2 : mom.variance
            prob=WEIGHTS[i]*WEIGHTS[j]
            if an<g.axes[2][1] || an>g.axes[2][end] || bn<g.axes[3][1] || bn>g.axes[3][end]
                clipping[q]+=prob
            end
            brackets=(bracket(g.axes[2],an),bracket(g.axes[3],bn),bracket(g.axes[4],vn))
            for mask in 0:7
                ind=ntuple(d->brackets[d][1]+((mask>>(d-1))&1),3)
                w=prob
                for d in 1:3
                    w*=((mask>>(d-1))&1)==1 ? brackets[d][2] : 1-brackets[d][2]
                end
                if w>0
                    push!(rows,linear[ind...]);push!(cols,q);push!(vals,w)
                end
            end
        end
    end
    outgoing=sparse(rows,cols,vals,nq,nq)
    TensorKernel(outgoing,sparse(transpose(outgoing)),clipping)
end

struct PriceDrift
    left::Vector{Int}
    right_weight::Vector{Float64}
    incoming::SparseMatrixCSC{Float64,Int} # rows source, columns destination
    outside::BitVector
end
function price_drift(g,p,n,G)
    nx=g.dims[1]; drift=(p.pi_target+G/(p.epsilon-1))*n.dt
    left=zeros(Int,nx);weights=zeros(nx);outside=falses(nx)
    rows=Int[];cols=Int[];vals=Float64[]
    for i in 1:nx
        dest=g.axes[1][i]-drift
        left[i],weights[i]=bracket(g.axes[1],dest)
        outside[i]=dest<g.axes[1][1] || dest>g.axes[1][end]
        push!(rows,i);push!(cols,left[i]);push!(vals,1-weights[i])
        push!(rows,i);push!(cols,left[i]+1);push!(vals,weights[i])
    end
    PriceDrift(left,weights,sparse(rows,cols,vals,nx,nx),outside)
end

# Each threaded column owns its entire output, avoiding concurrent scatter writes.
function quality_apply!(dest,source,kernel::SparseMatrixCSC)
    nx,nq=size(source)
    @threads :static for q in 1:nq
        @inbounds for x in 1:nx; dest[x,q]=0.; end
        @inbounds for k in nzrange(kernel,q)
            j=kernel.rowval[k];w=kernel.nzval[k]
            @simd for x in 1:nx
                dest[x,q]+=w*source[x,j]
            end
        end
    end
    dest
end
function price_value!(dest,source,drift)
    nx,nq=size(source)
    @threads :static for q in 1:nq
        @inbounds @simd for x in 1:nx
            j=drift.left[x];w=drift.right_weight[x]
            dest[x,q]=(1-w)*source[j,q]+w*source[j+1,q]
        end
    end
    dest
end
function price_forward!(dest,source,drift)
    nx,nq=size(source);K=drift.incoming
    @threads :static for q in 1:nq
        @inbounds for x in 1:nx
            total=0.
            for k in nzrange(K,x)
                total+=K.nzval[k]*source[K.rowval[k],q]
            end
            dest[x,q]=total
        end
    end
    dest
end

mutable struct TensorPolicy
    keep::Matrix{Float64} # diagonal term of pricing kernel
    reset::Matrix{Float64} # reset weights, normalized in each quality block
    mix::Matrix{Float64} # rank-one coefficient (keep+mix=1)
    entropy::Matrix{Float64}
    k::Matrix{Float64}
    probability::Matrix{Float64}
    profit::Matrix{Float64}
    exit::Matrix{Float64}
end
function TensorPolicy(nx,nq)
    TensorPolicy((zeros(nx,nq) for _ in 1:8)...)
end
mutable struct TensorWorkspace
    g::TensorGrid
    K0::TensorKernel
    K1::TensorKernel
    drift::PriceDrift
    policy::TensorPolicy
    work::Vector{Matrix{Float64}}
    V::Matrix{Float64}
    mu::Matrix{Float64}
    entry::Matrix{Float64}
end
function tensor_workspace(p,n,ini,G)
    g=tensor_grid(p,n,ini); nx=g.dims[1];nq=length(g.qstates)
    K0=quality_kernel(g,p,n,G);K1=quality_kernel(g,p,n,G;reset=true)
    println("Tensor kernels: ",g.dims," states=",nx*nq," nnz=",nnz(K0.outgoing),"/",nnz(K1.outgoing)," threads=",nthreads())
    ws=TensorWorkspace(g,K0,K1,price_drift(g,p,n,G),TensorPolicy(nx,nq),
        [zeros(nx,nq) for _ in 1:5],zeros(nx,nq),zeros(nx,nq),zeros(nx,nq))
    # A broad, strictly positive initial mass helps discover slow-moving classes.
    for q in 1:nq,x in 1:nx
        a,b,v=g.qstates[q]
        ws.mu[x,q]=exp(-g.axes[1][x]^2/.02-(a+G)^2/.2-b^2/.12-20v)
    end
    ws.mu./=sum(ws.mu)
    ws
end
function update_kernels!(ws,p,n,G)
    ws.K0=quality_kernel(ws.g,p,n,G)
    ws.K1=quality_kernel(ws.g,p,n,G;reset=true)
    ws.drift=price_drift(ws.g,p,n,G)
end
function continuation!(E0,E1,V,ws)
    W=ws.work[1]
    price_value!(W,V,ws.drift)
    quality_apply!(E0,W,ws.K0.outgoing)
    quality_apply!(E1,W,ws.K1.outgoing)
end

# Accurate, fast scalar optimization in log k; bracket safeguarded Newton.
function tensor_rd(gain,p,dt;cap=Inf)
    (gain<=0 || p.lambda0==0 || cap<=p.F_R) && return 0.
    logupper=log(p.lambda0*p.omega*gain)/(1-p.omega)
    hi=logupper;lo=logupper-100.;x=hi
    for _ in 1:12
        rate=p.lambda0*exp(p.omega*x)*dt
        f=(p.omega-1)*(x-logupper)-rate
        abs(f)<1e-12 && break
        f>0 ? (lo=x) : (hi=x)
        next=x-f/(p.omega-1-p.omega*rate)
        x=lo<next<hi ? next : (lo+hi)/2
    end
    k=min(exp(x),cap-p.F_R)
    step_rd_surplus(k,gain,p,dt)>0 ? k : 0.
end

function tensor_bellman!(dest,V,ws,p,n,Y,w,G; improve=true)
    E0=ws.work[2];E1=ws.work[3]
    continuation!(E0,E1,V,ws)
    pol=ws.policy;nx,nq=size(V)
    disc=exp(-(p.rho+(p.gamma-1)*G/(p.epsilon-1))*n.dt)
    center=log(p.epsilon/(p.epsilon-1)*w/p.A)
    tau=n.price_temperature
    @threads :static for q in 1:nq
        a,b,v=ws.g.qstates[q]
        @inbounds for x in 1:nx
            if improve
                actual=ws.g.axes[1][x]+center
                pol.profit[x,q]=operating_profit(actual,a,Y,w,p)
                k=tensor_rd(disc*(E1[x,q]-E0[x,q]),p,n.dt;cap=max(pol.profit[x,q],0.))
                pol.k[x,q]=k
                pol.probability[x,q]=-expm1(-p.lambda0*k^p.omega*n.dt)
            end
            prob=pol.probability[x,q]
            dest[x,q]=n.dt*(pol.profit[x,q]-rd_expense(pol.k[x,q],p))+disc*((1-prob)*E0[x,q]+prob*E1[x,q])
        end
        if improve
            m=-Inf;best=1
            @inbounds for x in 1:nx
                if dest[x,q]>m;m=dest[x,q];best=x;end
            end
            if tau==0
                @inbounds for x in 1:nx
                    pol.reset[x,q]=(x==best)
                    change=m-p.K>dest[x,q]+1e-12
                    pol.mix[x,q]=change
                    pol.keep[x,q]=!change
                    pol.entropy[x,q]=0.
                end
            else
                total=0.;meanstay=0.
                @inbounds for x in 1:nx
                    rx=exp((dest[x,q]-m)/tau)
                    pol.reset[x,q]=rx;total+=rx;meanstay+=rx*dest[x,q]
                end
                meanstay/=total
                @inbounds for x in 1:nx
                    stay=dest[x,q];scale=max(stay,m-p.K)
                    a0=exp((stay-scale)/tau)
                    remove=exp((stay-p.K-scale)/tau)
                    pool=exp((m-p.K-scale)/tau)*total
                    den=a0+pool-remove
                    d=(a0-remove)/den;t=pool/den
                    pol.keep[x,q]=d;pol.mix[x,q]=t
                    pol.reset[x,q]/=total
                    value=scale+tau*log(den)
                    expected=d*stay+t*meanstay
                    paid=t*(1-pol.reset[x,q])
                    pol.entropy[x,q]=value-expected+p.K*paid
                end
            end
        end
        pool=0.
        @inbounds for x in 1:nx;pool+=pol.reset[x,q]*dest[x,q];end
        @inbounds for x in 1:nx
            paid=pol.mix[x,q]*(1-pol.reset[x,q])
            dest[x,q]=pol.keep[x,q]*dest[x,q]+pol.mix[x,q]*pool-p.K*paid+pol.entropy[x,q]
            if improve
                pol.exit[x,q]=p.replacement_entry && dest[x,q]<=0 ? 1. : 0.
            end
            dest[x,q]*=1-pol.exit[x,q]
        end
    end
    p.replacement_entry && improve && update_entry!(ws,p)
    dest
end

function tensor_price_forward!(dest,source,pol)
    nx,nq=size(source)
    @threads :static for q in 1:nq
        pool=0.
        @inbounds for x in 1:nx;pool+=source[x,q]*pol.mix[x,q]*(1-pol.exit[x,q]);end
        @inbounds @simd for x in 1:nx
            dest[x,q]=pol.keep[x,q]*source[x,q]*(1-pol.exit[x,q])+pol.reset[x,q]*pool
        end
    end
    dest
end
function tensor_forward!(dest,source,ws)
    post=ws.work[1];weighted=ws.work[2];temp=ws.work[3];combined=ws.work[4]
    tensor_operating_forward!(post,source,ws)
    @. weighted=post*(1-ws.policy.probability)
    quality_apply!(combined,weighted,ws.K0.incoming)
    @. weighted=post*ws.policy.probability
    quality_apply!(temp,weighted,ws.K1.incoming)
    @. combined=combined+temp
    price_forward!(dest,combined,ws.drift)
end

# Matrix-free BiCGSTAB. Arrays are reused; the operation may have its own work buffers.
function bicgstab!(x,apply!,rhs;tol=1e-11,maxiter=1500,label="linear",verbose=false)
    r=similar(x);rr=similar(x);v=zero(x);pv=zero(x);s=similar(x);t=similar(x)
    apply!(v,x);@. r=rhs-v;copyto!(rr,r)
    rho_old=1.;alpha=1.;omega=1.
    err=norm(r); err<tol && return (iterations=0,residual=err,converged=true)
    for iteration in 1:maxiter
        rho=dot(rr,r)
        if abs(rho)<1e-300 || abs(omega)<1e-300;break;end
        beta=(rho/rho_old)*(alpha/omega)
        @. pv=r+beta*(pv-omega*v)
        apply!(v,pv)
        denom=dot(rr,v);abs(denom)<1e-300 && break
        alpha=rho/denom
        @. s=r-alpha*v
        if norm(s)<tol
            @. x=x+alpha*pv
            return (iterations=iteration,residual=norm(s),converged=true)
        end
        apply!(t,s)
        tt=dot(t,t);tt<1e-300 && break
        omega=dot(t,s)/tt
        @. x=x+alpha*pv+omega*s
        @. r=s-omega*t
        err=norm(r)
        if verbose && iteration%100==0
            println(label," iteration ",iteration," residual ",err);flush(stdout)
        end
        err<tol && return (iterations=iteration,residual=err,converged=true)
        rho_old=rho
    end
    (iterations=maxiter,residual=err,converged=false)
end

function tensor_policy_value!(dest,source,ws)
    E0=ws.work[2];E1=ws.work[3]
    continuation!(E0,E1,source,ws)
    pol=ws.policy;nx,nq=size(source)
    @threads :static for q in 1:nq
        pool=0.
        @inbounds for x in 1:nx
            mixed=(1-pol.probability[x,q])*E0[x,q]+pol.probability[x,q]*E1[x,q]
            dest[x,q]=mixed;pool+=pol.reset[x,q]*mixed
        end
        @inbounds @simd for x in 1:nx
            dest[x,q]=(1-pol.exit[x,q])*(pol.keep[x,q]*dest[x,q]+pol.mix[x,q]*pool)
        end
    end
    dest
end
function tensor_reward!(dest,ws,p,n)
    pol=ws.policy;nx,nq=size(dest)
    @threads :static for q in 1:nq
        pool=0.
        @inbounds for x in 1:nx
            flow=n.dt*(pol.profit[x,q]-rd_expense(pol.k[x,q],p))
            dest[x,q]=flow;pool+=pol.reset[x,q]*flow
        end
        @inbounds @simd for x in 1:nx
            paid=pol.mix[x,q]*(1-pol.reset[x,q])
            dest[x,q]=(1-pol.exit[x,q])*(pol.keep[x,q]*dest[x,q]+pol.mix[x,q]*pool-p.K*paid+pol.entropy[x,q])
        end
    end
    dest
end
function tensor_firm!(ws,p,n,Y,w,G;verbose=false)
    check=ws.work[5];err=Inf
    discount=exp(-(p.rho+(p.gamma-1)*G/(p.epsilon-1))*n.dt)
    tolerance=min(n.terminal_tolerance,1e-10)
    rhs=similar(ws.V)
    function apply!(dest,source)
        tensor_policy_value!(dest,source,ws)
        @. dest=source-discount*dest
    end
    for iteration in 1:n.terminal_iterations
        tensor_bellman!(check,ws.V,ws,p,n,Y,w,G;improve=true)
        err=maximum(abs,check.-ws.V)
        if verbose
            println("  firm policy ",iteration," residual ",err);flush(stdout)
        end
        err<tolerance && return err,iteration
        tensor_reward!(rhs,ws,p,n)
        # Avoid oversolving a provisional policy; the final Bellman check retains
        # the original strict tolerance. This is an inexact Howard iteration.
        linear_tolerance=max(min(1e-10,tolerance/10),min(1e-4,0.01*err))
        status=bicgstab!(ws.V,apply!,rhs;tol=linear_tolerance,maxiter=600,label="value",verbose=verbose)
        status.converged || println("  value linear iteration ",status)
    end
    throw(ErrorException("large firm iteration failed with residual $err"))
end
function tensor_invariant!(ws;verbose=false)
    rhs=copy(ws.mu);rhs./=sum(rhs)
    function apply!(dest,source)
        tensor_forward!(dest,source,ws)
        total=sum(source)
        @. dest=source-dest+rhs*total
    end
    status=bicgstab!(ws.mu,apply!,rhs;tol=2e-13,maxiter=2000,label="distribution",verbose=verbose)
    # Remove only numerical negative mass, and verify the actual stochastic equation.
    negative=0.
    for x in ws.mu;negative-=min(x,0.);end
    @. ws.mu=max(ws.mu,0.)
    ws.mu./=sum(ws.mu)
    out=ws.work[5];tensor_forward!(out,ws.mu,ws)
    err=sum(abs,out.-ws.mu)
    if err>1e-8
        throw(ErrorException("large invariant distribution failed: L1=$err negative=$negative linear=$status"))
    end
    err,negative,status
end

function tensor_moments(ws,p,n,C,Y,w,G)
    continuation!(ws.work[2],ws.work[3],ws.V,ws)
    E0=ws.work[2];E1=ws.work[3]
    disc=exp(-(p.rho+(p.gamma-1)*G/(p.epsilon-1))*n.dt)
    post=ws.work[5];tensor_operating_forward!(post,ws.mu,ws)
    nx,nq=size(post);center=log(p.epsilon/(p.epsilon-1)*w/p.A)
    # Per-block reductions avoid slow threaded atomics and large state-vector temporaries.
    sums=zeros(22,nq)
    violations=zeros(nq);kkt_errors=zeros(nq)
    @threads :static for q in 1:nq
        a,b,v=ws.g.qstates[q]
        @inbounds for x in 1:nx
            mass=post[x,q];before=ws.mu[x,q];pol=ws.policy
            xx=ws.g.axes[1][x]+center
            prob=pol.probability[x,q];k=pol.k[x,q]
            rate=p.lambda0*k^p.omega
            sums[1,q]+=mass*exp(a+(1-p.epsilon)*xx)
            sums[2,q]+=mass*exp(a-p.epsilon*xx)
            sums[3,q]+=before*b
            sums[4,q]+=before*a
            sums[5,q]+=before*v
            sums[6,q]+=before*(1-pol.exit[x,q])*pol.mix[x,q]*(1-pol.reset[x,q])/n.dt
            exited=before*pol.exit[x,q]/n.dt
            sums[19,q]+=exited
            sums[20,q]+=exited*(p.premium-b)
            sums[21,q]+=exited*(p.premium-a)
            sums[22,q]+=exited*b
            sums[7,q]+=mass*rd_expense(k,p)
            sums[17,q]+=mass*k
            sums[18,q]+=mass*(k>0)
            sums[8,q]+=mass*rate
            sums[9,q]+=mass*rate*(p.premium-b)
            clip=ws.drift.outside[x] ? 1. : (1-prob)*ws.K0.clipping[q]+prob*ws.K1.clipping[q]
            sums[10,q]+=mass*clip
            sums[11,q]+=mass*prob*(p.premium-b)/n.dt
            sums[12,q]+=mass*(1-prob)*G
            cap=max(pol.profit[x,q],0.)
            gain=disc*(E1[x,q]-E0[x,q])
            desired=tensor_rd(gain,p,n.dt)
            binding=k>0 && rd_expense(k,p)>=cap*(1-1e-9)
            sums[13,q]+=mass*binding
            sums[14,q]+=mass*(cap<=0)
            sums[15,q]+=mass*max(desired-k,0.)
            violations[q]=max(violations[q],rd_expense(k,p)-cap)
            if k>0 && gain>0 && p.lambda0>0
                logm=log(p.lambda0*p.omega*gain)+(p.omega-1)*log(k)-rate*n.dt
                marginal=exp(clamp(logm,-745.,700.))
                eta=binding ? max(marginal-1,0.) : 0.
                sums[16,q]+=mass*eta
                stationarity_error=binding ? max(1-marginal,0.) : abs(marginal-1)
                kkt_errors[q]=max(kkt_errors[q],stationarity_error)
            end
        end
    end
    totals=vec(sum(sums,dims=2));N=Y*totals[2]/p.A
    residual=[log(totals[1]),(Y-C-p.K*totals[6]-totals[7])/C,
              log(w)-log(p.chi)-p.varphi*log(N)-p.gamma*log(C),totals[3]]
    moments=(C=C,Y=Y,w=w,N=N,G=G,growth=G/(p.epsilon-1),
        price_frequency=totals[6],research=totals[7],variable_research=totals[17],
        rd_active_mass=totals[18],fixed_research=p.F_R*totals[18],innovation_frequency=totals[8],
        exit_rate=totals[19],entry_rate=totals[19],replacement_quality_flux=totals[20],replacement_taste_flux=totals[21],
        mean_exit_b=totals[19]>0 ? totals[22]/totals[19] : 0.,
        entry_value=sum(ws.entry.*ws.V),exit_max_value=maximum(ws.V.*ws.policy.exit),
        quality_flux=totals[9]+totals[20],clipping=totals[10],mean_b=totals[3],mean_a=totals[4],mean_variance=totals[5],
        continuous_growth_error=G-totals[9]-totals[20],
        taste_mean_error=totals[4]+G-totals[21],
        split_taste_mean_error=totals[4]-totals[3]-n.dt*totals[20]-
            n.dt*exp(-n.dt)/(-expm1(-n.dt))*totals[21]+G*n.dt/(-expm1(-n.dt)),
        split_growth_error=totals[12]-totals[11]-totals[20],
        budget_error=Y-p.K*totals[6]-totals[7]-C,
        rd_binding_mass=totals[13],rd_zero_cashflow_mass=totals[14],
        rd_cashflow_shortfall=totals[15],rd_mean_financing_multiplier=totals[16],
        rd_max_cashflow_violation=maximum(violations),rd_max_kkt_residual=maximum(kkt_errors))
    residual,moments
end
function tensor_residual!(ws,z,p,n;verbose=false,update=true)
    C,Y,w=exp.(z[1:3]);G=z[4]
    update && update_kernels!(ws,p,n,G)
    ferr,fit=tensor_firm!(ws,p,n,Y,w,G;verbose=verbose)
    derr,neg,status=tensor_invariant!(ws;verbose=verbose)
    residual,moments=tensor_moments(ws,p,n,C,Y,w,G)
    (residual=residual,moments=moments,firm_error=ferr,firm_iterations=fit,
     distribution_error=derr,negative_mass=neg,linear=status)
end

function solve_large_bgp(p,n,ini;guess=nothing,verbose=true,warmstart=nothing)
    candidate_workspace=Ref{Any}(nothing)
    if p.F_R>0 && n.bgp_branch=="auto"
        candidate=solve_zero_research_bgp(p,n,ini;guess=guess,verbose=verbose,warmstart=warmstart,workspace_sink=candidate_workspace)
        candidate!==nothing && return candidate
    end
    z=guess===nothing ? [log(.336),log(.338),log(.87),.0055] : copy(guess)
    # A rejected zero-growth candidate must not start the growth search at G=0:
    # that boundary can have multiple closed classes and a singular forward solve.
    z[4]<=0 && (z[4]=.0055)
    reused=candidate_workspace[]!==nothing
    ws=reused ? candidate_workspace[] : tensor_workspace(p,n,ini,z[4])
    reused && update_kernels!(ws,p,n,z[4])
    if warmstart !== nothing && !reused
        size(warmstart.V)==size(ws.V) || error("Warm-start value dimensions do not match")
        size(warmstart.mu)==size(ws.mu) || error("Warm-start mass dimensions do not match")
        ws.V .= warmstart.V; ws.mu .= warmstart.mu
        ws.mu ./=sum(ws.mu)
    end
    result=tensor_residual!(ws,z,p,n;verbose=verbose,update=false)
    history=Float64[]
    for iteration in 1:n.max_iterations
        err=norm(result.residual,Inf);push!(history,err)
        println("LARGE BGP iteration ",iteration," residual ",err," growth ",z[4]);flush(stdout)
        mkpath(n.output)
        open(joinpath(n.output,"large_progress.toml"),"w") do io
            TOML.print(io,Dict("iteration"=>iteration,"unknowns"=>z,"residuals"=>result.residual,
                "equilibrium_error"=>err,"firm_error"=>result.firm_error,"distribution_error"=>result.distribution_error,
                "research"=>result.moments.research,"rd_active_mass"=>result.moments.rd_active_mass,
                "exit_rate"=>result.moments.exit_rate))
        end
        err<n.tolerance && break
        J=zeros(4,4)
        # Policies depend on Y and w only through F=Y*exp((1-epsilon)*center).
        # Differentiate only log Y and G; C and w columns follow analytically.
        J[:,1]=[0.,-(1+result.residual[2]),-p.gamma,0.]
        for j in (2,4)
            trial=copy(z);h=j==4 ? 2e-6 : 1e-5;trial[j]+=h
            println("  aggregate derivative ",j);flush(stdout)
            rt=tensor_residual!(ws,trial,p,n;verbose=verbose,update=j==4)
            J[:,j]=(rt.residual-result.residual)/h
        end
        J[:,3]=(1-p.epsilon).*J[:,2]
        J[1,3]+=1-p.epsilon
        J[2,3]-=(1-p.epsilon)*exp(z[2]-z[1])
        J[3,3]+=1+p.varphi
        accepted=false
        for damping in (0.,1e-5,1e-3,0.1)
            step=-(J'*J+damping*I)\(J'*result.residual)
            step*=min(1.,.20/max(norm(step,Inf),eps()))
            for alpha in (1.,.5,.1,.025)
                trial=z+alpha*step
                trial[4]>0 || continue
                rt=tensor_residual!(ws,trial,p,n;verbose=verbose)
                if norm(rt.residual)<norm(result.residual)
                    z=trial;result=rt;accepted=true;break
                end
            end
            accepted && break
        end
        accepted || break
    end
    # Restore the actual accepted aggregate state after derivative/line-search trials.
    result=tensor_residual!(ws,z,p,n;verbose=verbose)
    merge(result,(large=true,converged=norm(result.residual,Inf)<n.tolerance,
                  unknowns=z,history=history,workspace=ws,
                  rates=bgp_growth_rates(z[4],p)))
end

function enhance_report!(data)
    haskey(data,"growth") || return data
    g=data["growth"];G=data["G"];pi=data["quality_adjusted_inflation"]
    data["consumption_growth_pct_per_year"]=100*expm1(g)
    data["quality_growth_pct_per_year"]=100*expm1(G)
    data["ces_inflation_pct_per_year"]=100*expm1(pi)
    data["nominal_price_growth_pct_per_year"]=100*expm1(pi+g)
    data["real_interest_rate"]=data["interest"]-pi
    haskey(data,"menu_expenditure") && (data["menu_expenditure_share"]=data["menu_expenditure"]/data["Y"])
    data["rd_expenditure_share"]=data["research"]/data["Y"]
    data
end
function write_solution_csv(data,path)
    enhance_report!(data)
    keys_sorted=sort!(collect(keys(data)))
    csv(x)=x isa AbstractString ? "\""*replace(x,"\""=>"\"\"")*"\"" : string(x)
    open(path,"w") do io
        println(io,join(keys_sorted,","))
        println(io,join((csv(data[k]) for k in keys_sorted),","))
    end
end
function write_large_bgp_results(result,p,n;directory=n.output)
    mkpath(directory);ws=result.workspace
    data=Dict{String,Any}(string(k)=>v for (k,v) in pairs(result.moments))
    for (key,value) in pairs(result.rates);data[string(key)]=value;end
    merge!(data,Dict("converged"=>result.converged,"max_equilibrium_residual"=>norm(result.residual,Inf),
        "bellman_residual"=>result.firm_error,"distribution_residual"=>result.distribution_error,
        "mass_error"=>abs(sum(ws.mu)-1),"negative_mass_before_clipping"=>result.negative_mass,
        "state_count"=>prod(ws.g.dims),"nx"=>ws.g.dims[1],"na"=>ws.g.dims[2],
        "nb"=>ws.g.dims[3],"nv"=>ws.g.dims[4],"dt"=>n.dt,"julia_version"=>string(VERSION),
        "threads"=>nthreads(),"price_temperature"=>n.price_temperature,
        "backend"=>"tensor-factorized matrix-free","scope"=>"stationary finite-grid BGP"))
    price_mass=vec(sum(ws.mu;dims=2))
    data["rd_cashflow_constraint"]=true
    data["F_R"]=p.F_R
    data["lambda0"]=p.lambda0
    for field in fieldnames(Parameters);data["parameter_"*string(field)]=getfield(p,field);end
    data["replacement_entry"]=p.replacement_entry
    data["equilibrium_selection"]=hasproperty(result,:equilibrium_selection) ? result.equilibrium_selection : "positive-growth aggregate root"
    data["price_grid_min"]=first(ws.g.axes[1])
    data["price_grid_max"]=last(ws.g.axes[1])
    data["price_grid_spacing"]=minimum(diff(ws.g.axes[1]))
    data["price_grid_spacing_max"]=maximum(diff(ws.g.axes[1]))
    data["bgp_branch"]=n.bgp_branch
    data["price_nodes_mass_above_1e6"]=count(>(1e-6),price_mass)
    data["price_effective_nodes"]=1/sum(abs2,price_mass)
    data["price_boundary_mass"]=first(price_mass)+last(price_mass)
    post=ws.work[5];tensor_operating_forward!(post,ws.mu,ws)
    data["price_drift_clipping"]=sum(post[x,q] for q in axes(post,2),x in axes(post,1) if ws.drift.outside[x])
    data["menu_expenditure"]=p.K*result.moments.price_frequency
    write_solution_csv(data,joinpath(directory,"solution.csv"))
    open(joinpath(directory,"balanced_growth.toml"),"w") do io;TOML.print(io,data);end
    open(joinpath(directory,"checkpoint.toml"),"w") do io
        TOML.print(io,Dict("unknowns"=>result.unknowns,"residuals"=>result.residual,"history"=>result.history))
    end
    data
end

# Cache validity covers all numerical model sources, not just the tensor backend.
function bgp_cache_signature(config,z)
    sources=("DemandUncertainty.jl","BalancedGrowth.jl","LargeBGP.jl","ZeroResearchBGP.jl","ReplacementEntry.jl")
    (read(config,String),z,string(VERSION),
        join((read(joinpath(@__DIR__,file),String) for file in sources),"\n"))
end
