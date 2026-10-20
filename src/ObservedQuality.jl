module ObservedQuality
using DemandUncertainty, LinearAlgebra, SparseArrays, TOML, Serialization, Printf, SHA
using Base.Threads
import DemandUncertainty: Parameters, bracket, tail_axis, operating_profit, rd_expense, step_rd_surplus

Base.@kwdef struct Settings
    nx::Int=101
    nz::Int=101
    nq::Int=101
    xmin::Float64=-0.35
    xmax::Float64=0.30
    zmin::Float64=-9.0
    zmax::Float64=3.9
    qmin::Float64=-8.0
    qmax::Float64=3.6
    dt::Float64=0.125
    price_temperature::Float64=1e-5
    tolerance::Float64=1e-5
    terminal_tolerance::Float64=1e-10
    terminal_iterations::Int=500
    max_iterations::Int=35
    draw_nodes::Int=21
    taste_nodes::Int=5
    output::String="results_observed"
end
function config(path)
    d=TOML.parsefile(path)
    get(d,"information","")=="observed_quality" || error("Expected observed_quality config")
    p=Parameters(; (Symbol(k)=>v for (k,v) in d["economic"])...)
    n=Settings(; (Symbol(k)=>v for (k,v) in d["numerical"])...)
    @assert p.gamma==1 && p.replacement_entry
    @assert p.rho>0 && p.sigma_a>0 && p.sigma_Q>0 && p.K>0 && p.F_R>=0
    @assert p.lambda0>=0 && 0<p.omega<1 && p.epsilon>1 && p.chi>0
    @assert min(n.nx,n.nz,n.nq)>=5 && n.dt>0 && n.price_temperature>=0
    p,n
end
function normal_quadrature(count)
    e=eigen(SymTridiagonal(zeros(count),sqrt.(Float64.(1:count-1))))
    e.values,vec(e.vectors[1,:].^2)
end
struct StateGrid
    axes::NTuple{3,Vector{Float64}}
    dims::NTuple{3,Int}
    qstates::Vector{NTuple{2,Float64}}
end
function grid(n)
    # Dense price/quality cores and explicit lower/upper tails, all three real states.
    ax=(tail_axis(n.xmin,n.xmax,n.nx,4,-.18;upper=4,corehi=.12),
        tail_axis(n.zmin,n.zmax,n.nz,div(n.nz,5),-1.5;upper=div(n.nz,8),corehi=1.5),
        tail_axis(n.qmin,n.qmax,n.nq,div(n.nq,5),-1.2;upper=div(n.nq,8),corehi=1.2))
    dims=length.(ax)
    StateGrid(ax,dims,vec([(ax[2][i],ax[3][j]) for i in 1:dims[2],j in 1:dims[3]]))
end
struct TensorKernel
    outgoing::SparseMatrixCSC{Float64,Int}
    incoming::SparseMatrixCSC{Float64,Int}
    clipping::Vector{Float64}
end
function quality_kernel(g,p,n,G;reset=false)
    rows=Int[];cols=Int[];vals=Float64[]
    clipping=zeros(length(g.qstates));linear=LinearIndices(g.dims[2:3])
    zn,zw=normal_quadrature(n.taste_nodes);qn,qw=normal_quadrature(n.draw_nodes)
    reset || (qn=[0.];qw=[1.])
    h=n.dt;e=exp(-h);sd=p.sigma_a*sqrt(-expm1(-2h)/2)
    for (j,(z,q)) in enumerate(g.qstates), (xi,wi) in zip(zn,zw), (epsj,wj) in zip(qn,qw)
        znext=q+(z-q)*e-G*h+sd*xi
        qnext=reset ? p.premium+p.sigma_Q*epsj : q-G*h
        prob=wi*wj
        outside=znext<first(g.axes[2]) || znext>last(g.axes[2]) || qnext<first(g.axes[3]) || qnext>last(g.axes[3])
        clipping[j]+=prob*outside
        iz,tz=bracket(g.axes[2],znext);iq,tq=bracket(g.axes[3],qnext)
        for dz in 0:1,dq in 0:1
            weight=prob*(dz==0 ? 1-tz : tz)*(dq==0 ? 1-tq : tq)
            weight<=0 && continue
            push!(rows,linear[iz+dz,iq+dq]);push!(cols,j);push!(vals,weight)
        end
    end
    out=sparse(rows,cols,vals,length(clipping),length(clipping))
    TensorKernel(out,sparse(transpose(out)),clipping)
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
    g::StateGrid
    K0::TensorKernel
    K1::TensorKernel
    drift::PriceDrift
    policy::TensorPolicy
    work::Vector{Matrix{Float64}}
    V::Matrix{Float64}
    mu::Matrix{Float64}
    entry::Matrix{Float64}
    draw_nodes::Int
    redraw_weights::Vector{Float64}
end
function workspace(p,n,G)
    g=grid(n);nx=n.nx;nq=length(g.qstates)
    ws=TensorWorkspace(g,quality_kernel(g,p,n,G),quality_kernel(g,p,n,G;reset=true),
        price_drift(g,p,n,G),TensorPolicy(nx,nq),[zeros(nx,nq) for _ in 1:5],
        zeros(nx,nq),zeros(nx,nq),zeros(nx,nq),n.draw_nodes,redraw_weights(g,p,n))
    for j in 1:nq,x in 1:nx
        z,q=g.qstates[j];ws.mu[x,j]=exp(-g.axes[1][x]^2/.02-z^2/.4-q^2/.5)
    end
    ws.mu./=sum(ws.mu)
    println("Observed-quality grid ",g.dims," states=",length(ws.V)," threads=",nthreads());flush(stdout)
    ws
end
function update_entry!(ws,p)
    fill!(ws.entry,0.)
    nodes,weights=normal_quadrature(ws.draw_nodes)
    linear=LinearIndices(ws.g.dims[2:3])
    iz,tz=bracket(ws.g.axes[2],p.premium)
    # Each draw's free initial price maximizes its interpolated value, not the
    # value averaged across draws. Different entrants may choose different prices.
    for (epsj,prob) in zip(nodes,weights)
        q=p.premium+p.sigma_Q*epsj
        iq,tq=bracket(ws.g.axes[3],q)
        inds=[linear[iz+dz,iq+dq] for dz in 0:1,dq in 0:1]
        weights4=[(dz==0 ? 1-tz : tz)*(dq==0 ? 1-tq : tq) for dz in 0:1,dq in 0:1]
        best=argmax([sum(weights4[k]*ws.V[x,inds[k]] for k in eachindex(inds)) for x in axes(ws.V,1)])
        for k in eachindex(inds);ws.entry[best,inds[k]]+=prob*weights4[k];end
    end
    ws.entry
end
function update_kernels!(ws,p,n,G)
    ws.K0=quality_kernel(ws.g,p,n,G)
    ws.K1=quality_kernel(ws.g,p,n,G;reset=true)
    ws.drift=price_drift(ws.g,p,n,G)
end
# Independent quality redraw is rank one along quality for each taste and price.
# Factor it instead of multiplying a dense redraw-by-OU transition.
function redraw_weights(g,p,n)
    nodes,weights=normal_quadrature(n.draw_nodes);out=zeros(g.dims[3])
    for (epsj,prob) in zip(nodes,weights)
        j,t=bracket(g.axes[3],p.premium+p.sigma_Q*epsj)
        out[j]+=prob*(1-t);out[j+1]+=prob*t
    end
    out
end
function redraw_value!(dest,source,ws)
    nx,nz,nq=ws.g.dims
    a=reshape(source,nx,nz,nq);b=reshape(dest,nx,nz,nq);weights=ws.redraw_weights
    @threads :static for z in 1:nz
        @inbounds for x in 1:nx
            total=0.
            for q in 1:nq;total+=weights[q]*a[x,z,q];end
            for q in 1:nq;b[x,z,q]=total;end
        end
    end
    dest
end
function redraw_forward!(dest,source,ws)
    nx,nz,nq=ws.g.dims
    a=reshape(source,nx,nz,nq);b=reshape(dest,nx,nz,nq);weights=ws.redraw_weights
    @threads :static for z in 1:nz
        @inbounds for x in 1:nx
            total=0.
            for q in 1:nq;total+=a[x,z,q];end
            for q in 1:nq;b[x,z,q]=weights[q]*total;end
        end
    end
    dest
end
function continuation!(E0,E1,V,ws)
    W=ws.work[1]
    price_value!(W,V,ws.drift)
    quality_apply!(E0,W,ws.K0.outgoing)
    redraw_value!(E1,W,ws)
    quality_apply!(W,E1,ws.K0.outgoing)
    copyto!(E1,W)
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
        a,b=ws.g.qstates[q]
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
    quality_apply!(temp,weighted,ws.K0.incoming)
    redraw_forward!(weighted,temp,ws)
    @. combined=combined+weighted
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

function tensor_operating_forward!(dest,source,ws)
    tensor_price_forward!(dest,source,ws.policy)
    exited=sum(source.*ws.policy.exit)
    @. dest=dest+exited*ws.entry
    dest
end

function moments(ws,p,n,C,Y,w,G)
    continuation!(ws.work[2],ws.work[3],ws.V,ws)
    disc=exp(-p.rho*n.dt)
    post=ws.work[5];tensor_operating_forward!(post,ws.mu,ws)
    center=log(p.epsilon/(p.epsilon-1)*w/p.A)
    sums=zeros(18,length(ws.g.qstates));violation=0.;kkt=0.
    for j in eachindex(ws.g.qstates), x in axes(post,1)
        z,q=ws.g.qstates[j];mass=post[x,j];before=ws.mu[x,j];pol=ws.policy
        xx=ws.g.axes[1][x]+center;k=pol.k[x,j];rate=p.lambda0*k^p.omega
        sums[1,j]+=mass*exp(z+(1-p.epsilon)*xx)
        sums[2,j]+=mass*exp(z-p.epsilon*xx)
        sums[3,j]+=before*q;sums[4,j]+=before*z
        sums[5,j]+=before*(1-pol.exit[x,j])*pol.mix[x,j]*(1-pol.reset[x,j])/n.dt
        out=before*pol.exit[x,j]/n.dt
        sums[6,j]+=out;sums[7,j]+=out*(p.premium-q);sums[8,j]+=out*(p.premium-z)
        sums[9,j]+=mass*rd_expense(k,p);sums[10,j]+=mass*k;sums[11,j]+=mass*(k>0)
        sums[12,j]+=mass*rate;sums[13,j]+=mass*rate*(p.premium-q)
        prob=pol.probability[x,j]
        sums[14,j]+=mass*((1-prob)*ws.K0.clipping[j]+prob*ws.K1.clipping[j])
        sums[15,j]+=mass*ws.drift.outside[x]
        cap=max(pol.profit[x,j],0.)
        sums[16,j]+=mass*(cap<=p.F_R)
        binding=k>0 && rd_expense(k,p)>=cap*(1-1e-9)
        sums[17,j]+=mass*binding
        # Exact q moment for OU-then-redraw split: q_after=(1-P)(q-Gh)+P*delta.
        sums[18,j]+=mass*prob*(p.premium-q+G*n.dt)/n.dt
        violation=max(violation,rd_expense(k,p)-cap)
        if k>0
            gain=disc*(ws.work[3][x,j]-ws.work[2][x,j])
            marginal=p.lambda0*p.omega*k^(p.omega-1)*exp(-rate*n.dt)*gain
            kkt=max(kkt,binding ? max(1-marginal,0.) : abs(marginal-1))
        end
    end
    t=vec(sum(sums,dims=2));N=Y*t[2]/p.A
    residual=[log(t[1]),(Y-C-p.K*t[5]-t[9])/C,
        log(w/(p.chi*N^p.varphi*C)),t[3]]
    d=Dict{String,Any}("C"=>C,"Y"=>Y,"w"=>w,"N"=>N,"G"=>G,
        "g"=>G/(p.epsilon-1),"price_frequency"=>t[5],"exit_rate"=>t[6],"entry_rate"=>t[6],
        "replacement_quality_flux"=>t[7],"replacement_taste_flux"=>t[8],
        "research"=>t[9],"variable_research"=>t[10],"fixed_research"=>p.F_R*t[11],
        "rd_active_mass"=>t[11],"innovation_frequency"=>t[12],"clipping"=>t[14],
        "price_drift_clipping"=>t[15],"rd_financially_excluded_mass"=>t[16],"rd_binding_mass"=>t[17],
        "mean_relative_quality"=>t[3],"mean_relative_taste"=>t[4],
        "continuous_growth_error"=>G-t[13]-t[7],"split_growth_error"=>G-t[18]-t[7],
        "continuous_taste_error"=>t[4]-t[3]+G-t[8],
        "split_taste_error"=>t[4]-t[3]-n.dt*t[7]-n.dt*exp(-n.dt)/(-expm1(-n.dt))*t[8]+G*n.dt/(-expm1(-n.dt)),
        "rd_max_cashflow_violation"=>violation,"rd_max_kkt_residual"=>kkt,
        "entry_value"=>sum(ws.entry.*ws.V),"exit_max_value"=>maximum(ws.V.*ws.policy.exit))
    residual,d
end

# The price-gap frame makes policies depend on the profit scale F and G only.
# Price-index/resources are solved analytically; chi is held fixed throughout.
function evaluate!(ws,p,n,logF,G)
    F=exp(logF);w0=p.A*(p.epsilon-1)/p.epsilon
    update_kernels!(ws,p,n,G)
    ferr,fit=tensor_firm!(ws,p,n,F,w0,G)
    derr,negative,_=tensor_invariant!(ws)
    rr,canonical=moments(ws,p,n,1.,F,w0,G)
    S=exp(rr[1]);center=log(S)/(p.epsilon-1)
    Y=F*S;C=Y-p.K*canonical["price_frequency"]-canonical["research"]
    C>0 || error("Nonpositive consumption")
    w=w0*exp(center)
    residual,report=moments(ws,p,n,C,Y,w,G)
    report["bellman_residual"]=ferr;report["distribution_residual"]=derr
    report["negative_mass_removed"]=negative;report["firm_iterations"]=fit
    (residual=[residual[3],residual[4]],full_residual=residual,report=report)
end
function warmstart!(ws,saved)
    oldax=saved.axes;newax=ws.g.axes;linear=LinearIndices(ws.g.dims)
    oldV=reshape(saved.V,length.(oldax));newV=reshape(ws.V,ws.g.dims)
    for I in CartesianIndices(newV)
        brackets=ntuple(d->bracket(oldax[d],newax[d][I[d]]),3)
        val=0.
        for mask in 0:7
            index=ntuple(d->brackets[d][1]+((mask>>(d-1))&1),3)
            weight=prod(((mask>>(d-1))&1)==1 ? brackets[d][2] : 1-brackets[d][2] for d in 1:3)
            val+=weight*oldV[index...]
        end
        newV[I]=val
    end
    fill!(ws.mu,0.);newmu=vec(ws.mu);oldmu=reshape(saved.mu,length.(oldax))
    for I in CartesianIndices(oldmu)
        oldmu[I]<=0 && continue
        brackets=ntuple(d->bracket(newax[d],oldax[d][I[d]]),3)
        for mask in 0:7
            index=ntuple(d->brackets[d][1]+((mask>>(d-1))&1),3)
            weight=prod(((mask>>(d-1))&1)==1 ? brackets[d][2] : 1-brackets[d][2] for d in 1:3)
            newmu[linear[index...]]+=weight*oldmu[I]
        end
    end
    ws.mu./=sum(ws.mu)
end
function solve(p,n;guess=[log(.312),.073],warm=nothing)
    theta=copy(guess);ws=workspace(p,n,theta[2])
    warm===nothing || warmstart!(ws,warm)
    result=evaluate!(ws,p,n,theta...);history=Float64[]
    for iteration in 1:n.max_iterations
        err=norm(result.residual,Inf);push!(history,err)
        println("OBSERVED BGP ",iteration," F=",exp(theta[1])," G=",theta[2]," residual=",result.residual,
            " RD=",result.report["research"]," active=",result.report["rd_active_mass"]," exit=",result.report["exit_rate"]);flush(stdout)
        mkpath(n.output)
        open(joinpath(n.output,"progress.toml"),"w") do io
            TOML.print(io,Dict("iteration"=>iteration,"unknowns"=>theta,"residuals"=>result.residual))
        end
        err<n.tolerance && break
        J=zeros(2,2)
        for j in 1:2
            h=j==1 ? 1e-4 : 1e-5
            trial=copy(theta);trial[j]+=h
            rt=evaluate!(ws,p,n,trial...)
            J[:,j]=(rt.residual-result.residual)/h
        end
        accepted=false
        for damping in (0.,1e-5,1e-3,.1)
            step=-(J'J+damping*I)\(J'result.residual)
            step*=min(1.,.15/max(abs(step[1]),eps()),.03/max(abs(step[2]),eps()))
            for alpha in (1.,.5,.1,.025)
                trial=theta+alpha*step
                trial[2]>0 || continue
                rt=evaluate!(ws,p,n,trial...)
                if norm(rt.residual)<norm(result.residual)
                    theta=trial;result=rt;accepted=true;break
                end
            end
            accepted && break
        end
        accepted || break
    end
    result=evaluate!(ws,p,n,theta...)
    (workspace=ws,theta=theta,report=result.report,residual=result.full_residual,
     converged=norm(result.full_residual,Inf)<n.tolerance,history=history)
end
function signature(configpath)
    bytes=vcat(read(configpath),read(@__FILE__),read(joinpath(@__DIR__,"DemandUncertainty.jl")))
    bytes2hex(sha256(bytes))
end
function save_result(result,p,n,configpath)
    mkpath(n.output);r=copy(result.report)
    r["information"]="observed_quality";r["state_dimensions"]=3;r["grid_counts"]=[n.nx,n.nz,n.nq]
    r["states"]=n.nx*n.nz*n.nq;r["dt"]=n.dt;r["julia_version"]=string(VERSION)
    r["converged"]=result.converged;r["max_equilibrium_residual"]=norm(result.residual,Inf)
    r["residuals"]=result.residual;r["consumption_growth_pct_per_year"]=100expm1(r["g"])
    r["inflation_pct_per_year"]=100expm1(p.pi_target)
    r["nominal_price_trend_pct_per_year"]=100expm1(p.pi_target+r["g"])
    for f in fieldnames(Parameters);r["parameter_"*string(f)]=getfield(p,f);end
    for (d,label) in enumerate(("price","taste","quality"))
        masses=dropdims(sum(reshape(result.workspace.mu,result.workspace.g.dims),dims=Tuple(setdiff(1:3,[d]))),dims=Tuple(setdiff(1:3,[d])))
        r[label*"_lower_boundary_mass"]=first(masses);r[label*"_upper_boundary_mass"]=last(masses)
    end
    open(joinpath(n.output,"balanced_growth.toml"),"w") do io;TOML.print(io,r);end
    columns=sort([k for (k,v) in r if v isa Number || v isa Bool || v isa String])
    open(joinpath(n.output,"solution.csv"),"w") do io
        println(io,join(columns,","));println(io,join((r[k] for k in columns),","))
    end
    cp(configpath,joinpath(n.output,"run_calibration.toml");force=true)
    open(joinpath(n.output,"checkpoint.toml"),"w") do io;TOML.print(io,Dict("unknowns"=>result.theta));end
    ws=result.workspace
    # No 4D learning cache is accepted or overwritten.
    serialize(joinpath(n.output,"observed_state.jls"),(signature=signature(configpath),
        axes=ws.g.axes,V=ws.V,mu=ws.mu,theta=result.theta,report=r,policy=ws.policy,entry=ws.entry))
    println("Saved ",n.output," converged=",result.converged," residual=",r["max_equilibrium_residual"])
    r
end

end # module
