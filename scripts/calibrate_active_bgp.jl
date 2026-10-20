# Construct a positive-R&D/turnover equilibrium, calibrating the household scale
# chi from its labour condition. No firm's policy or exit flow is imposed.
# julia --project=. scripts/calibrate_active_bgp.jl config seed_checkpoint warm_cache
using DemandUncertainty,TOML,Serialization,LinearAlgebra
include(joinpath(@__DIR__,"price_regrid.jl"))
const DU=DemandUncertainty
function main(ARGS)
config=abspath(ARGS[1]);p,n,ini=load_config(config)
@assert n.bgp_branch=="positive" && p.replacement_entry && p.gamma==1
seed=TOML.parsefile(ARGS[2])["unknowns"]
saved=deserialize(ARGS[3])
warm=price_regrid(ARGS[3],config,p,n,saved.signature[2])
G=max(seed[4],.001)
center=log(p.epsilon/(p.epsilon-1)*exp(seed[3])/p.A)
F=exp(seed[2]+(1-p.epsilon)*center)
w0=p.A*(p.epsilon-1)/p.epsilon
ws=DU.tensor_workspace(p,n,ini,G);ws.V .=warm.V;ws.mu .=warm.mu
history=Float64[]
function evaluate(G)
    DU.tensor_residual!(ws,[log(.33),log(F),log(w0),G],p,n;verbose=true)
end
result=evaluate(G)
lo=nothing;hi=nothing
for iteration in 1:30
    f=result.moments.mean_b;push!(history,abs(f))
    println("ACTIVE CALIBRATION ",iteration," G=",G," mean_b=",f,
        " research=",result.moments.research," exit=",result.moments.exit_rate);flush(stdout)
    abs(f)<.8*n.tolerance && break
    f>0 ? (lo=(G,f)) : (hi=(G,f))
    h=max(2e-6,1e-4*G)
    trial=evaluate(G+h)
    slope=(trial.moments.mean_b-f)/h
    next=G-f/slope
    if !isfinite(next) || next<=0 || (lo!==nothing && hi!==nothing && !(lo[1]<next<hi[1]))
        next=lo!==nothing && hi!==nothing ? (lo[1]+hi[1])/2 : max(.001,G*(f>0 ? 1.2 : .8))
    end
    next=clamp(next,.5*G,1.5*G)
    accepted=false
    for alpha in (1.,.5,.25,.1)
        proposed=G+alpha*(next-G)
        tested=evaluate(proposed)
        if abs(tested.moments.mean_b)<abs(f)
            G=proposed;result=tested;accepted=true;break
        end
        tested.moments.mean_b>0 ? (lo=(proposed,tested.moments.mean_b)) : (hi=(proposed,tested.moments.mean_b))
    end
    if !accepted
        lo!==nothing && hi!==nothing || error("No improving growth calibration")
        G=(lo[1]+hi[1])/2;result=evaluate(G)
    end
end
@assert abs(result.moments.mean_b)<n.tolerance
S=exp(result.residual[1]); L=result.moments.N*p.A/F
newcenter=log(S)/(p.epsilon-1)
Y=F*S; C=Y-p.K*result.moments.price_frequency-result.moments.research
w=w0*exp(newcenter); N=F*exp(-newcenter)*L/p.A
chi=w/(N^p.varphi*C^p.gamma)
updated=Parameters(; (field=>(field==:chi ? chi : getfield(p,field)) for field in fieldnames(Parameters))...)
z=[log(C),log(Y),log(w),G]
verified=DU.tensor_residual!(ws,z,updated,n;verbose=true)
@assert norm(verified.residual,Inf)<n.tolerance
@assert verified.moments.research>0 && verified.moments.exit_rate>0
# Record the calibrated primitive before making a signature for the new cache.
d=TOML.parsefile(config);d["economic"]["chi"]=chi
open(config,"w") do io;TOML.print(io,d);end
mkpath(n.output);cp(config,joinpath(n.output,"run_calibration.toml");force=true)
r=merge(verified,(large=true,converged=true,unknowns=z,history=history,workspace=ws,
    rates=bgp_growth_rates(G,updated),
    equilibrium_selection="positive growth with endogenous R&D and turnover; chi calibrated from labour supply"))
report=write_bgp_results(r,updated,n)
serialize(joinpath(n.output,"plot_state.jls"),(signature=DU.bgp_cache_signature(config,z),V=ws.V,mu=ws.mu))
open(joinpath(n.output,"active_calibration.toml"),"w") do io
    TOML.print(io,Dict("operating_profit_scale"=>F,"chi_before"=>p.chi,"chi_after"=>chi,
        "G"=>G,"mean_b"=>verified.moments.mean_b,"max_equilibrium_residual"=>norm(verified.residual,Inf),
        "reason"=>"chi is the household labour-disutility scale; firm's policies are re-solved and not imposed"))
end
if (n.nx,n.na,n.nb,n.nv)==(51,51,51,51)
    cp(joinpath(n.output,"solution.csv"),joinpath(@__DIR__,"..","bgp_solution_51.csv");force=true)
end
println(report)

end
main(ARGS)
