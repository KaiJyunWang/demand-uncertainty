# Conservative mass transfer and multilinear value interpolation between BGP grids.
# Both policies and the equilibrium are subsequently re-solved.
using Serialization,TOML,Interpolations
function price_regrid(path,config,p,n,guess)
    saved=deserialize(path)
    old=TOML.parse(saved.signature[1])
    on=DemandUncertainty.Numerics(; (Symbol(k)=>v for (k,v) in old["numerical"])...)
    oldaxes=(DemandUncertainty.price_axis(on),
        DemandUncertainty.taste_axis(on),
        DemandUncertainty.belief_axis(on),
        collect(range(0,old["economic"]["sigma_Q"]^2,length=on.nv)))
    newaxes=DemandUncertainty.tensor_grid(p,n).axes
    guess==saved.signature[2] || error("Warm-start aggregates must match saved values")
    itp=extrapolate(interpolate(oldaxes,reshape(saved.V,length.(oldaxes)),Gridded(Linear())),Flat())
    dims=length.(newaxes);V=zeros(dims);mu=zeros(dims)
    Threads.@threads for j in eachindex(V)
        I=CartesianIndices(dims)[j]
        V[j]=itp(ntuple(d->newaxes[d][I[d]],4)...)
    end
    linear=LinearIndices(dims)
    oldmass=reshape(saved.mu,length.(oldaxes))
    for I in CartesianIndices(oldmass)
        m=oldmass[I];m<=0 && continue
        bw=ntuple(d->DemandUncertainty.bracket(newaxes[d],oldaxes[d][I[d]]),4)
        for mask in 0:15
            J=ntuple(d->bw[d][1]+((mask>>(d-1))&1),4)
            weight=prod(((mask>>(d-1))&1)==1 ? bw[d][2] : 1-bw[d][2] for d in 1:4)
            weight>0 && (mu[linear[J...]]+=m*weight)
        end
    end
    println("Regridded warm start: mass ",sum(mu)," dimensions ",dims);flush(stdout)
    (V=reshape(V,n.nx,:),mu=reshape(mu,n.nx,:))
end
