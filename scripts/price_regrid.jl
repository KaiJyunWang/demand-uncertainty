# Transfer a converged solution to a different price grid as an initial guess.
# Mass is deposited linearly; values are linearly interpolated. Both are re-solved.
using Serialization,TOML
function price_regrid(path,config,p,n,guess)
    saved=deserialize(path)
    old=TOML.parse(saved.signature[1]);new=TOML.parsefile(config)
    old["economic"]==new["economic"] || error("Price regridding requires unchanged economic parameters")
    for key in ("na","nb","nv","amin","amax","bmin","bmax")
        old["numerical"][key]==new["numerical"][key] || error("Changed nonprice grid: $key")
    end
    guess==saved.signature[2] || error("Use the cached equilibrium aggregates as the regrid guess")
    ox=collect(range(old["numerical"]["xmin"],old["numerical"]["xmax"];length=old["numerical"]["nx"]))
    ax=collect(range(n.xmin,n.xmax;length=n.nx))
    nq=size(saved.V,2);V=zeros(n.nx,nq);mu=similar(V);fill!(mu,0.)
    for i in eachindex(ax)
        left,w=DemandUncertainty.bracket(ox,ax[i])
        @views V[i,:].=(1-w).*saved.V[left,:].+w.*saved.V[left+1,:]
    end
    for i in eachindex(ox)
        left,w=DemandUncertainty.bracket(ax,ox[i])
        @views mu[left,:].+=(1-w).*saved.mu[i,:]
        @views mu[left+1,:].+=w.*saved.mu[i,:]
    end
    println("Regridded warm start: mass ",sum(mu),", price spacing ",ax[2]-ax[1]);flush(stdout)
    (V=V,mu=mu)
end
