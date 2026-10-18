using Test, DemandUncertainty, TOML, Serialization
include(joinpath(@__DIR__,"..","scripts","price_regrid.jl"))
@testset "price-grid warm-start transfer" begin
    mktempdir() do dir
        old=TOML.parsefile(joinpath(@__DIR__,"..","calibration_bgp.toml"))
        for key in ("nx","na","nb","nv"); old["numerical"][key]=5;end
        old["numerical"]["xmin"]=-.45;old["numerical"]["xmax"]=.45
        oldpath=joinpath(dir,"old.toml")
        open(io->TOML.print(io,old),oldpath,"w")
        new=deepcopy(old);new["numerical"]["xmin"]=-.3;new["numerical"]["xmax"]=.3
        newpath=joinpath(dir,"new.toml")
        open(io->TOML.print(io,new),newpath,"w")
        p,n,ini=load_config(newpath);z=[0.,0.,0.,.003]
        ox=collect(range(-.45,.45;length=5))
        V=repeat(ox,1,125);mu=zeros(5,125);mu[2:4,1]=[.2,.5,.3]
        cache=joinpath(dir,"state.jls")
        serialize(cache,(signature=(read(oldpath,String),z),V=V,mu=mu))
        out=price_regrid(cache,newpath,p,n,z)
        ax=collect(range(-.3,.3;length=5))
        @test sum(out.mu)≈1.
        @test minimum(out.mu)>=0.
        @test out.V≈repeat(ax,1,125)
        @test sum(ax.*vec(sum(out.mu;dims=2)))≈sum(ox.*vec(sum(mu;dims=2)))
        @test_throws ErrorException price_regrid(cache,newpath,p,n,ones(4))
    end
end
