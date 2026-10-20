using TOML
root=normpath(joinpath(@__DIR__,".."));out=joinpath(root,"results_bgp")
open(joinpath(out,"parameter_search.csv"),"w") do io
    println(io,"candidate,grid,F_R,lambda0,premium,sigma_Q,K,converged,rd_active_mass,exit_rate,G,clipping,equilibrium_error")
    for i in 1:9
        dir=joinpath("/tmp","active_bgp_search_$i")
        isfile(joinpath(dir,"balanced_growth.toml")) || continue
        r=TOML.parsefile(joinpath(dir,"balanced_growth.toml"));p=TOML.parsefile(joinpath(dir,"calibration.toml"))["economic"]
        println(io,join((i,r["nx"],p["F_R"],p["lambda0"],p["premium"],p["sigma_Q"],p["K"],
            r["converged"],r["rd_active_mass"],r["exit_rate"],r["G"],r["clipping"],r["max_equilibrium_residual"]),","))
    end
end
