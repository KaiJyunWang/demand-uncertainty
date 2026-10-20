# Entry is an injection into the population operator only. An incumbent's killed
# Bellman row has zero terminal payoff and never includes successor value.
function update_entry!(ws,p)
    fill!(ws.entry,0.)
    brackets=(bracket(ws.g.axes[2],p.premium),bracket(ws.g.axes[3],p.premium),bracket(ws.g.axes[4],p.sigma_Q^2))
    inds=Int[];weights=Float64[];linear=LinearIndices(ws.g.dims[2:4])
    for mask in 0:7
        I=ntuple(d->brackets[d][1]+((mask>>(d-1))&1),3)
        weight=prod(((mask>>(d-1))&1)==1 ? brackets[d][2] : 1-brackets[d][2] for d in 1:3)
        weight<=0 && continue
        push!(inds,linear[I...]);push!(weights,weight)
    end
    # Initial nominal price is free, and maximizes the interpolated entry value.
    values=[sum(weight*ws.V[x,q] for (q,weight) in zip(inds,weights)) for x in axes(ws.V,1)]
    best=argmax(values)
    for (q,weight) in zip(inds,weights);ws.entry[best,q]+=weight;end
    ws.entry
end
function tensor_operating_forward!(dest,source,ws)
    tensor_price_forward!(dest,source,ws.policy)
    exited=sum(source.*ws.policy.exit)
    @. dest=dest+exited*ws.entry
    dest
end
