


include("util/flux.jl")
include("util/device.jl")

function update_target_network(model, target_network)
    for ps ∈ zip(params(agent.model),
                 params(agent.target_network))
        copyto!(ps[2], ps[1])
    end
end


