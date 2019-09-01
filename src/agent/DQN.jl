
using Flux
using Random

mutable struct DQNAgent{M, TN, O, AP<:AbstractValuePolicy, Φ} <: AbstractAgent
    model::M
    target_network::TN
    opt::O
    ap::AP
    er::ExperienceReplay
    γ::Float32
    batch_size::Int64
    tn_counter_init::Int64
    target_network_counter::Int64
    action::Int64
    prev_s::Φ
end

function JuliaRL.start!(agent::DQNAgent, env_s_tp1, rng::AbstractRNG=Random.GLOBAL_RNG; kwargs...)
    # Start an Episode
    agent.action = sample(agent.ap,
                          # Flux.data(agent.model(env_s_tp1)),
                          agent.model(env_s_tp1),
                          rng)
    return agent.action
end

function JuliaRL.step!(agent::DQNAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG; kwargs...)
    
    add!(agent.er, (Float32.(agent.prev_s), agent.action, Float32.(env_s_tp1), r, !terminal))
    
    if size(agent.er)[1] > 1000
        update_params!(agent, rng)
    end
    
    
    agent.prev_s .= env_s_tp1
    # println(Flux.data(agent.model(agent.prev_s)))
    agent.action = sample(agent.ap,
                          # Flux.data(agent.model(agent.prev_s)),
                          agent.model(agent.prev_s),
                          rng)

    return agent.action
end

function update_params!(agent, rng::AbstractRNG)
    

    exp = sample(agent.er, agent.batch_size; rng=rng)
    s_t = reduce(vcat, exp[!, :s]')'
    ps = params(agent.model)
    
    γ = agent.γ.*exp[!, :t]
    action_idx = [CartesianIndex(exp[i, :a], i) for i in 1:agent.batch_size]
    
    q_tp1 = maximum(
        # transpose(Flux.data(agent.target_network(s_tp1)));
        transpose(agent.target_network(reduce(vcat, exp[!, :s′]')'));
        dims=2)[:,1]

    target = exp[!, :r] .+ γ.*q_tp1

    # gs = Flux.gradient(ps) do
    #     Flux.mse(exp[!, :r] .+ agent.γ.*q_tp1.*(exp[!, :t]), q_t)
    # end

    # Flux.Tracker.update!(agent.opt, ps, gs)
    
    # model = agent.model
    
    gs = Flux.gradient(ps) do
        q_t = getindex(agent.model(s_t), action_idx)
        return Flux.mse(target, q_t)
    end
    Flux.Optimise.update!(agent.opt, ps, gs)

    if agent.target_network_counter == 1
        agent.target_network_counter = agent.tn_counter_init
        agent.target_network = deepcopy(agent.model)
        Flux.testmode!(agent.target_network)
    else
        agent.target_network_counter -= 1
    end
    
end
