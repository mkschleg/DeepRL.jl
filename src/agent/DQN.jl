
using Flux
using Random

mutable struct DQNAgent{M, TN, O, AP<:AbstractValuePolicy, Φ, ERP} <: AbstractAgent
    model::M
    target_network::TN
    opt::O
    ap::AP
    er::ExperienceReplay{ERP}
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
                          agent.model(env_s_tp1),
                          rng)
    return agent.action
end

function JuliaRL.step!(agent::DQNAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG; kwargs...)
    
    add!(agent.er, (Float32.(agent.prev_s), agent.action, copy(Float32.(env_s_tp1)), r, !terminal))
    
    if size(agent.er)[1] > 1000
        update_params!(agent, rng)
    end
    
    agent.prev_s .= env_s_tp1
    agent.action = sample(agent.ap,
                          agent.model(agent.prev_s),
                          rng)

    return agent.action
end

function update_params!(agent::DQNAgent, rng)
    

    e = sample(agent.er, agent.batch_size; rng=rng)
    s_t = hcat(e.s...)
    s_tp1 = hcat(e.sp...)
    ps = params(agent.model)

    γ = agent.γ.*(e.t)
    action_idx = [CartesianIndex(e.a[i], i) for i in 1:agent.batch_size]

    q_tp1 = maximum(agent.target_network(s_tp1); dims=1)[1,:]
    
    target = (e.r .+ γ.*q_tp1)
    gs = Flux.gradient(ps) do
        q_t = agent.model(s_t)[action_idx]
        return Flux.mse(target, q_t)
    end
    Flux.Optimise.update!(agent.opt, ps, gs)

    if agent.target_network_counter == 1
        agent.target_network_counter = agent.tn_counter_init
        agent.target_network = mapleaves(
            Flux.Tracker.data,
            deepcopy(agent.model))
    else
        agent.target_network_counter -= 1
    end

    return nothing
    
end
