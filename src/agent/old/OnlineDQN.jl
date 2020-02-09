using Flux
using Random

mutable struct OnlineDQNAgent{M, TN, O, LU, AP<:AbstractValuePolicy, Φ, OB<:OnlineReplay} <: AbstractAgent
    model::M
    target_network::TN
    opt::O
    lu::LU
    ap::AP
    buffer::OB
    γ::Float32
    batch_size::Int64
    tn_counter_init::Int64
    target_network_counter::Int64
    action::Int64
    prev_s::Φ
    step::Int64
end

OnlineDQNAgent(model, target_network, opt, lu, ap, γ, batch_size, tn_counter_init, s) =
    OnlineDQNAgent(model,
             target_network,
             opt,
             lu,
             ap,
             OnlineReplay(batch_size,
                          (Array{Float32, 1}, Int64, Array{Float32, 1}, Float32, Bool),
                          (:s, :a, :sp, :r, :t)),
             γ,
             batch_size,
             tn_counter_init,
             tn_counter_init,
             0,
             s, 0)


function RLCore.start!(agent::OnlineDQNAgent, env_s_tp1, rng::AbstractRNG; kwargs...)
    # Start an Episode
    agent.action = sample(agent.ap,
                          agent.model(env_s_tp1),
                          rng)
    agent.step += 1
    agent.prev_s .= env_s_tp1
    return agent.action
end

function RLCore.step!(agent::OnlineDQNAgent, env_s_tp1, r, terminal, rng::AbstractRNG; kwargs...)
    
    add!(agent.buffer, (Float32.(agent.prev_s), agent.action, copy(Float32.(env_s_tp1)), r, terminal))

    if agent.step > 1000
        e = sample(agent.buffer, agent.batch_size; rng=rng)
        update_params!(agent, e)
    end

    agent.prev_s .= env_s_tp1
    agent.action = sample(agent.ap,
                          agent.model(agent.prev_s),
                          rng)
    agent.step += 1
    return agent.action
end

function update_params!(agent::OnlineDQNAgent, e)

    if agent.tn_counter_init > 0
        update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t, agent.target_network)

        if agent.target_network_counter == 1
            agent.target_network_counter = agent.tn_counter_init
            agent.target_network = mapleaves(
                Flux.Tracker.data,
                deepcopy(agent.model))
        else
            agent.target_network_counter -= 1
        end
    else
        update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t)
    end
    return nothing
    
end
