
using Flux
using Random

mutable struct DQNAgent{M, TN, O, LU, AP<:AbstractValuePolicy, Φ, ERP} <: AbstractAgent
    model::M
    target_network::TN
    opt::O
    lu::LU
    ap::AP
    er::ExperienceReplay{ERP}
    γ::Float32
    batch_size::Int64
    tn_counter_init::Int64
    target_network_counter::Int64
    action::Int64
    prev_s::Φ
    INFO::Dict{Symbol, Any}
end

DQNAgent(model, target_network, opt, lu, ap, size_buffer, γ, batch_size, tn_counter_init, s) =
    DQNAgent(model,
             target_network,
             opt,
             lu,
             ap,
             ExperienceReplay(100000,
                              (typeof(s), Int64, typeof(s), Float32, Bool),
                              (:s, :a, :sp, :r, :t)),
             γ,
             batch_size,
             tn_counter_init,
             tn_counter_init,
             0,
             s,
             Dict{Symbol,Any}())


function RLCore.start!(agent::DQNAgent, env_s_tp1, rng::AbstractRNG; callback=nothing, kwargs...)
    # Start an Episode
    agent.INFO[:training_loss] = 0.0f0
    agent.action = sample(agent.ap,
                          agent.model(env_s_tp1),
                          rng)
    return agent.action
end

function RLCore.step!(agent::DQNAgent, env_s_tp1, r, terminal, rng::AbstractRNG; callback=nothing, kwargs...)
    
    add!(agent.er, (agent.prev_s, agent.action, env_s_tp1, r, terminal))
    
    if size(agent.er)[1] > 1000
        e = sample(agent.er, agent.batch_size; rng=rng)
        update_params!(agent, e)
    end

    if callback != nothing
        callback(agent)
    end
    
    
    agent.prev_s .= env_s_tp1
    agent.action = sample(agent.ap,
                          agent.model(agent.prev_s),
                          rng)

    return agent.action
end

function update_params!(agent::DQNAgent, e)

    if agent.target_network isa Nothing
        update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t)
    else
        agent.INFO[:training_loss] =
            update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t, agent.target_network)

        # Update target network
        if agent.target_network_counter == 1
            agent.target_network_counter = agent.tn_counter_init
            for ps ∈ zip(params(agent.model), params(agent.target_network))
                ps[2] .= ps[1]
            end
        else
            agent.target_network_counter -= 1
        end

    end
    return nothing
end



