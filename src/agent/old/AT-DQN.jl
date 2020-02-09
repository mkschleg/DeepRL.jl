


using Flux
using Random

mutable struct AT_DQNAgent{M, H<:Horde, TN, O, LU, AP<:AbstractValuePolicy, Φ, ERP} <: AbstractAgent
    model::M
    target_network::TN
    horde::H
    opt::O
    lu::LU
    ap::AP
    er::ExperienceReplay{ERP}
    γ::Float32
    batch_size::Int64
    tn_counter_init::Int64
    target_network_counter::Int64
    action::Int64
    μ_prob::Float64
    prev_s::Φ
end

AT_DQNAgent(model, target_network, horde, opt, lu, ap, size_buffer, γ, batch_size, tn_counter_init, s) =
    AT_DQNAgent(model,
                target_network,
                horde,
                opt,
                lu,
                ap,
                ExperienceReplay(100000,
                                 (Array{Float32, 1}, Int64, Array{Float32, 1}, Float32, Bool, Float32),
                                 (:s, :a, :sp, :r, :t, :pi)),
                γ,
                batch_size,
                tn_counter_init,
                tn_counter_init,
                0,
                0.0,
                s)


function RLCore.start!(agent::AT_DQNAgent, env_s_tp1, rng::AbstractRNG; kwargs...)
    # Start an Episode

    values = agent.model(env_s_tp1)[1:(end-length(agent.horde))]
    agent.action = sample(agent.ap,
                          values,
                          rng)
    agent.μ_prob = get_prob(agent.ap, values, agent.action)
    agent.prev_s .= env_s_tp1
    
    return agent.action
end

function RLCore.step!(agent::AT_DQNAgent, env_s_tp1, r, terminal, rng::AbstractRNG; kwargs...)
    
    add!(agent.er, (copy(Float32.(agent.prev_s)), agent.action, copy(Float32.(env_s_tp1)), r, terminal, 1.0f0))
    
    if size(agent.er)[1] > 1000
        e = sample(agent.er, agent.batch_size; rng=rng)
        update_params!(agent, e)
    end
    
    
    agent.prev_s .= env_s_tp1
    
    values = agent.model(agent.prev_s)[1:(end-length(agent.horde))]
    # println(values)
    agent.action = sample(agent.ap,
                          values,
                          rng)
    agent.μ_prob = get_prob(agent.ap, values, agent.action)

    return agent.action
end

function update_params!(agent::AT_DQNAgent, e)

    if agent.tn_counter_init > 0
        update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t, agent.target_network, agent.horde)

        if agent.target_network_counter == 1
            agent.target_network_counter = agent.tn_counter_init
            agent.target_network = mapleaves(
                Flux.Tracker.data,
                deepcopy(agent.model))
        else
            agent.target_network_counter -= 1
        end
    else
        update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t, nothing, agent.horde)
    end
    return nothing
end

