
using Flux
using Random

mutable struct ImageDQNAgent{M, TN, O, LU, AP<:AbstractValuePolicy, Φ, ERP, IMB} <: AbstractAgent
    model::M
    target_network::TN
    opt::O
    lu::LU
    ap::AP
    er::ExperienceReplay{ERP}
    image_buffer::IMB
    γ::Float32
    batch_size::Int64
    tn_counter_init::Int64
    target_network_counter::Int64
    action::Int64
    prev_s::Φ
end

ImageDQNAgent(model, target_network, opt, lu, ap, size_buffer, im, γ, batch_size, tn_counter_init, s) =
    ImageDQNAgent(model,
             target_network,
             opt,
             lu,
             ap,
             ExperienceReplay(size_buffer,
                              (Int64, Int64, Int64, Float32, Bool),
                              (:s, :a, :sp, :r, :t)),
             ImageBuffer(size_buffer+2, im, Array{Float32, 3}),
             γ,
             batch_size,
             tn_counter_init,
             tn_counter_init,
             0,
             0)


function JuliaRL.start!(agent::ImageDQNAgent, env_s_tp1, rng::AbstractRNG; kwargs...)
    # Start an Episode
    agent.action = sample(agent.ap,
                          agent.model(env_s_tp1),
                          rng)
    agent.prev_s = add!(agent.imb, env_s_tp1)
    return agent.action
end

function JuliaRL.step!(agent::ImageDQNAgent, env_s_tp1, r, terminal, rng::AbstractRNG; kwargs...)

    img_idx = add!(agent.imb, env_s_tp1)
    
    add!(agent.er, (agent.prev_s, agent.action, img_idx, r, terminal))
    
    if size(agent.er)[1] > 1000
        e = sample(agent.er, agent.batch_size; rng=rng)
        update_params!(agent, e)
    end

    
    agent.prev_s .= env_s_tp1
    agent.action = sample(agent.ap,
                          agent.model(agent.prev_s),
                          rng)

    return agent.action
end

function update_params!(agent::ImageDQNAgent, e)

    if agent.tn_counter_init > 0
        
        update!(agent.model, agent.lu, agent.opt, agent.image_buffer[e.s], e.a, agent.image_buffer[e.sp], e.r, e.t, agent.target_network)

        if agent.target_network_counter == 1
            agent.target_network_counter = agent.tn_counter_init
            agent.target_network = deepcopy(mapleaves(
                Flux.Tracker.data,
                agent.model))
        else
            agent.target_network_counter -= 1
        end
    else
        update!(agent.model, agent.lu, agent.opt, agent.image_buffer[e.s], e.a, agent.image_buffer[e.sp], e.r, e.t)
    end
    return nothing
    
end
