
using Flux
using Random

mutable struct ImageDQNAgent{M, TN, O, LU, AP<:AbstractValuePolicy, Φ, ER<:AbstractImageReplay} <: AbstractAgent
    model::M
    target_network::TN
    opt::O
    lu::LU
    ap::AP
    er::ER
    batch_size::Int
    tn_counter_init::Int
    target_network_counter::Int
    wait_time::Int
    wait_time_counter::Int
    action::Int
    prev_s::Φ
end

ImageDQNAgent(model, target_network, image_replay, opt, lu, ap, size_buffer, batch_size, tn_counter_init, wait_time) =
    ImageDQNAgent(model,
                  target_network,
                  opt,
                  lu,
                  ap,
                  image_replay,
                  batch_size,
                  tn_counter_init,
                  tn_counter_init,
                  wait_time,
                  0,
                  0,
                  zeros(Int, 1))


function RLCore.start!(agent::ImageDQNAgent, env_s_tp1, rng::AbstractRNG; kwargs...)
    # Start an Episode
    agent.prev_s = copy(add!(agent.er, env_s_tp1))
    # @show size(view(agent.er.image_buffer, agent.prev_s))
    agent.action = sample(agent.ap,
                          agent.model(gpu(reshape(getindex(agent.er.image_buffer, agent.prev_s), (84,84,4,1)))),
                          rng)

    return agent.action
end

function RLCore.step!(agent::ImageDQNAgent, env_s_tp1, r, terminal, rng::AbstractRNG; kwargs...)

    # cur_s = add!(agent.er, env_s_tp1, findfirst((a)->a==agent.action, agent.ap.action_set), r, terminal)
    cur_s = add!(agent.er, env_s_tp1, agent.action + 1, r, terminal)

    agent.wait_time_counter -= 1
    if size(agent.er)[1] > 50
        e = sample(agent.er, agent.batch_size; rng=rng)
        update_params!(agent, e)
        agent.wait_time_counter = agent.wait_time
    end

    
    agent.prev_s .= cur_s
    agent.action = sample(agent.ap,
                          agent.model(gpu(reshape(getindex(agent.er.image_buffer, agent.prev_s), (84,84,4,1)))),
                          rng)

    return agent.action
end

function update_params!(agent::ImageDQNAgent, e)

    if agent.tn_counter_init > 0 
        
        update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t, agent.target_network)
        
        if agent.target_network_counter == 1
            agent.target_network_counter = agent.tn_counter_init
            agent.target_network = deepcopy(mapleaves(
                Flux.Tracker.data,
                agent.model))
        else
            agent.target_network_counter -= 1
        end

    else
        update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t)
        agent.wait_time_counter = agent.wait_time
    end
    return nothing
    
end
