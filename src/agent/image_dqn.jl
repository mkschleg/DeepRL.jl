
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
    prev_s_idx::Φ
    prev_s::CuArray{Float32, 4}
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
                  zeros(Int, image_replay.hist),
                  gpu(zeros(Float32,
                            image_replay.image_buffer.img_size...,
                            image_replay.hist,
                            1)))


function RLCore.start!(agent::ImageDQNAgent, env_s_tp1, rng::AbstractRNG; kwargs...)
    # Start an Episode
    agent.prev_s_idx .= add!(agent.er, env_s_tp1)

    copyto!(agent.prev_s[:,:,:,1], getindex(agent.er.image_buffer, agent.prev_s_idx)./256f0)

    agent.action = sample(agent.ap,
                          cpu(agent.model(CuArray(agent.prev_s))),
                          rng)

    return agent.action
end

function RLCore.step!(agent::ImageDQNAgent, env_s_tp1, r, terminal, rng::AbstractRNG; kwargs...)

    cur_s = add!(agent.er, env_s_tp1, findfirst((a)->a==agent.action, agent.ap.action_set), r, terminal)

    agent.wait_time_counter -= 1
    if size(agent.er)[1] > 50000 && agent.wait_time_counter == 0
        update_params!(agent,
                       sample(agent.er, agent.batch_size; rng=rng))
        agent.wait_time_counter = agent.wait_time
    end

    agent.prev_s_idx .= cur_s

    copyto!(agent.prev_s[:,:,:,1], getindex(agent.er.image_buffer, agent.prev_s_idx)./256f0)
    
    agent.action = sample(agent.ap,
                          cpu(agent.model(CuArray(agent.prev_s))),
                          rng)

    return agent.action
end

function update_params!(agent::ImageDQNAgent, e)

    s = CuArray(e.s)
    r = CuArray(e.r)
    t = CuArray(e.t)
    sp = CuArray(e.sp)

    update!(agent.model, agent.lu, agent.opt, s, e.a, sp, r, t, agent.target_network)

    if !(agent.target_network isa Nothing)
        if agent.target_network_counter == 1
            agent.target_network_counter = agent.tn_counter_init
            for ps ∈ zip(collect(params(agent.model)),
                         collect(params(agent.target_network)))
                ps[2] .= ps[1]
            end
        else
            agent.target_network_counter -= 1
        end
    end

    return nothing
    
end
