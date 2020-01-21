
using Flux
using Random




Base.@kwdef mutable struct DQNAgent{M, TN, O, LU, AP<:AbstractValuePolicy, Φ, ER<:AbstractReplay} <: AbstractAgent
    model::M
    target_network::TN
    lu::LU
    opt::O
    ap::AP
    replay::ER
    prev_s::Φ
    batch_size::Int = 32
    target_update_freq::Int = 10000
    target_network_counter::Int = 0
    update_freq::Int = 4
    update_freq_counter::Int = 0
    min_mem_size::Int = 10000
    action::Int = 0
    INFO::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

# const ImageDQNAgent = DQNAgent{M, TN, O, LU, AP, Φ, ER<:AbstractImageReplay}

DQNAgent{Φ}(model,
            target_network,
            optimizer,
            learning_update,
            acting_policy,
            replay,
            feature_size,
            batch_size,
            target_update_freq,
            update_freq,
            min_mem_size) where {Φ <: Number} =
                DQNAgent(model = model,
                         target_network = target_network,
                         lu = learning_update,
                         opt = optimizer,
                         ap = acting_policy,
                         replay = replay,
                         prev_s = zeros(Φ, feature_size),
                         batch_size = batch_size,
                         target_update_freq = target_update_freq,
                         target_network_counter = target_update_freq,
                         update_freq = update_freq,
                         min_mem_size = min_mem_size)

DQNAgent(model, target_network, optimizer, learning_update,
         acting_policy, replay, feature_size,
         batch_size, target_update_freq, update_freq,
         min_mem_size) =
             DQNAgent{Float64}(model, target_network, optimizer, learning_update,
                               acting_policy, replay, feature_size,
                               batch_size, target_update_freq, update_freq,
                               min_mem_size)




function RLCore.start!(agent::DQNAgent,
                       env_s_tp1,
                       rng::AbstractRNG;
                       kwargs...)
    
    agent.INFO[:training_loss] = 0.0f0
    # Start an Episode
    agent.prev_s .= if agent.replay isa AbstractImageReplay
        add!(agent.replay, env_s_tp1)
    else
        env_s_tp1
    end

    prev_s = if agent.replay isa AbstractImageReplay
        cat(getindex(agent.replay.image_buffer, agent.prev_s_idx)./256f0;
            dims=4) |> gpu
    else
        agent.prev_s |> gpu
    end

    agent.action = sample(agent.ap,
                          agent.model(prev_s),
                          rng)

    return agent.action
end

function RLCore.step!(agent::DQNAgent,
                      env_s_tp1,
                      r,
                      terminal,
                      rng::AbstractRNG;
                      kwargs...)

    
    add_ret = add!(agent.replay,
                   (agent.prev_s,
                    findfirst((a)->a==agent.action,
                              agent.ap.action_set),
                    env_s_tp1,
                    r,
                    terminal))

    cur_s = if agent.replay isa AbstractImageReplay
        add_ret
    else
        env_s_tp1
    end
    
    update_params!(agent, rng)

    agent.prev_s .= cur_s

    prev_s = if agent.replay isa AbstractImageReplay
        cat(getindex(agent.replay.image_buffer, agent.prev_s_idx)./256f0;
            dims=4) |> gpu
    else
        agent.prev_s |> gpu
    end

    agent.action = sample(agent.ap,
                          agent.model(prev_s),
                          rng)

    return agent.action
end

function update_params!(agent::DQNAgent, rng)
    

    if size(agent.replay)[1] > agent.min_mem_size
        agent.update_freq_counter -= 1
        if agent.update_freq_counter <= 0

            e = sample(agent.replay,
                       agent.batch_size;
                       rng=rng)
            s = gpu(e.s)
            r = gpu(e.r)
            t = gpu(e.t)
            sp = gpu(e.sp)
            
            agent.INFO[:training_loss] = update!(agent.model,
                                                 agent.lu,
                                                 agent.opt,
                                                 s,
                                                 e.a,
                                                 sp,
                                                 r,
                                                 t,
                                                 agent.target_network)

            agent.update_freq_counter = agent.update_freq
        end
    end
    
    # Target network updates 
    if !(agent.target_network isa Nothing)
        agent.target_network_counter -= 1
        if agent.target_network_counter <= 0
            agent.target_network_counter = agent.target_update_freq
            for ps ∈ zip(params(agent.model),
                         params(agent.target_network))
                ps[2] .= ps[1]
            end
        end
    end

    return nothing    
end
