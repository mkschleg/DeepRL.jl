
using Flux
using Random
using BSON
using MinimalRLCore

Base.@kwdef mutable struct DQNAgent{M, TN, O, LU, AP<:AbstractValuePolicy, Φ, ER<:AbstractReplay} <: AbstractAgent
    # Models
    model::M
    target_network::TN

    # Learning
    lu::LU
    opt::O
    replay::ER
    
    # Acting policy (Abstract Value Policy)
    ap::AP

    # extra
    prev_s::Φ
    
    # params
    batch_size::Int = 32
    target_update_freq::Int = 10000
    update_freq::Int = 4
    min_mem_size::Int = 10000
    
    action::Int = 0
    training_steps::Int = 0
    INFO::Dict{Symbol, Any} = Dict{Symbol, Any}()
end

const ImageDQNAgent =
    DQNAgent{M, TN, O, LU, AP, Φ, ER} where {M, TN, O, LU, AP<:AbstractValuePolicy, Φ, ER<:AbstractImageReplay}

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
                         replay = replay,
                         ap = acting_policy,
                         prev_s = zeros(Φ, feature_size),
                         batch_size = batch_size,
                         target_update_freq = target_update_freq,
                         update_freq = update_freq,
                         min_mem_size = min_mem_size)

DQNAgent(model, target_network, optimizer, learning_update,
         acting_policy, replay, feature_size,
         batch_size, target_update_freq, update_freq,
         min_mem_size) =
             DQNAgent{Float32}(model,
                               target_network,
                               optimizer,
                               learning_update,
                               acting_policy,
                               replay,
                               feature_size,
                               batch_size,
                               target_update_freq,
                               update_freq,
                               min_mem_size)


get_state(agent::DQNAgent, s) = s
get_state(agent::ImageDQNAgent, s) =
    agent.replay.img_norm(
        reshape(getindex(agent.replay.image_buffer, s),
                agent.replay.image_buffer.img_size..., agent.replay.hist,
                1))

warmup_replay(agent::DQNAgent, env_s_tp1) = env_s_tp1
warmup_replay(agent::ImageDQNAgent, env_s_tp1) = add!(agent.replay, env_s_tp1)

function MinimalRLCore.start!(agent::DQNAgent,
                       env_s_tp1,
                       rng::AbstractRNG=Random.GLOBAL_RNG)

    agent.prev_s = warmup(agent.replay, env_s_tp1)
    
    state = get_state(agent, agent.prev_s) #|> gpu

    agent.action = sample(agent.ap,
                          agent.model(state),
                          rng)

    return agent.action
end

function MinimalRLCore.step!(agent::DQNAgent,
                             env_s_tp1,
                             r,
                             terminal,
                             rng::AbstractRNG=Random.GLOBAL_RNG)

    
    add_ret = add!(agent.replay,
                   (agent.prev_s,
                    findfirst((a)->a==agent.action,
                              agent.ap.action_set)::Int,
                    env_s_tp1,
                    r,
                    terminal))

    update_params!(agent, rng)

    agent.prev_s .= if agent isa ImageDQNAgent
        add_ret
    else
        env_s_tp1
    end

    prev_s = get_state(agent, agent.prev_s) |> gpu

    agent.action = sample(agent.ap,
                          agent.model(prev_s),
                          rng)

    return agent.action
end

function update_params!(agent::DQNAgent, rng)
    

    if size(agent.replay)[1] > agent.min_mem_size
        if agent.training_steps%agent.update_freq == 0

            e = sample(agent.replay,
                       agent.batch_size;
                       rng=rng)
            s = gpu(e.s)
            r = gpu(e.r)
            t = gpu(e.t)
            sp = gpu(e.sp)
            
            ℒ = update!(agent.model,
                        agent.lu,
                        agent.opt,
                        s,
                        e.a,
                        sp,
                        r,
                        t,
                        agent.target_network)
            # agent.INFO[:loss] = ℒ
        end
    end
    
    # Target network updates
    if !(agent.target_network isa Nothing)
        if agent.training_steps%agent.target_update_freq == 0
            for ps ∈ zip(params(agent.model),
                         params(agent.target_network))
                ps[2] .= ps[1]
            end
        end
    end

    agent.training_steps += 1

    return nothing    
end


