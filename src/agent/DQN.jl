
using Flux
using Random
using BSON
using MinimalRLCore


Base.@kwdef mutable struct DQNAgent{M, TN, O, LU, AP<:AbstractValuePolicy, Φ, ER<:AbstractReplay, SB, SP1, SP2, UC<:Val} <: AbstractAgent
    # Models
    model::M
    target_network::TN

    # Learning
    learning_update::LU
    optimizer::O

    # Acting policy (Abstract Value Policy)
    acting_policy::AP

    # State/experience processing.
    replay::ER
    state_buffer::SB
    state_preproc::SP1
    state_postproc::SP2
    prev_s::Φ
    
    # params
    batch_size::Int
    target_update_freq::Int
    update_freq::Int
    min_mem_size::Int

    # minor details
    action::Int = 0
    training_steps::Int = 0

    # extra
    device::UC = Val{:cpu}()
    
    # INFO::Dict{Symbol, Any} = Dict{Symbol, Any}()
end


function DQNAgent(model,
                  target_network,
                  learning_update,
                  optimizer,
                  acting_policy,
                  replay_size,
                  hist_length,
                  example_state,
                  batch_size,
                  target_update_freq,
                  update_freq,
                  min_mem_size;
                  device = Val{:cpu}(),
                  state_preproc = identity,
                  state_postproc = identity,
                  hist_squeeze=false)

    proc_state = state_preproc(example_state)
    
    @assert hist_length >= 1
    state_buffer = if hist_length == 1
        DeepRL.StateBuffer{eltype(proc_state)}(replay_size, length(proc_state))
    else
        DeepRL.HistStateBuffer{eltype(proc_state)}(replay_size, size(proc_state), hist_length, hist_squeeze)
    end
    
    prev_s = if state_buffer isa Nothing
        state_processor(example_state)
    elseif state_buffer isa StateBuffer
        0
    elseif state_buffer isa HistStateBuffer
        zeros(Int, state_buffer.hist_length)
    else
        throw("Unknown StateBuffer please use default DQN constructor.")
    end

    replay = ExperienceReplayDef(replay_size, length(prev_s), eltype(prev_s))
    
    DQNAgent(model = model,
             target_network = target_network,
             learning_update = learning_update,
             optimizer = optimizer,
             acting_policy = acting_policy,
             replay = replay,
             state_buffer = state_buffer,
             state_preproc = state_preproc,
             state_postproc = state_postproc,
             prev_s = prev_s,
             batch_size = batch_size,
             target_update_freq = target_update_freq,
             update_freq = update_freq,
             min_mem_size = min_mem_size,
             device = device)
    
end

function process_state(agent::DQNAgent, s)
    # this stores the state in the state buffer (if not Nothing) and preproccesses the state.
    if agent.state_buffer isa Nothing
        agent.state_preproc(s)
    else
        push!(agent.state_buffer, agent.state_preproc(s))
        laststate(agent.state_buffer)
    end
end

function get_state(agent::DQNAgent, s)
    if agent.state_buffer isa Nothing
        to_device(agent.device, agent.state_postproc(s))
    else
        to_device(agent.device, agent.state_postproc(agent.state_buffer[s]))
    end
end


function MinimalRLCore.start!(agent::DQNAgent,
                       env_s_tp1,
                       rng::AbstractRNG=Random.GLOBAL_RNG)

    agent.prev_s = process_state(agent, env_s_tp1)
    state = get_state(agent, agent.prev_s)
    resh_state = reshape(state, size(state)..., 1)

    agent.action = sample(agent.acting_policy,
                          to_host(agent.model(resh_state)),
                          rng)

    return agent.action
end

function MinimalRLCore.step!(agent::DQNAgent,
                             env_s_tp1,
                             r,
                             terminal,
                             rng::AbstractRNG=Random.GLOBAL_RNG)

    proc_state = process_state(agent, env_s_tp1)

    add_ret = add_exp!(agent.replay,
                       (agent.prev_s,
                        findfirst((a)->a==agent.action,
                                  agent.acting_policy.action_set)::Int,
                        proc_state,
                        Float32(r), # Atari implementation returns float64s for the reward.
                        terminal))

    update_params!(agent, rng)

    agent.prev_s = proc_state
    prev_s = get_state(agent, agent.prev_s)
    resh_prev_s = reshape(prev_s, size(prev_s)..., 1)

    agent.action = sample(agent.acting_policy,
                          to_host(agent.model(resh_prev_s)),
                          rng)

    return agent.action
end

function update_params!(agent::DQNAgent, rng)
    

    if length(agent.replay) > agent.min_mem_size
        if agent.training_steps%agent.update_freq == 0

            e = sample(rng, agent.replay,
                       agent.batch_size)
            
            s = get_state(agent, e.s)
            r = to_device(agent.device, e.r)
            t = to_device(agent.device, e.t)
            sp = get_state(agent, e.sp)

            ℒ = update!(agent.model,
                        agent.learning_update,
                        agent.optimizer,
                        s,
                        e.a,
                        sp,
                        r,
                        t,
                        agent.target_network)
        end
    end
    
    # Target network updates
    if !(agent.target_network isa Nothing)
        if agent.training_steps%agent.target_update_freq == 0
            for ps ∈ zip(params(agent.model),
                         params(agent.target_network))
                copyto!(ps[2], ps[1])
            end
        end
    end

    agent.training_steps += 1

    return nothing    
end


