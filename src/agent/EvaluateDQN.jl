

Base.@kwdef mutable struct EvaluateDQNAgent{M, AP<:AbstractValuePolicy, Φ, IR<:HistImageReplay} <: AbstractAgent
    model::M
    ap::AP
    replay::IR
    prev_s::Φ
    hist::Int
    action::Int = 0
    INFO::Dict{Symbol, Any} = Dict{Symbol, Any}()
end


EvaluateDQNAgent(model, ϵ, hist, action_set, img_manip, img_norm, img_size) =
    EvaluateDQNAgent(
        model = model,
        ap = ϵGreedy(0.1, action_set),
        replay = HistImageReplay(1, img_size, img_manip, img_norm, hist, 1),
        prev_s = zeros(Int, hist),
        hist = hist
    )


function RLCore.start!(agent::EvaluateDQNAgent,
                       env_s_tp1,
                       rng::AbstractRNG;
                       kwargs...)
    
    agent.prev_s .= add!(agent.replay, env_s_tp1)
    state = agent.replay.img_norm(
        reshape(getindex(agent.replay.image_buffer, agent.prev_s),
                agent.replay.image_buffer.img_size..., agent.replay.hist,
                1)) 
    agent.action = sample(agent.ap,
                          agent.model(state),
                          rng)#::eltype(agent.ap)
    return agent.action
end

function RLCore.step!(agent::EvaluateDQNAgent,
                      env_s_tp1,
                      r,
                      terminal,
                      rng::AbstractRNG;
                      kwargs...)

    add_ret = add!(agent.replay,
                   (agent.prev_s,
                    findfirst((a)->a==agent.action,
                              agent.ap.action_set)::Int,
                    env_s_tp1,
                    r,
                    terminal))
    agent.prev_s .= add_ret
    prev_s = 
        agent.replay.img_norm(
            reshape(getindex(agent.replay.image_buffer, agent.prev_s),
                    agent.replay.image_buffer.img_size..., agent.replay.hist,
                    1))

    agent.action = sample(agent.ap,
                          agent.model(prev_s),
                          rng)
    return agent.action
end
