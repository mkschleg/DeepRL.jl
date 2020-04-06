
using Flux
using Random
using MinimalRLCore


Base.@kwdef struct EvaluateAgent{A, SB, Φ, P}
    agent::A
    prev_s::Φ
    policy::AP
end

function EvaluateAgent(agent::DQNAgent, eval_policy::AP)

    EvaluateAgent(agent,
                  zeros(eltype(agent.state_buffer),
                        size(agent.state_buffer)[1:end-1]...,
                        hist_length(agent.state_buffer)),
                  eval_policy)
end

function MinimalRLCore.start!(agent::EvaluateAgent{DQNAgent},
                              env_s_tp1,
                              rng::AbstractRNG=Random.GLOBAL_RNG)

    agent.prev_s
    
    agent.state_postproc(agent.state_preproc(env_s_tp1))
    
end

function MinimalRLCore.step!(agent::EvaluateAgent{DQNAgent},
                             env_s_tp1,
                             r,
                             terminal,
                             rng::AbstractRNG=Random.GLOBAL_RNG)
end

