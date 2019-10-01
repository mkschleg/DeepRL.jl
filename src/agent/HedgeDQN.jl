
using Flux
using Random


mutable struct HedgeModel{M, F}
    h::M
    f::F
end

function HedgeModel(h::M; init=Flux.glorot_uniform) where {M<:Chain}
    out = size(h[end].W)[1]
    f = [Dense(size(h[1].W)[2], out; init=init)]
    for i in 1:length(h)
        push!(f, Dense(size(h[1.W])[1], out; init=init))
    end
    HedgeModel(model, f)
end



mutable struct HedgeDQNAgent{M, TN, O, LU, AP<:AbstractValuePolicy, Φ, ERP} <: AbstractAgent
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
end

HedgeDQNAgent(model, target_network, opt, lu, ap, size_buffer, γ, batch_size, tn_counter_init, s) =
    HedgeDQNAgent(model,
                  target_network,
                  opt,
                  lu,
                  ap,
                  ExperienceReplay(100000,
                                   (Array{Float32, 1}, Int64, Array{Float32, 1}, Float32, Bool),
                                   (:s, :a, :sp, :r, :t)),
                  γ,
                  batch_size,
                  tn_counter_init,
                  tn_counter_init,
                  0,
                  s)


function JuliaRL.start!(agent::HedgeDQNAgent, env_s_tp1, rng::AbstractRNG; kwargs...)
    # Start an Episode
    agent.action = sample(agent.ap,
                          agent.model(env_s_tp1),
                          rng)
    return agent.action
end

function JuliaRL.step!(agent::HedgeDQNAgent, env_s_tp1, r, terminal, rng::AbstractRNG; kwargs...)
    
    add!(agent.er, (Float32.(agent.prev_s), agent.action, copy(Float32.(env_s_tp1)), r, terminal))
    
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

function update_params!(agent::HedgeDQNAgent, e)

    if agent.tn_counter_init > 0
        update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t, agent.target_network)

        if agent.target_network_counter == 1
            agent.target_network_counter = agent.tn_counter_init
            agent.target_network = mapleaves(
                Flux.Tracker.data,
                deepcopy(agent.model))
        else
            agent.target_network_counter -= 1
        end
    else
        update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t)
    end
    return nothing
    
end
