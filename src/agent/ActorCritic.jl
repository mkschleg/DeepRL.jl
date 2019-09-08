using Flux
using Random
using Distributions

mutable struct ActorCriticNetwork{M, Π, V, D}
    base::M
    π::Π
    v::V
    dist::D
end



mutable struct ActorCriticAgent{M, TN, O, LU, AP<:AbstractValuePolicy, Φ, ERP} <: AbstractAgent
    network::M
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

function ActorCriticAgent(model, target_network, opt, lu, ap, size_buffer, γ, batch_size, tn_counter_init, s)

    model = ActorCriticNetwork(
        Chain(Dense(2, 128, Flux.relu; initW=Flux.glorot_uniform)
              Dense(128, 128, Flux.relu; initW=Flux.glorot_uniform),
              Dense(128, 32, Flux.relu; initW=Flux.glorot_uniform)),
        [Dense(32, 1), Dense(32, 1, Flux.relu)],
        Dense(32, 1),
        Normal)
    # model = Chain(Dense(length(s), 128, Flux.relu; initW=Flux.glorot_uniform),
    #               Dense(128, 128, Flux.relu; initW=Flux.glorot_uniform),
    #               Dense(128, 32, Flux.relu; initW=Flux.glorot_uniform),
    #               Dense(32, num_actions; initW=Flux.glorot_uniform))

    ActorCriticAgent(model,
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
end


function JuliaRL.start!(agent::ActorCriticAgent, env_s_tp1, rng::AbstractRNG=Random.GLOBAL_RNG; kwargs...)
    # Start an Episode
    agent.action = sample(agent.ap,
                          agent.model(env_s_tp1),
                          rng)
    return agent.action
end

function JuliaRL.step!(agent::ActorCriticAgent, env_s_tp1, r, terminal, rng=Random.GLOBAL_RNG; kwargs...)
    
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

function update_params!(agent::ActorCriticAgent, e)
    
    # update!(agent.model, agent.lu, agent.opt, e.s, e.a, e.sp, e.r, e.t, agent.target_network)

    s_t = hcat(e.s...)
    s_tp1 = hcat(e.sp...)
    

    b_tp1 = Flux.data(base(s_tp1))
    v_tp1 = Flux.data(agent.model.v(b_tp1))
    target = e.r .+ agent.γ.*(.!e.t).*v_tp1
    
    # update value function

    gs = Flux.Tracker.Gradient(params(agent.model.base, agent.model.π, agent.model.v)) do
        b_t = base(s_t)
        δ = target .- agent.model.v(b_t)
        μ, σ = agent.model.π[1](b_t)
    end

    # update policy


    if agent.target_network_counter == 1
        agent.target_network_counter = agent.tn_counter_init
        agent.target_network = mapleaves(
            Flux.Tracker.data,
            deepcopy(agent.model))
    else
        agent.target_network_counter -= 1
    end

    return nothing
    
end
