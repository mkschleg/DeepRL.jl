module MountainCarOnlineEEGExperiment

using DeepRL
using Flux
using Random
using ProgressMeter
using Plots
using TensorBoardLogger
using Logging
using LinearAlgebra

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


function construct_agent(s, num_actions, tn_counter_init, α)
    
    ϵ=0.1
    γ=1.0f0
    batch_size=32
    # tn_counter_init=50

    model = Chain(Dense(length(s), 128, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(128, 128, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(128, 32, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(32, num_actions; initW=Flux.glorot_uniform))

    target_network = mapleaves(Flux.Tracker.data, deepcopy(model)::typeof(model))

    return OnlineDQNAgent(model,
                          target_network,
                          ADAM(α),
                          QLearning(γ),
                          ϵGreedy(ϵ),
                          γ,
                          batch_size,
                          tn_counter_init,
                          s)
end


abstract type AbstractFluxWidget end

mutable struct EEG{T<:AbstractFloat} <: AbstractFluxWidget
    activations::Array{Array{Array{T}, 1}, 1}
end

EEG(type::Type{T}) where {T<:AbstractFloat}  = EEG(Array{Array{Array{T}, 1}, 1}())

function update!(eeg::EEG, model, input)
    push!(eeg.activations, DeepRL.FluxUtils.get_activations(mapleaves(Flux.data, model), input))
end

function episode!(env, agent, rng, max_episode_steps, max_steps, cur_steps, widgets::Array{<:AbstractFluxWidget, 1})
    terminal = false
    s_t = start!(env, rng)
    action = start!(agent, s_t, rng)

    total_rew = 0
    steps = 1

    for w in widgets
        update!(w, agent.model, s_t)
    end

    while !terminal
        
        s_tp1, rew, terminal = step!(env, action)
        if steps == max_episode_steps || steps+cur_steps == max_steps
            terminal = true
        end
        action = step!(agent, s_tp1, rew, terminal, rng)
        total_rew += rew
        steps += 1
        for w in widgets
            update!(w, agent.model, s_tp1)
        end
    end
    return total_rew, steps
end


function main_experiment(seed, num_episodes; α=0.001, tn_counter=50, max_steps=50000)

    lg=TBLogger("tensorboard_logs/run", min_level=Logging.Info)
    
    mc = MountainCar(0.0, 0.0, true)
    Random.seed!(Random.GLOBAL_RNG, seed)

    s = JuliaRL.get_state(mc)
    
    agent = construct_agent(s, length(JuliaRL.get_actions(mc)), tn_counter, α)::OnlineDQNAgent

    total_rews = zeros(num_episodes)
    steps = zeros(Int64, num_episodes)

    
    front = ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇']
    p = ProgressMeter.Progress(
        num_episodes;
        dt=0.01,
        desc="Episode:",
        barglyphs=ProgressMeter.BarGlyphs('|','█',front,' ','|'),
        barlen=Int64(floor(500/length(front))))

    data_range = collect.(collect(Iterators.product(-1.0:0.01:1.0, -1.0:0.01:1.0)))
    # with_logger(lg) do

    eeg = EEG(Float32)
    
    for e in 1:num_episodes
        total_rews[e], steps[e] = episode!(mc, agent, Random.GLOBAL_RNG, 10000, max_steps, sum(steps), [eeg])
        if sum(steps) >= max_steps
            total_rews = total_rews[1:e]
            steps = steps[1:e]
            break;
        end
        next!(p)        
    end

    # end

    return agent.model, total_rews, steps, eeg
    
end

end
