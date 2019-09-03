

module MountainCarExperiment

using DeepRL
using Flux
using Random
using ProgressMeter

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


function construct_agent(s, num_actions)
    
    ϵ=0.1
    γ=1.0f0
    batch_size=32
    tn_counter_init = 50

    model = Chain(Dense(length(s), 128, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(128, 128, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(128, 32, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(32, num_actions; initW=Flux.glorot_uniform))

    target_network = mapleaves(Flux.Tracker.data, deepcopy(model)::typeof(model))

    return DQNAgent(model,
                    target_network,
                    ADAM(0.001),
                    QLearning(γ),
                    ϵGreedy(ϵ),
                    1000000,
                    γ,
                    batch_size,
                    tn_counter_init,
                    s)
end


function episode!(env, agent, rng, max_steps)
    terminal = false
    s_t = start!(env, rng)
    action = start!(agent, s_t, rng)

    total_rew = 0
    steps = 0

    while !terminal
        
        s_tp1, rew, terminal = step!(env, action)
        if steps == max_steps
            terminal = true
        end
        action = step!(agent, s_tp1, rew, terminal, rng)
        total_rew += rew
        steps += 1
    end
    return total_rew, steps
end


function main_experiment(seed, num_episodes)

    mc = MountainCar(0.0, 0.0, true)
    Random.seed!(Random.GLOBAL_RNG, seed)

    s = JuliaRL.get_state(mc)
    
    agent = construct_agent(s, length(JuliaRL.get_actions(mc)))::DQNAgent

    total_rews = zeros(num_episodes)
    steps = zeros(Int64, num_episodes)

    
    front = ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇']
    p = ProgressMeter.Progress(
        num_episodes;
        dt=0.01,
        desc="Episode:",
        barglyphs=ProgressMeter.BarGlyphs('|','█',front,' ','|'),
        barlen=Int64(floor(500/length(front))))
    
    for e in 1:num_episodes
        total_rews[e], steps[e] = episode!(mc, agent, Random.GLOBAL_RNG, 50000)
        next!(p)
    end

    return total_rews, steps
    
end

end

