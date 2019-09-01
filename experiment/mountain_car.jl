

module MountainCarExperiment

using DeepRL
using Flux
using Random
using ProgressMeter

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


function get_action(agent::DQNAgent, state, rng::AbstractRNG)
    if Float32.(state) == agent.prev_s
        return agent.action
    end

    return sample(agent.ap,
                  Flux.data(agent.model(state)),
                  rng)
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


function construct_agent(s, num_actions)
    
    ϵ=0.1
    γ=1.0f0
    batch_size=32
    tn_counter_init = 50

    # model = Chain(Dense(length(s), 128, Flux.relu; initW=(dims...)->glorot_uniform(rng, dims...)),
    #               Dense(128, 128, Flux.relu; initW=(dims...)->glorot_uniform(rng, dims...)),
    #               Dense(128, 32, Flux.relu; initW=(dims...)->glorot_uniform(rng, dims...)),
    #               Dense(32, length(JuliaRL.get_actions(mc)); initW=(dims...)->glorot_uniform(rng, dims...)))

    model = Chain(Dense(length(s), 128, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(128, 128, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(128, 32, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(32, num_actions; initW=Flux.glorot_uniform))

    target_network = mapleaves(Flux.Tracker.data, deepcopy(model)::typeof(model))
    # target_network = deepcopy(model)
    
    opt = ADAM(0.001)

    er = ExperienceReplay(100000,
                          (Array{Float32, 1}, Int64, Array{Float32, 1}, Float32, Bool),
                          (:s, :a, :sp, :r, :t))

    return DQNAgent(model,
                    target_network,
                    opt,
                    ϵGreedy(ϵ),
                    er,
                    γ,
                    batch_size,
                    tn_counter_init,
                    tn_counter_init,
                    -1,
                    s)
end



function main_experiment(seed, num_episodes)

    mc = MountainCar(0.0, 0.0, true)
    # rng = MersenneTwister(seed)
    Random.seed!(Random.GLOBAL_RNG, seed)

    s = JuliaRL.get_state(mc)
    
    agent = construct_agent(s, length(JuliaRL.get_actions(mc)))::DQNAgent

    total_rews = zeros(num_episodes)
    steps = zeros(Int64, num_episodes)

    
    front = ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇']
    p = ProgressMeter.Progress(num_episodes; dt=0.01, desc="Episode:", barglyphs=ProgressMeter.BarGlyphs('|','█',front,' ','|'), barlen=Int64(floor(500/length(front))))
    
    for e in 1:num_episodes
        total_rews[e], steps[e] = episode!(mc, agent, Random.GLOBAL_RNG, 50000)
        next!(p)
    end

    return total_rews, steps
    
end

end

