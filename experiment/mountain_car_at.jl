

module MountainCarExperiment

using DeepRL
using Flux
using Random
using ProgressMeter
using Plots
using TensorBoardLogger
using Logging
using LinearAlgebra

using RLCore.GVFParamFuncs

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))

function construct_agent(s, num_actions)
    
    ϵ=0.1
    γ=1.0f0
    batch_size=32
    tn_counter_init=50

    gms = 1 .- 2.0f0.^(-7:0)

    gvfs = [[GVF(ScaledCumulant(1/(1-_γ), FeatureCumulant(1)), ConstantDiscount(_γ), NullPolicy()) for _γ in gms];
            # [GVF(ScaledCumulant(1/(1-_γ), FeatureCumulant(1)), ConstantDiscount(_γ), PersistentPolicy(2)) for _γ in gms];
            # [GVF(ScaledCumulant(1/(1-_γ), FeatureCumulant(1)), ConstantDiscount(_γ), PersistentPolicy(3)) for _γ in gms]
            ]
    horde = Horde(gvfs)
    # horde = Horde(GVF[])

    q_stream = Chain(Dense(128, 32, Flux.relu; initW=Flux.glorot_uniform),
                     Dense(32, num_actions; initW=Flux.zeros))

    gvf_stream = Chain(Dense(128, 32, Flux.relu; initW=Flux.glorot_uniform),
                       Dense(32, length(horde); initW=Flux.zeros))
    
    model = Chain(Dense(length(s), 128, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(128, 128, Flux.relu; initW=Flux.glorot_uniform),
                  DeepRL.FluxUtils.ConcatStreams(q_stream, gvf_stream))

    # println(params(model[end]))

    target_network = mapleaves(Flux.Tracker.data, deepcopy(model)::typeof(model))

    return AT_DQNAgent(model,
                       target_network,
                       horde,
                       ADAM(0.001),
                       AuxQLearning(0.01f0, QLearning(γ), TDLearning()),
                       ϵGreedy(ϵ),
                       1000000,
                       γ,
                       batch_size,
                       tn_counter_init,
                       s)
end



function plot_layers(agent::DQNAgent, data_range)


    y = maximum.(Flux.data.(agent.model.(data_range)))

end


function episode!(env, agent, rng, max_steps)
    terminal = false
    s_t = start!(env, rng)
    action = start!(agent, s_t, rng)

    total_rew = 0
    steps = 0

    while !terminal
        
        s_tp1, rew, terminal = step!(env, action)

        action = step!(agent, s_tp1, rew, terminal, rng)
        total_rew += rew
        steps += 1
        
        if steps == max_steps
            break
        end
        
    end
    return total_rew, steps
end


function main_experiment(seed, num_episodes)

    lg=TBLogger("tensorboard_logs/run", min_level=Logging.Info)
    
    mc = MountainCar(0.0, 0.0, true)
    Random.seed!(Random.GLOBAL_RNG, seed)

    s = RLCore.get_state(mc)
    
    agent = construct_agent(s, length(RLCore.get_actions(mc)))

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
    for e in 1:num_episodes
        total_rews[e], steps[e] = episode!(mc, agent, Random.GLOBAL_RNG, 5000)
        next!(p)        
    end

    # end

    return agent.model, total_rews, steps
    
end

end
