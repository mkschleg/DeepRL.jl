
module MountainCarExperiment

using DeepRL
using Flux
using Random
using ProgressMeter
using TensorBoardLogger
using Logging
using Statistics

import DeepRL: Macros
import .Macros: @generate_config_funcs, @generate_working_function, @generate_ann_size_helper


using MinimalRLCore

@generate_config_funcs begin
    info"""
    Experiment details.
    --------------------
    - `seed::Int`: seed of RNG
    - `steps::Int`: Number of steps taken in the experiment
    """
    seed => 1
    steps => 300000
end

function build_ann(config, in, actions, rng)
    
    NN_config = [
        Dict("type"=>"Dense", "out"=>64, "bias"=>true, "activation"=>"relu"),
        Dict("type"=>"Dense", "out"=>64, "bias"=>true, "activation"=>"relu"),
        Dict("type"=>"Dense", "out"=>"actions", "bias"=>true, "activation"=>"linear")
    ]

    model = DeepRL.build_ann_from_config(
        rng,
        (in,),
        NN_config,
        init=Flux.glorot_uniform(rng),
        actions=actions)
end

function construct_agent(env, config, rng=Random.default_rng())

    ϵ=0.1
    γ=1.0f0
    batch_size=32
    
    tn_update_freq=100
    update_freq=1
    min_mem_size=1000
    er_size = 10000
    hist = 1

    num_actions = length(get_actions(env))
    s = MinimalRLCore.get_state(env)

    model = build_ann(config, length(s), num_actions, rng)
    
    target_network = deepcopy(model)

    return DQNAgent(
        model,
        target_network,
        QLearning(γ, Flux.mse),
        DeepRL.RMSPropTFCentered(0.001),
        ϵGreedy(ϵ, num_actions),
        er_size,
        hist,
        s,
        batch_size,
        tn_update_freq,
        update_freq,
        min_mem_size,
        hist_squeeze = Val{true}()
    )
end


"""
    construct_env

Construct direction tmaze using:
- `size::Int` size of hallway.
"""
function construct_env(config, args...)
    mc = MountainCar(0.0, 0.0, true)
end

Macros.@generate_ann_size_helper
Macros.@generate_working_function

"""
    main_experiment

Run an experiment from config. See [`MountainCarExperiment.working_experiment`](@ref) 
for details on running on the command line and [`DirectionalTMazeERExperiment.default_config`](@ref) 
for info about the default configuration.
"""
function main_experiment(config;
                         progress=false,
                         testing=false,
                         overwrite=false)

    seed = config["seed"]
    max_num_steps = config["steps"]
    
    mc = MountainCar(0.0, 0.0, true)

    Random.seed!(seed)
    rng = Random.default_rng()

    s = MinimalRLCore.get_state(mc)

    agent = construct_agent(mc, config, rng) #construct_agent(s, length(MinimalRLCore.get_actions(mc)))

    total_rews = Float32[]
    steps = Int[]

    max_episode_length = 5000
    
    front = ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇']
    p = ProgressMeter.Progress(
        max_num_steps;
        dt=0.1,
        desc="Step:",
        barglyphs=ProgressMeter.BarGlyphs('|','█',front,' ','|'))

    eps = 1
    while sum(steps) < max_num_steps
        cur_step = 0
        episode_cut_off = min(max_episode_length, max_num_steps - sum(steps))
        tr, stp =
            run_episode!(mc, agent, episode_cut_off, rng) do (s, a, s′, r)
                cur_step+=1
                # next!(p, showvalues=[(:step, sum(steps)+cur_step), (:episode, eps)])
                next!(p)
            end
        push!(total_rews, tr)
        push!(steps, stp)
        
        eps += 1
    end

    return agent.model, total_rews, steps
    
end

end

