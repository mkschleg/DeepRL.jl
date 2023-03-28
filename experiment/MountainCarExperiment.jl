
module MountainCarExperiment

using DeepRL
using Flux
using Random
using ProgressMeter
using TensorBoardLogger
using Logging
using Statistics

import ChoosyDataLoggers

ChoosyDataLoggers.@init
function __init__()
    ChoosyDataLoggers.@register
end

import Reproduce: experiment_wrapper

import DeepRL: Macros
import .Macros:
    @generate_config_funcs,
    @generate_working_function,
    @generate_ann_size_helper,
    @config_to_param


using MinimalRLCore

@generate_config_funcs begin
    info"""
    Experiment details.
    --------------------
    - `seed::Int`: seed of RNG
    - `steps::Int`: Number of steps taken in the experiment
    """
    seed => 1
    steps => 200000

    info"""
    ### Logging Extras
    
    By default the experiment will log and save (depending on the synopsis flag) the logging group `:EXP`. 
    You can add extra logging groups and [group, name] pairs using the below arguments. Everything 
    added to `save_extras` will be passed to the save operation, and will be logged automatically. The 
    groups and names added to `log_extras` will be ommited from save_results but still passed back to the user
    through the data dict.

    - `<log_extras::Vector{Union{String, Vector{String}}>`: which group and <name> to log to the data dict. This **will not** be passed to save.
    - `<save_extras::Vector{Union{String, Vector{String}}>`: which groups and <names> to log to the data dict. This **will** be passed to save.
    """

    info"""
    Environment details
    -------------------
    This experiment uses the MountainCar environment. There is no configuration.
    """
    
    info"""
    agent details
    -------------
    ### NeuralNetwork
    The RNN used for this experiment and its total hidden size, 
    as well as a flag to use (or not use) zhu's deep 
    action network. See 
    - `latent_size::Int`: The size of the hidden layers in the neural networks.
    """
    latent_size => 64

    info"""
    ### Optimizer details
    Flux optimizers are used. See flux documentation and `ExpUtils.Flux.get_optimizer` for details.
    - `opt::String`: The name of the optimizer used
    - Parameters defined by the particular optimizer.
    """
    # opt => "RMSPropTFCentered"
    eta => 0.001

    info"""
    ### Learning update and replay details including:
    - Replay: 
        - `replay_size::Int`: How many transitions are stored in the replay.
        - `warm_up::Int`: How many steps for warm-up (i.e. before learning begins).
    """
    replay_size => 10000
    warm_up => 1000
    
    info"""
    - Update details: 
        - `lupdate::String`: Learning update name
        - `gamma::Float`: the discount for learning update.
        - `batch_size::Int`: size of batch
        - `truncation::Int`: Length of sequences used for training.
        - `update_wait::Int`: Time between updates (counted in agent interactions)
        - `target_update_wait::Int`: Time between target network updates (counted in agent interactions)
        - `hs_strategy::String`: Strategy for dealing w/ hidden state in buffer.
    """

    update => "QLearningMSE"
    gamma => 1.0    
    batch_size=>32
    hist => 1
    epsilon => 0.1
    
    update_freq => 1
    target_update_wait => 100

end

function build_ann(config, in, actions, rng)
    
    NN_config = [
        Dict("type"=>"Dense", "out"=>64, "bias"=>true, "activation"=>"relu"),
        Dict("type"=>"Dense", "out"=>64, "bias"=>true, "activation"=>"relu"),
        Dict("type"=>"Dense", "out"=>"actions", "bias"=>true, "activation"=>"linear")
    ]

    model = DeepRL.build_ann_from_config(
        (in,),
        NN_config,
        config,
        init=Flux.glorot_uniform(rng),
        actions=actions)
end

function construct_agent(env, config, rng=Random.default_rng())

    ϵ = config["epsilon"]
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
        DeepRL.RMSPropTFCentered(config["eta"]),
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

    # seed = config["seed"]
    @config_to_param seed config
    @config_to_param steps config
    # max_num_steps = config["steps"]
    
    mc = construct_env(config)

    Random.seed!(seed)
    rng = Random.default_rng()

    agent = construct_agent(mc, config, rng) #construct_agent(s, length(MinimalRLCore.get_actions(mc)))

    # extras = union(get(config, "log_extras", []), get(config, "save_extras", []))
    # data, logger = ExpUtils.construct_logger(extra_groups_and_names=extras)
    
    experiment_wrapper(config) do config

        max_episode_length = 5000

        total_rews = Float32[]
        steps_vec = Int[]
        
        front = ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇']
        p = ProgressMeter.Progress(
            steps;
            dt=0.1,
            desc="Step:",
            barglyphs=ProgressMeter.BarGlyphs('|','█',front,' ','|'))

        eps = 1
        while sum(steps_vec) < steps
            cur_step = 0
            episode_cut_off = min(max_episode_length, steps - sum(steps_vec))
            tr, stp =
                run_episode!(mc, agent, episode_cut_off, rng) do (s, a, s′, r)
                    cur_step+=1
                    next!(p)
                end
            push!(total_rews, tr)
            push!(steps_vec, stp)
            
            eps += 1
        end

        agent.model, total_rews, steps_vec
    end
    
end

end

