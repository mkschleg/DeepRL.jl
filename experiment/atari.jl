
using DeepRL
using Flux
using Random
using ProgressMeter
# using Plots
using TensorBoardLogger
using Logging
using LinearAlgebra
using BSON: @save
# using Distributions

flatten(x) = reshape(x, :, size(x, 4))

function construct_agent(env)
    # Replicate Dopamine's Results!
    
    γ=0.99
    batch_size=32
    buffer_size = 1000000
    tn_update_freq= 8000
    hist_length = 4
    update_freq = 4
    min_mem_size = 20000

    learning_rate = 0.00025
    momentum_term = 0.00
    squared_grad_term = 0.95
    min_grad_term = 1e-5

    example_state = MinimalRLCore.get_state(env)

    init_f = Flux.glorot_uniform
    
    model = Chain(
        Conv((8,8), hist_length=>32, relu, stride=4, init=init_f),
        Conv((4,4), 32=>64, relu, stride=2, init=init_f),
        Conv((3,3), 64=>64, relu, stride=1, init=init_f),
        flatten,
        Dense(3136, 512, relu, initW=init_f),
        Dense(512, length(get_actions(env)), identity, initW=init_f)) |> gpu

    target_network  = deepcopy(model)


    return DQNAgent(
        model,
        target_network,
        QLearningHuberLoss(γ),
        DeepRL.RMSPropTFCentered(learning_rate,
                                 squared_grad_term,
                                 momentum_term,
                                 min_grad_term),
        DeepRL.ϵGreedyDecay((1.0, 0.01), 250000, min_mem_size, get_actions(env)),
        buffer_size,
        hist_length,
        example_state,
        batch_size,
        tn_update_freq,
        update_freq,
        min_mem_size;
        hist_squeeze = Val{false}(),
        state_preproc = DeepRL.image_manip_atari,
        state_postproc = DeepRL.image_norm,
        rew_transform = (r)->clamp(Float32(r), -1.0f0, 1.0f0),
        device = Flux.use_cuda[] ? Val{:gpu}() : Val{:cpu}())
end


function main_experiment(seed,
                         num_frames,
                         save_loc,
                         checkin_step;
                         gamename="breakout",
                         prog_meter_offset=0)

    max_episode_length = 27000 # From Dopamine.
    
    save_loc = joinpath(save_loc, "run_$(seed)")
    if !isdir(save_loc)
        mkpath(save_loc)
    end
    
    model_save_loc = joinpath(save_loc, "models")
    if !isdir(model_save_loc)
        mkdir(model_save_loc)
    end

    res_save_file = joinpath(save_loc, "results.bson")
    

    Random.seed!(Random.GLOBAL_RNG, seed)
    env = DeepRL.RevisitingALEAtari(gamename, rand(UInt16))

    agent = construct_agent(env)
    total_rews = Array{Int,1}()
    steps = Array{Int,1}()

    front = ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇']
    p = ProgressMeter.Progress(
        num_frames;
        dt=1,
        desc="Step: ",
        barglyphs=ProgressMeter.BarGlyphs('|','█',front,' ','|'),
        barlen=Int64(floor(100/length(front))),
        offset=prog_meter_offset)

    start_time = time()
    
    lg=TBLogger(joinpath(dirname(save_loc), "tensorboard_logs/run_$(seed)"), min_level=Logging.Info)

    eps = 0
    total_steps = 0
    prev_log_steps = 0
    with_logger(lg) do
        while sum(steps) < num_frames
            cur_step = sum(steps)
            episode_cut_off = min(max_episode_length, num_frames - sum(steps))
            tr, stp =
                run_episode!(env, agent, episode_cut_off) do (s, a, s′, r)
                    next!(p, showvalues=[(:step, cur_step), (:episode, eps)])
                    if (cur_step) % checkin_step == 0
                        model = cpu(agent.model)
                        total_time = time() - start_time
                        @save model_save_loc*"/step_$(cur_step).bson" model total_rews steps total_time
                    end
                    cur_step+=1
                end
            push!(total_rews, tr)
            push!(steps, stp)
            
            @info "" returns = tr
            @info "" steps = stp
            eps += 1
        end
        
    end
    end_time = time()
    close(env)

    model = cpu(agent.model)
    total_time = end_time - start_time
    @save res_save_file model total_rews steps total_time
    
end
