
using DeepRL
using Flux
using Random
using ProgressMeter
# using Plots
using TensorBoardLogger
using Logging
using LinearAlgebra
using BSON: @save
using Distributions

function kaiming_uniform(dims...; gain=sqrt(2))
   fan_in = length(dims) <= 2 ? dims[end] : div(*(dims...), dims[end])
   bound = sqrt(3.0) * gain / sqrt(fan_in)
   return Float32.(rand(Uniform(-bound, bound), dims...))
 end

 function kaiming_normal(dims...; gain=sqrt(2))
   fan_in = length(dims) <= 2 ? dims[end] : div(*(dims...), dims[end])
   std = gain / sqrt(fan_in)
   return Float32.(rand(Normal(0.0, std), dims...))
 end



flatten(x) = reshape(x, :, size(x, 4))

function construct_agent(env)
    # Replicate Marlos' work!
    
    γ=0.99
    batch_size=32
    buffer_size = 1000000
    tn_update_freq= 10000
    hist_length = 4
    update_wait = 4
    min_mem_size = 50000

    learning_rate = 0.00025
    momentum_term = 0.95
    squared_grad_term = 0.95
    min_grad_term = 1e-2

    image_replay = DeepRL.HistImageReplay(buffer_size,
                                          (84,84),
                                          DeepRL.image_manip_atari,
                                          DeepRL.image_norm,
                                          hist_length,
                                          batch_size)

    init_f = Flux.glorot_normal
    model = Chain(
        Conv((8,8), 4=>32, relu, stride=4, init=init_f),
        Conv((4,4), 32=>64, relu, stride=2, init=init_f),
        Conv((3,3), 64=>64, relu, stride=1, init=init_f),
        flatten,
        Dense(3136, 512, relu, initW=init_f),
        Dense(512, length(get_actions(env)), identity, initW=init_f)) |> gpu

    target_network  = deepcopy(model)
    
    agent = DQNAgent{Int}(model,
                          target_network,
                          DeepRL.RMSPropTF(learning_rate,
                                           squared_grad_term,
                                           momentum_term,
                                           min_grad_term),
                          QLearningHuberLoss(γ),
                          DeepRL.ϵGreedyDecay((1.0, 0.01), 1000000, min_mem_size, get_actions(env)),
                          image_replay,
                          hist_length,
                          batch_size,
                          tn_update_freq,
                          update_wait,
                          min_mem_size)

    return agent
end



function episode!(env, agent, rng, max_steps, total_steps, progress_bar=nothing, save_callback=nothing, e=0)
    terminal = false
    
    s_t = start!(env, rng)
    action = start!(agent, s_t, rng)

    total_rew = 0
    steps = 0
    if !(save_callback isa Nothing)
        save_callback(agent, total_steps+steps)
    end
    if !(progress_bar isa Nothing)
        next!(progress_bar, showvalues=[(:episode, e), (:step, total_steps+steps)])
    end
    steps = 1

    while !terminal
        
        s_tp1, rew, terminal = step!(env, action)

        action = step!(agent, s_tp1, clamp(rew, -1, 1), terminal, rng)

        if !(save_callback isa Nothing)
            save_callback(agent, total_steps+steps)
        end
        
        total_rew += rew
        steps += 1
        if !(progress_bar isa Nothing)
            next!(progress_bar, showvalues=[(:episode, e), (:step, total_steps+steps)])
        end

        if (total_steps+steps >= max_steps) || (steps >= 18000) # 5 Minutes of Gameplay = 18k steps.
            break
        end
    end
    return total_rew, steps
end



function main_experiment(seed,
                         num_frames,
                         save_loc;
                         gamename="breakout",
                         prog_meter_offset=0,
                         checkin_step)

    lg=TBLogger("tensorboard_logs/run", min_level=Logging.Info)

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
        dt=0.01,
        desc="Step: ",
        barglyphs=ProgressMeter.BarGlyphs('|','█',front,' ','|'),
        barlen=Int64(floor(100/length(front))),
        offset=prog_meter_offset)

    start_time = time()

    save_callback(agnt, s) = begin
        if (s) % checkin_step == 0
            model = cpu(agnt.model)
            total_time = time() - start_time
            @save model_save_loc*"/step_$(s).bson" model total_rews steps total_time
        end
    end

    
    e = 0
    total_steps = 0
    while sum(steps) < num_frames
        tr, s = episode!(env,
                         agent,
                         Random.GLOBAL_RNG,
                         num_frames,
                         total_steps,
                         p,
                         save_callback,
                         e)
        push!(total_rews, tr)
        push!(steps, s)
        total_steps += s
        e += 1
    end
    end_time = time()
    close(env)

    model = cpu(agent.model)
    total_time = end_time - start_time
    @save res_save_file model total_rews steps total_time
    
    # return agent.model, total_rews, steps
    
end

# end
