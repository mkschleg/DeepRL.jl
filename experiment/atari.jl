

# module AtariExperiment

using DeepRL
using Flux
using Random
using ProgressMeter
using Plots
using TensorBoardLogger
using Logging
using LinearAlgebra
using BSON: @save

glorot_uniform(rng::Random.AbstractRNG, dims...) =
    (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) =
    randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))

he_normal(rng::Random.AbstractRNG, dims...) =
    randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims)) * 0.5f0

image_norm(img) = img./256f0
flatten(x) = reshape(x, :, size(x, 4))

function construct_agent(env)
    # Replicate Marlos' work!
    
    γ=0.99
    batch_size=32
    buffer_size = 1000000
    tn_update_freq= 10000
    hist_length = 4
    update_wait = 5
    min_mem_size = 50000

    learning_rate = 0.00025
    momentum_term = 0.95
    squared_grad_term = 0.95
    min_grad_term = 0.01

    image_replay = DeepRL.HistImageReplay(buffer_size,
                                          (84,84),
                                          DeepRL.image_manip_atari,
                                          image_norm,
                                          hist_length,
                                          batch_size)

    model = Chain(
        Conv((8,8), 4=>32, relu, stride=4),
        Conv((4,4), 32=>64, relu, stride=2),
        Conv((3,3), 64=>64, relu, stride=1),
        flatten,
        Dense(3136, 512, relu),
        Dense(512, length(get_actions(env)))) |> gpu

    target_network  = deepcopy(model)
    
    agent = DQNAgent{Int}(model,
                          target_network,
                          DeepRL.RMSPropTF(learning_rate,
                                           squared_grad_term,
                                           momentum_term,
                                           min_grad_term),
                          QLearning(γ),
                          DeepRL.ϵGreedyDecay((1.0, 0.01), 1000000, 10000, get_actions(env)),
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
        if (total_steps+steps) % 50000 == 0
            # println("Save Model!")
            save_callback(agent, total_steps+steps)
        end
    end
    if !(progress_bar isa Nothing)
        next!(progress_bar, showvalues=[(:episode, e), (:step, total_steps+steps)])
    end
    steps = 1

    while !terminal
        
        s_tp1, rew, terminal = step!(env, action)

        action = step!(agent, s_tp1, rew, terminal, rng)

        if !(save_callback isa Nothing) 
            if (total_steps+steps) % 50000 == 0
                save_callback(agent, total_steps+steps)
            end
        end
        
        total_rew += rew        
        steps += 1
        if !(progress_bar isa Nothing)
            next!(progress_bar, showvalues=[(:episode, e), (:step, total_steps+steps)])
        end

        if total_steps+steps >= max_steps
            terminal = break
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
    env = Atari(gamename;
                seed=rand(UInt16),
                frameskip=5,
                repeat_action_probability=0.25f0,
                reward_clip=true)

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
        # println("episode")
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
