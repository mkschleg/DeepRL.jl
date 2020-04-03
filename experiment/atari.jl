
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
    agent = DeepRL.RevisitingALEDQNBaseline(env)

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
