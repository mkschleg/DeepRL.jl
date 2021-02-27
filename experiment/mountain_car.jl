

module MountainCarExperiment

using DeepRL
using Flux
using Random
using ProgressMeter
using TensorBoardLogger
using Logging
using Statistics


function construct_agent(s, num_actions)
    
    ϵ=0.1
    γ=1.0f0
    batch_size=32
    
    tn_update_freq=100
    update_freq=1
    min_mem_size=1000
    er_size = 10000
    hist = 1

    model = Chain(Dense(length(s)*hist, 64, Flux.relu),
                  Dense(64, 64, Flux.relu),
                  Dense(64, num_actions))

    # er = DeepRL.ExperienceReplayDef(er_size, 1, Int, DeepRL.StateBuffer{Float32}(er_size, length(s)))
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


function main_experiment(seed, max_num_steps)

    mc = MountainCar(0.0, 0.0, true)
    Random.seed!(Random.GLOBAL_RNG, seed)
    rng = Random.GLOBAL_RNG

    s = MinimalRLCore.get_state(mc)
    
    agent = construct_agent(s, length(MinimalRLCore.get_actions(mc)))

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

