

module AtariExperiment

using DeepRL
using Flux
using Random
using ProgressMeter
using Plots
using TensorBoardLogger
using Logging
using LinearAlgebra
using BSON: @save

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


function episode!(env, agent, rng, max_steps, total_steps, progress_bar=nothing, save_callback=nothing)
    terminal = false
    s_t = start!(env, rng)
    action = start!(agent, s_t, rng)

    total_rew = 0
    steps = 1

    while !terminal
        
        s_tp1, rew, terminal = step!(env, action)

        action = step!(agent, s_tp1, rew, terminal, rng)
        total_rew += rew
        steps += 1
        if !(progress_bar isa Nothing)
            next!(progress_bar)
        end
        if !(save_callback isa Nothing) && total_steps+steps % 100000 == 0
            save_callback(agent, total_steps+steps)
        end

        if steps == max_steps
            terminal = break
        end
        
    end
    return total_rew, steps
end

flatten(x) = reshape(x, :, size(x, 4))

function main_experiment(seed, num_max_steps; gamename="breakout")

    lg=TBLogger("tensorboard_logs/run", min_level=Logging.Info)

    if !isdir("models")
        mkdir("models")
    end
    
    ϵ=0.1
    γ=0.99
    batch_size=32
    buffer_size = 1000000
    tn_counter_init=10000
    hist_length = 4
    update_wait = 4

    Random.seed!(Random.GLOBAL_RNG, seed)
    env = Atari(gamename; seed=rand(UInt32), frameskip=4)

    image_replay = DeepRL.HistImageReplay(buffer_size, DeepRL.image_manip_atari, (84,84), hist_length)

    s_prototype = zeros(Float32, 84, 84, hist_length, batch_size)

    model = Chain(
           Conv((8,8), 4=>32, relu, stride=4),
           Conv((4,4), 32=>64, relu, stride=2),
           Conv((3,3), 64=>64, relu, stride=1),
           flatten,
           Dense(3136, 512, relu),
           Dense(512, 4)) |> gpu

    target_network  = deepcopy(model)
    
    agent = ImageDQNAgent(model,
                          target_network,
                          image_replay,
                          RMSProp(0.00025, 0.95),
                          QLearning(γ),
                          ϵGreedy(ϵ, get_actions(env)),
                          500000,
                          batch_size,
                          tn_counter_init,
                          update_wait)
    
    total_rews = Array{Int,1}()
    steps = Array{Int,1}()

    save_callback(agnt, step) = begin
        model = agnt.model
        @save "models/step_$(step).bson" model
    end
    
    front = ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇']
    p = ProgressMeter.Progress(
        num_max_steps;
        dt=0.5,
        desc="Step: ",
        barglyphs=ProgressMeter.BarGlyphs('|','█',front,' ','|'),
        barlen=Int64(floor(500/length(front))))

    # data_range = collect.(collect(Iterators.product(-1.0:0.01:1.0, -1.0:0.01:1.0)))
    # # with_logger(lg) do
    e = 0
    total_steps = 0
    while sum(steps) < num_max_steps
        tr, s = episode!(env, agent, Random.GLOBAL_RNG, Inf, total_steps, p, save_callback)
        push!(total_rews, tr)
        push!(steps, s)
        total_steps += s
        e += 1
    end

    # end

    close(env)
    
    return agent.model, total_rews, steps
    
end

end
