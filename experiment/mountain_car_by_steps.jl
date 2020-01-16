

module MountainCarStepsExperiment

using DeepRL
using Flux
using Random
using ProgressMeter
using Plots
using TensorBoardLogger
using Logging
using LinearAlgebra
using BSON

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


function construct_agent(s, num_actions, tn_counter_init = 50)
    
    ϵ=0.1
    γ=1.0f0
    batch_size=32
    # tn_counter_init = 50

    model = Chain(Dense(length(s), 128, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(128, 128, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(128, 32, Flux.relu; initW=Flux.glorot_uniform),
                  Dense(32, num_actions; initW=Flux.glorot_uniform))

    target_network = mapleaves(Flux.Tracker.data, deepcopy(model)::typeof(model))

    return DQNAgent(model,
                    target_network,
                    ADAM(0.001),
                    QLearning(γ),
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

function get_heatmap(agent::DQNAgent, data_range; kwargs...)
    data = collect.(collect(Iterators.product(data_range...)))
    ret = Flux.data(agent.model(hcat(reshape(data, length(data), 1)...)))
    ret_max_action = reshape(getindex.(findmax(ret; dims=1)[2], 1), size(data)...)
    
    return heatmap(
        collect(data_range[1]),
        collect(data_range[2]),
        ret_max_action; kwargs...)
end

mutable struct HeatmapAnimation
    anim::Animation
    counter_init::Int64
    counter::Int64
    display::Bool
    num_frames::Int64
    HeatmapAnimation(anim, counter_init, display) = new(anim, counter_init, counter_init, display, 0)
end

HeatmapAnimation(counter_init::Int64, display=false) = HeatmapAnimation(Animation(), counter_init, display)

function add_heatmap!(hmanim::HeatmapAnimation, agent::DQNAgent, data_range)
    if hmanim.counter == 1
        plt = get_heatmap(agent, data_range; title="$(hmanim.num_frames*hmanim.counter_init)", clim=(1.0,3.0))
        hmanim.num_frames += 1
        if hmanim.display
            gui(plt)
        else
            frame(hmanim.anim, plt)
        end
        hmanim.counter = hmanim.counter_init
    else
        hmanim.counter -= 1
    end
end

mutable struct ModelSnapshot
    loc::String
    base_file::String
    counter_init::Int64
    counter::Int64
    num_snapshots::Int64
    function ModelSnapshot(loc::AbstractString, snapshot_time::Int64, base_file::AbstractString="model")
        if !isdir(loc)
            mkdir(loc)
        else
            throw("Directory $(loc) already exists")
        end
        new(loc, base_file, snapshot_time, 1, 0)
    end
end

function take_snapshot!(mss::ModelSnapshot, agent::DQNAgent)
    if mss.counter == 1
        bson(joinpath(mss.loc, "$(mss.base_file)_$(mss.num_snapshots*mss.counter_init).bson"), model=agent.model)
        mss.counter = mss.counter_init
        mss.num_snapshots += 1
    else
        mss.counter -= 1
    end
end

function episode!(env, agent, rng, max_steps, anim, data_range, mss)
    terminal = false
    s_t = start!(env, rng)
    action = start!(agent, s_t, rng)
    add_heatmap!(anim, agent, data_range)
    take_snapshot!(mss, agent)
    
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
        add_heatmap!(anim, agent, data_range)
        take_snapshot!(mss, agent)
    end
    return total_rew, steps
end


function main_experiment(seed, num_episodes, tn_counter=50; gif_save_file="my_gif.gif", model_snapshot_loc="dqn_snapshots")

    # lg=TBLogger("tensorboard_logs/run", min_level=Logging.Info)
    
    mc = MountainCar(0.0, 0.0, true)
    Random.seed!(Random.GLOBAL_RNG, seed)

    s = JuliaRL.get_state(mc)
    
    agent = construct_agent(s, length(JuliaRL.get_actions(mc)), tn_counter)::DQNAgent

    total_rews = zeros(num_episodes)
    steps = zeros(Int64, num_episodes)

    
    front = ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇']
    p = ProgressMeter.Progress(
        num_episodes;
        dt=0.01,
        desc="Episode:",
        barglyphs=ProgressMeter.BarGlyphs('|','█',front,' ','|'),
        barlen=Int64(floor(300/length(front))))

    gr()
    if !isdir("plts")
        mkdir("plts")
    end

    data_ranges = (0.0:0.005:1.0, 0.0:0.005:1.0)
    anim = HeatmapAnimation(50, false)
    mss = ModelSnapshot(model_snapshot_loc, 50)
    
    # with_logger(lg) do
    for e in 1:num_episodes
        total_rews[e], steps[e] = episode!(mc, agent, Random.GLOBAL_RNG, 50000, anim, data_ranges, mss)
        next!(p)
    end

    if !anim.display
        gif(anim.anim, gif_save_file)
    end
    # end

    return total_rews, steps
    
end

end
