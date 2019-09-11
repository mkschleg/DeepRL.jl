module DeepRL

greet() = print("Hello Deep Reinforcement Learning!")

using Reexport
@reexport using JuliaRL

export ExperienceReplay,
    WeightedExperienceReplay,
    size, getindex, add!, sample

include("replay.jl")


export MountainCar, Atari
include("env.jl")

export ϵGreedy, sample
include("policy.jl")

export QLearning, DoubleQLearning
include("update.jl")

export DQNAgent
include("agent.jl")


include("utils.jl")

end # module
