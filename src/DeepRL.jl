module DeepRL

greet() = print("Hello Deep Reinforcement Learning!")

using Reexport, Lazy
@reexport using MinimalRLCore

export ExperienceReplay,
    WeightedExperienceReplay, OnlineReplay,
    size, getindex, add!, sample

include("util/table_buffer.jl")
include("replay.jl")
include("image_buffer.jl")
include("optimizers.jl")

export MountainCar, Atari
include("env.jl")

export ϵGreedy, sample
include("policy.jl")

export
    QLearning,
    DoubleQLearning,
    QLearningHuberLoss,
    DoubleQLearningHuberLoss

include("update.jl")

export DQNAgent
include("agent.jl")

include("utils.jl")


end # module
