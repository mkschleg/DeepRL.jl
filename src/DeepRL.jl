module DeepRL

greet() = print("Hello Deep Reinforcement Learning!")

using Reexport, Lazy
@reexport using JuliaRL

export ExperienceReplay,
    WeightedExperienceReplay, OnlineReplay,
    size, getindex, add!, sample

include("replay.jl")
include("ImageBuffer.jl")

export MountainCar, Atari
include("env.jl")

export ϵGreedy, sample
include("policy.jl")

export QLearning, DoubleQLearning
include("update.jl")

export DQNAgent, OnlineDQNAgent
include("agent.jl")


include("utils.jl")

end # module
