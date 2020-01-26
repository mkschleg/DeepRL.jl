module DeepRL

greet() = print("Hello Deep Reinforcement Learning!")

using Reexport, Lazy
@reexport using RLCore

export ExperienceReplay,
    WeightedExperienceReplay, OnlineReplay,
    size, getindex, add!, sample

include("replay.jl")
include("image_buffer.jl")
include("optimizers.jl")

export MountainCar, Atari
include("env.jl")

export ϵGreedy, sample
include("policy.jl")

export QLearning, DoubleQLearning, AuxQLearning, TDLearning
include("update.jl")

export DQNAgent, OnlineDQNAgent, AT_DQNAgent, ImageDQNAgent
include("agent.jl")



include("utils.jl")

end # module
