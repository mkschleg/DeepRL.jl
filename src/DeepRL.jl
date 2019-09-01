module DeepRL

greet() = print("Hello Deep Reinforcement Learning!")

using Reexport
@reexport using JuliaRL

export ExperienceReplay,
    WeightedExperienceReplay,
    size, getindex, add!, sample

include("replay.jl")


export MountainCar
include("env.jl")

export ϵGreedy, sample
include("policy.jl")

export DQNAgent
include("agent.jl")


end # module
