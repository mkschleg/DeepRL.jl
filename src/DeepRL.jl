module DeepRL

greet() = print("Hello Deep Reinforcement Learning!")

import Reexport: @reexport
import MacroTools: @forward

@reexport using MinimalRLCore

export ExperienceReplay,
    WeightedExperienceReplay,
    OnlineReplay,
    getindex,
    add!,
    sample

include("state_buffer.jl")
include("replay.jl")
include("optimizers.jl")

export MountainCar, Atari
include("env.jl")

export ϵGreedy, sample
include("policy.jl")


include("update_utils.jl")

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
