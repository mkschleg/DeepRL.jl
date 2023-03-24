module DeepRL

greet() = print("Hello Deep Reinforcement Learning!")

import Reexport: @reexport
import MacroTools: @forward

import MinimalRLCore



# export ExperienceReplay,
#     WeightedExperienceReplay,
#     OnlineReplay,
#     getindex,
#     add!,
#     sample

include("util/state_buffer.jl")
include("util/replay.jl")
include("utils.jl")

include("optimizers.jl")

export MountainCar, Atari
include("environments.jl")

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


# include("exp_utils.jl")

end # module
