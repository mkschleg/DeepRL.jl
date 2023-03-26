


include("environments/mountain_car.jl")
include("environments/atari.jl")


# import MinimalRLCore
# import Gymnasium
# mutable struct GymnasiumEnv <: MinimalRLCore.AbstractEnvironment
#     env::Gymnasium.GymnasiumEnv
#     GymnasiumEnv(args...; kwargs...) = new(Gymnasium.make(args...; kwargs...))
# end

# MinimalRLCore.get_reward(env::GymnasiumEnv) = env.env.reward
# MinimalRLCore.is_terminal(env::GymnasiumEnv) = env.env.terminal
# MinimalRLCore.get_state(env::GymnasiumEnv) = env.env.observations

# MinimalRLCore.start!(env::GymnasiumEnv, rng::AbstractRNG) = first(Gymnasium.reset!(env.env, seed=rand(Int, rng)))
# MinimalRLCore.start!(env::GymnasiumEnv, seed::Int) = first(Gymnasium.reset!(env.env, seed=seed))
# MinimalRLCore.step!(env::GymnasiumEnv, action, args...) = Gymnasium.step!(env.env, action)[1:3]



