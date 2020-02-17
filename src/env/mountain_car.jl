
using Random

import RLCore

module MountainCarConst
const vel_limit = (-0.07, 0.07)
const pos_limit = (-1.2, 0.5)
const pos_initial_range = (-0.6, 0.4)

const Reverse=1
const Neutral=2
const Accelerate=3
end

"""
MountainCar(pos=0.0, vel=0.0, normalized::Bool=false)
MountainCar(rng::AbstractRNG, normalized::Bool=false)


    The mountain car environment.

"""
mutable struct MountainCar <: RLCore.AbstractEnvironment
    pos::Float64
    vel::Float64
    actions::AbstractSet
    normalized::Bool
    function MountainCar(pos=0.0, vel=0.0, normalized::Bool=false)
        mcc = MountainCarConst
        @boundscheck (pos >= mcc.pos_limit[1] && pos <= mcc.pos_limit[2])
        @boundscheck (vel >= mcc.vel_limit[1] && vel <= mcc.vel_limit[2])
        new(pos, vel, Set([mcc.Reverse, mcc.Neutral, mcc.Accelerate]), normalized)
    end
end


RLCore.get_actions(env::MountainCar) = env.actions
valid_action(env::MountainCar, action) = action in env.actions


function RLCore.reset!(env::MountainCar, rng::AbstractRNG; kwargs...)
    env.pos = (rand(rng)*(MountainCarConst.pos_initial_range[2]
                          - MountainCarConst.pos_initial_range[1])
               + MountainCarConst.pos_initial_range[1])
    env.vel = 0.0
end

function RLCore.reset!(env::MountainCar,
                        start_state::T;
                        kwargs...) where {T<:AbstractArray}
    if env.normalized
        env.pos = start_state[1]
        env.vel = start_state[2]
    else
        pos_limit = MountainCarConst.pos_limit
        vel_limit = MountainCarConst.vel_limit
        env.pos = (start_state[1]*(pos_limit[2] - pos_limit[1])) + pos_limit[1]
        env.vel = (start_state[2]*(vel_limit[2] - vel_limit[1])) + vel_limit[1]
    end
end


function RLCore.environment_step!(env::MountainCar, action, rng::AbstractRNG; kwargs...)
    
    @boundscheck valid_action(env, action)
    env.vel =
        clamp(env.vel + (action - 2)*0.001 - 0.0025*cos(3*env.pos),
              MountainCarConst.vel_limit...)
    env.pos = clamp(env.pos + env.vel,
                    MountainCarConst.pos_limit...)
end


function RLCore.get_reward(env::MountainCar) # -> determines if the agent_state is terminal
    if env.pos >= MountainCarConst.pos_limit[2]
        return 0
    end
    return -1
end


function RLCore.is_terminal(env::MountainCar) # -> determines if the agent_state is terminal
    return env.pos >= MountainCarConst.pos_limit[2]
end


function RLCore.get_state(env::MountainCar)
    if env.normalized
        return get_normalized_state(env)
    else
        return [env.pos, env.vel]
    end
end


function get_normalized_state(env::MountainCar)
    pos_limit = MountainCarConst.pos_limit
    vel_limit = MountainCarConst.vel_limit
    return [(env.pos - pos_limit[1])/(pos_limit[2] - pos_limit[1]),
            (env.vel - vel_limit[1])/(vel_limit[2] - vel_limit[1])]
end
