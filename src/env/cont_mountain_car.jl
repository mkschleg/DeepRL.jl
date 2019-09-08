using Random
using IntervalSets

import JuliaRL

module ContMountainCarConst
const vel_limit = (-0.07, 0.07)
const pos_limit = (-1.2, 0.5)
const pos_initial_range = (-0.6, 0.4)

const ActionRange = -1..1

end

"""
ContMountainCar(pos=0.0, vel=0.0, normalized::Bool=false)
ContMountainCar(rng::AbstractRNG, normalized::Bool=false)


    The mountain car environment.

"""
mutable struct ContMountainCar <: JuliaRL.AbstractEnvironment
    pos::Float64
    vel::Float64
    actions::Interval{:closed, :closed, Float64}
    normalized::Bool
    function ContMountainCar(pos=0.0, vel=0.0, normalized::Bool=false)
        mcc = ContMountainCarConst
        @boundscheck (pos >= mcc.pos_limit[1] && pos <= mcc.pos_limit[2])
        @boundscheck (vel >= mcc.vel_limit[1] && vel <= mcc.vel_limit[2])
        new(pos, vel, ContMountainCarCons.ActionRange, normalized)
    end
end


JuliaRL.get_actions(env::ContMountainCar) = env.actions
valid_action(env::ContMountainCar, action) = action ∈ env.actions


function JuliaRL.reset!(env::ContMountainCar, rng::AbstractRNG; kwargs...)
    env.pos = (rand(rng)*(ContMountainCarConst.pos_initial_range[2]
                          - ContMountainCarConst.pos_initial_range[1])
               + ContMountainCarConst.pos_initial_range[1])
    env.vel = 0.0
end

function JuliaRL.reset!(env::ContMountainCar,
                        start_state::T;
                        kwargs...) where {T<:AbstractArray}
    if env.normalized
        env.pos = start_state[1]
        env.vel = start_state[2]
    else
        pos_limit = ContMountainCarConst.pos_limit
        vel_limit = ContMountainCarConst.vel_limit
        env.pos = (start_state[1]*(pos_limit[2] - pos_limit[1])) + pos_limit[1]
        env.vel = (start_state[2]*(vel_limit[2] - vel_limit[1])) + vel_limit[1]
    end
end


function JuliaRL.environment_step!(env::ContMountainCar,
                                   action;
                                   rng=Random.GLOBAL_RNG, kwargs...)
    
    @boundscheck valid_action(env, action)
    env.vel =
        clamp(env.vel + (action)*0.001 - 0.0025*cos(3*env.pos),
              ContMountainCarConst.vel_limit...)
    env.pos =
        clamp(env.pos + env.vel,
              ContMountainCarConst.pos_limit...)
end


function JuliaRL.get_reward(env::ContMountainCar) # -> determines if the agent_state is terminal
    if env.pos >= ContMountainCarConst.pos_limit[2]
        return 0
    end
    return -1
end


function JuliaRL.is_terminal(env::ContMountainCar) # -> determines if the agent_state is terminal
    return env.pos >= ContMountainCarConst.pos_limit[2]
end


function JuliaRL.get_state(env::ContMountainCar)
    if env.normalized
        return get_normalized_state(env)
    else
        return [env.pos, env.vel]
    end
end


function get_normalized_state(env::ContMountainCar)
    pos_limit = ContMountainCarConst.pos_limit
    vel_limit = ContMountainCarConst.vel_limit
    return [(env.pos - pos_limit[1])/(pos_limit[2] - pos_limit[1]),
            (env.vel - vel_limit[1])/(vel_limit[2] - vel_limit[1])]
end
