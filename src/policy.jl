
import CuArrays: CuArray

abstract type AbstractPolicy end

abstract type AbstractValuePolicy <: AbstractPolicy end

action_set(ap::AbstractValuePolicy) = nothing
Base.eltype(ap::AbstractValuePolicy) = eltype(action_set(ap))

_get_max_action(ap::AbstractValuePolicy, values) =
    action_set(ap)[findmax(values)[2]]
_get_max_action(ap::AbstractValuePolicy, values::CuArray) = 
    action_set(ap)[findmax(cpu(values))[2]]

"""
    ϵGreedy(ϵ, action_set)
    ϵGreedy(ϵ, num_actions)

Simple ϵGreedy value policy.
"""
Base.@kwdef struct ϵGreedy{AS} <: AbstractValuePolicy
    ϵ::Float64
    action_set::AS
end

ϵGreedy(ϵ::Float64, num_actions::Int) = ϵGreedy(ϵ, 1:num_actions)

action_set(ap::ϵGreedy) = ap.action_set

"""
    sample(ap::ϵGreedy, values, rng)

Select an action according to the values.
"""
function sample(ap::ϵGreedy, values, rng)
    if rand(rng) > ap.ϵ
        return ap.action_set[findmax(values)[2]]
    else
        return rand(rng, ap.action_set)
    end
end

"""
    get_prob(ap::ϵGreedy, values, action)

Get probabiliyt of action according to values.
"""
function get_prob(ap::ϵGreedy, values, action)
    if action == findmax(values)[2]
        return 1 - ap.ϵ + (ap.ϵ / length(action_set))
    else
        return ap.ϵ / length(action_set)
    end
end


"""
    ϵGreedyDecay{AS}(ϵ_range, decay_period, warmup_steps, action_set::AS)
    ϵGreedyDecay(ϵ_range, end_step, num_actions)

This is an acting policy which decays exploration linearly over time. This api will possibly change overtime once I figure out a better way to specify decaying epsilon.

# Arguments
`ϵ_range::Tuple{Float64, Float64}`: (max epsilon, min epsilon)
`decay_period::Int`: period epsilon decays
`warmup_steps::Int`: number of steps before decay starts
"""
Base.@kwdef mutable struct ϵGreedyDecay{AS} <: AbstractValuePolicy
    ϵ_range::Tuple{Float64, Float64}
    decay_period::Int
    warmup_steps::Int
    cur_step::Int = 0
    action_set::AS
    ϵGreedyDecay(ϵ_range, decay_period, warmup_steps, action_set::AS) where {AS} =
        new{AS}(ϵ_range, decay_period, warmup_steps, 0, action_set)
end

ϵGreedyDecay(ϵ_range, end_step, num_actions) = ϵGreedyDecay(ϵ_range, end_step, 1:num_actions)

action_set(ap::ϵGreedyDecay) = ap.action_set

function _get_eps_for_step(ap::ϵGreedyDecay, step=ap.cur_step)
    ϵ_min = ap.ϵ_range[2]
    ϵ_max = ap.ϵ_range[1]
    
    steps_left = ap.decay_period + ap.warmup_steps - step
    bonus = (ϵ_max - ϵ_min) * steps_left / ap.decay_period
    bonus = clamp(bonus, 0.0, ϵ_max - ϵ_min)
    ϵ_min + bonus
end

function sample(ap::ϵGreedyDecay, values, rng)
    ap.cur_step += 1
    ϵ = _get_eps_for_step(ap::ϵGreedyDecay)
    if rand(rng) > ϵ
        return _get_max_action(ap, values)
    else
        return rand(rng, ap.action_set)
    end
end

function sample(ap::ϵGreedyDecay, values, rng, step)
    ϵ = _get_eps_for_step(ap::ϵGreedyDecay, step)
    if rand(rng) > ϵ
        return _get_max_action(ap, values)
    else
        return rand(rng, ap.action_set)
    end
end

function get_prob(ap::ϵGreedyDecay, values, action, step=ap.cur_step)
    ϵ = _get_eps_for_step(ap, step)
    if action == findmax(values)[2]
        return 1 - ϵ + (ap.ϵ / length(action_set))
    else
        return ϵ / length(action_set)
    end
end


