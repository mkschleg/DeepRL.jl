abstract type AbstractPolicy end

abstract type AbstractValuePolicy <: AbstractPolicy end

struct ϵGreedy{AS} <: AbstractValuePolicy
    ϵ::Float64
    action_set::AS
end

ϵGreedy(ϵ, num_actions::Int) = ϵGreedy(ϵ, 1:num_actions)

function sample(ap::ϵGreedy, values, rng)
    if rand(rng) > ap.ϵ
        return ap.action_set[findmax(values)[2]]
    else
        return rand(rng, ap.action_set)
    end
end

function get_prob(ap::ϵGreedy, values, action)
    if action == findmax(values)[2]
        return 1 - ap.ϵ + (ap.ϵ / length(action_set))
    else
        return ap.ϵ / length(action_set)
    end
end

mutable struct ϵGreedyDecay{AS} <: AbstractValuePolicy
    ϵ_range::Tuple{Float64, Float64}
    decay_period::Int
    warmup_steps::Int
    cur_step::Int
    action_set::AS
    ϵGreedyDecay(ϵ_range, decay_period, warmup_steps, action_set::AS) where {AS} = new{AS}(ϵ_range, decay_period, warmup_steps, 0, action_set)
end

ϵGreedyDecay(ϵ_range, end_step, num_actions) = ϵGreedyDecay(ϵ_range, end_step, 1:num_actions)

function sample(ap::ϵGreedyDecay, values, rng)
    ϵ_min = ap.ϵ_range[2]
    ϵ_max = ap.ϵ_range[1]
    ap.cur_step += 1
    
    steps_left = ap.decay_period + ap.warmup_steps - ap.cur_step
    bonus = (ϵ_max - ϵ_min) * steps_left / ap.decay_period
    bonus = clamp(bonus, 0.0, ϵ_max - ϵ_min)
    ϵ = ϵ_min + bonus
    # return epsilon + bonus
    if rand(rng) > ϵ
        return ap.action_set[findmax(values)[2]]
    else
        return rand(rng, ap.action_set)
    end
end

# function get_prob(ap::ϵGreedyDecay, values, action)
#     if action == findmax(values)[2]
#         return 1 - ap.ϵ + (ap.ϵ / length(action_set))
#     else
#         return ap.ϵ / length(action_set)
#     end
# end
