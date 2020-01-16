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
