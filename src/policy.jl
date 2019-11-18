abstract type AbstractPolicy end

abstract type AbstractValuePolicy <: AbstractPolicy end

struct ϵGreedy <: AbstractValuePolicy
    ϵ::Float64
end

function sample(ap::ϵGreedy, values, rng)
    if rand(rng) > ap.ϵ
        return findmax(values)[2]
    else
        return rand(rng, 1:length(values))
    end
end

function get_prob(ap::ϵGreedy, values, action)
    if action == findmax(values)[2]
        return 1 - ap.ϵ + (ap.ϵ / length(values))
    else
        return ap.ϵ / length(values)
    end
end
