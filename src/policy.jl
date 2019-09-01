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
