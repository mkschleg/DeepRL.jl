include("buffer.jl")
include("sumtree.jl")

import Random
import Base.getindex, Base.size

mutable struct ExperienceReplay{CB}
    buffer::CB
end

ExperienceReplay(size, types, column_names) =
    ExperienceReplay(CircularBuffer(size, types, column_names))

size(er::ExperienceReplay) = size(er.buffer)
getindex(er::ExperienceReplay, idx) = getindex(er.buffer, idx)
add!(er::ExperienceReplay, experience) = add!(er.buffer, experience)

function sample(er::ExperienceReplay, batch_size; rng=Random.GLOBAL_RNG)
    idx = rand(rng, 1:size(er), batch_size)
    return getrow(er.buffer, idx)
end


mutable struct WeightedExperienceReplay{CB}
    buffer::CB
    sumtree::SumTree
end

WeightedExperienceReplay(size, types, column_names) =
    new(CircularBuffer(size, types, column_names),
        SumTree{Int64}(size))

size(er::WeightedExperienceReplay) = size(er.buffer)
getindex(er::WeightedExperienceReplay, idx) = getindex(er.buffer, idx)
function add!(er::WeightedExperienceReplay, experience, weight)
    idx = add!(er.buffer, experience)
    add!(er.sumtree, weight, idx)
    return
end

function sample(er::WeightedExperienceReplay, batch_size; rng=Random.GLOBAL_RNG)
    batch_idx, batch_priorities, idx = sample(er.sumtree, batch_size; rng=rng)
    return getrow(er.buffer, idx)
end

