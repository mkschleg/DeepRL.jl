

mutable struct PrioritizedExperienceReplay{CB<:CircularBuffer} <: AbstractWeightedReplay
    buffer::CB
    sumtree::SumTree
end

PrioritizedExperienceReplay(size, types, column_names) =
    new(CircularBuffer(size, types, column_names),
        SumTree{Int64}(size))

# size(er::PrioritizedExperienceReplay) = size(er.buffer)
@forward PrioritizedExperienceReplay.buffer getindex

function add!(er::PrioritizedExperienceReplay, experience, weight)
    idx = add!(er.buffer, experience)
    add!(er.sumtree, weight, idx)
    return
end

function update!(er::PrioritizedExperienceReplay, idx, weight)
end


function sample(er::PrioritizedExperienceReplay, batch_size; rng=Random.GLOBAL_RNG)
    batch_idx, batch_priorities, idx = sample(er.sumtree, batch_size; rng=rng)
    return getrow(er.buffer, idx)
end
