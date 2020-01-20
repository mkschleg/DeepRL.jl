include("util/buffer.jl")
include("util/sumtree.jl")

import Random
import Base.getindex, Base.size
import DataStructures

abstract type AbstractReplay end

abstract type AbstractWeightedReplay <: AbstractReplay end

mutable struct ExperienceReplay{CB<:CircularBuffer} <: AbstractReplay
    buffer::CB
end

ExperienceReplay(size, types, column_names) =
    ExperienceReplay(CircularBuffer(size, types, column_names))

size(er::ExperienceReplay) = size(er.buffer)
@forward ExperienceReplay.buffer getindex

add!(er::ExperienceReplay, experience) = add!(er.buffer, experience)

function sample(er::ExperienceReplay, batch_size::Int; rng=Random.GLOBAL_RNG)
    idx = rand(rng, 1:size(er), batch_size)
    return getindex(er, idx)
end

mutable struct OnlineReplay{CB<:DataStructures.CircularBuffer, T<:Tuple} <: AbstractReplay
    buffer::CB
    column_names::T
end

OnlineReplay(size, types, column_names) =
    OnlineReplay(DataStructures.CircularBuffer{Tuple{types...}}(size), tuple(column_names...))

size(er::OnlineReplay) = size(er.buffer)
@forward OnlineReplay.buffer Base.lastindex
function getindex(er::OnlineReplay, idx)
    data = er.buffer[idx]
    NamedTuple{er.column_names}((getindex.(data, i) for i in 1:length(er.column_names)))
end
isfull(er::OnlineReplay) = DataStructures.isfull(er.buffer)
add!(er::OnlineReplay, experience) = push!(er.buffer, experience)

function sample(er::OnlineReplay, batch_size; rng=Random.GLOBAL_RNG)
    @assert batch_size <= size(er.buffer)[1]
    return er[(end-batch_size+1):end]
end

mutable struct WeightedExperienceReplay{CB<:CircularBuffer} <: AbstractWeightedReplay
    buffer::CB
    sumtree::SumTree
end

WeightedExperienceReplay(size, types, column_names) =
    new(CircularBuffer(size, types, column_names),
        SumTree{Int64}(size))

size(er::WeightedExperienceReplay) = size(er.buffer)
@forward WeightedExperienceReplay.buffer getindex

function add!(er::WeightedExperienceReplay, experience, weight)
    idx = add!(er.buffer, experience)
    add!(er.sumtree, weight, idx)
    return
end

function sample(er::WeightedExperienceReplay, batch_size; rng=Random.GLOBAL_RNG)
    batch_idx, batch_priorities, idx = sample(er.sumtree, batch_size; rng=rng)
    return getrow(er.buffer, idx)
end





