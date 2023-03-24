
include("buffer.jl")
include("sumtree.jl")

import Random
# import Base.getindex, Base.size
import DataStructures
import MacroTools: @forward

abstract type AbstractReplay end
abstract type AbstractWeightedReplay <: AbstractReplay end


proc_state(er::AbstractReplay, x) = identity(x)
Base.keys(er::AbstractReplay) = 1:length(er)

function Base.iterate(asr::AbstractReplay)

    state = 1
    result = asr[state]
    state += 1
    (result, state)
end

function Base.iterate(asr::AbstractReplay, state::Integer)

    if state > length(asr)
        return nothing
    end
    
    result = asr[state]
    state += 1
    
    (result, state)
end

mutable struct ExperienceReplay{CB<:CircularBuffer} <: AbstractReplay
    buffer::CB
end

const ExperienceReplayDef{TPL, TPS} = ExperienceReplay{CircularBuffer{TPL, TPS, (:s, :a, :sp, :r, :t)}}

ExperienceReplay(size, types, shapes, column_names) = begin
    cb = CircularBuffer(size, types, shapes, column_names)
    ExperienceReplay(cb)
end

ExperienceReplayDef(size, obs_size, obs_type) =
    ExperienceReplay(size,
                     (obs_type, Int, obs_type, Float32, Bool),
                     (obs_size, 1, obs_size, 1, 1),
                     (:s, :a, :sp, :r, :t))

Base.length(er::ExperienceReplay) = length(er.buffer)
Base.getindex(er::ExperienceReplay, idx) = er.buffer[idx]
Base.view(er::ExperienceReplay, idx) = @view er.buffer[idx]

Base.push!(er::ExperienceReplay, experience) = push!(er.buffer, experience)

sample(er::ExperienceReplay, batch_size) = sample(Random.GLOBAL_RNG, er, batch_size)

function sample(rng::Random.AbstractRNG, er::ExperienceReplay, batch_size)
    idx = rand(rng, 1:length(er), batch_size)
    return er[idx]
end


mutable struct WeightedExperienceReplay{CB<:CircularBuffer} <: AbstractWeightedReplay
    buffer::CB
    sumtree::SumTree
end

WeightedExperienceReplay(size, types, column_names) =
    WeightedExperienceReplay(
        CircularBuffer(size, types, column_names),
        SumTree{Int}(size))

@forward WeightedExperienceReplay.buffer getindex

function Base.push!(er::WeightedExperienceReplay, experience, weight)
    idx = push!(er.buffer, experience)
    add!(er.sumtree, weight, idx)
    return
end

function sample(er::WeightedExperienceReplay, batch_size; rng=Random.GLOBAL_RNG)
    batch_idx, batch_priorities, idx = sample(er.sumtree, batch_size; rng=rng)
    return er[idx]
end




# include("util/buffer.jl")
# include("util/sumtree.jl")

# import Random
# # import Base.getindex, Base.size
# import DataStructures

# abstract type AbstractReplay end

# function sample
# end

# function push!
# end

# abstract type AbstractWeightedReplay <: AbstractReplay end

# mutable struct ExperienceReplay{CB} <: AbstractReplay
#     buffer::CB
# end

# ExperienceReplay(size, types, shapes, column_names) = begin
#     cb = CircularBuffer(size, types, shapes, column_names)
#     ExperienceReplay(cb)
# end

# ExperienceReplayDef(size, obs_size, obs_type) =
#     ExperienceReplay(size,
#                      (obs_type, Int, obs_type, Float32, Bool),
#                      (obs_size, 1, obs_size, 1, 1),
#                      (:s, :a, :sp, :r, :t))

# Base.length(er::ExperienceReplay) = length(er.buffer)
# Base.getindex(er::ExperienceReplay, idx) = er.buffer[idx]

# Base.push!(er::ExperienceReplay, experience) = Base.push!(er.buffer, experience)

# sample(er::ExperienceReplay, batch_size) =
#     sample(Random.GLOBAL_RNG, er, batch_size)

# function sample(rng::Random.AbstractRNG, er::ExperienceReplay, batch_size)
#     idx = rand(rng, 1:length(er), batch_size)
#     return er[idx]
# end

# mutable struct OnlineReplay{CB<:DataStructures.CircularBuffer, T<:Tuple} <: AbstractReplay
#     buffer::CB
#     column_names::T
# end

# OnlineReplay(size, types, column_names) =
#     OnlineReplay(DataStructures.CircularBuffer{Tuple{types...}}(size), tuple(column_names...))

# # size(er::OnlineReplay) = size(er.buffer)
# @forward OnlineReplay.buffer Base.lastindex
# function getindex(er::OnlineReplay, idx)
#     data = er.buffer[idx]
#     NamedTuple{er.column_names}((getindex.(data, i) for i in 1:length(er.column_names)))
# end
# isfull(er::OnlineReplay) = DataStructures.isfull(er.buffer)
# Base.push!(er::OnlineReplay, experience) = push!(er.buffer, experience)

# function sample(er::OnlineReplay, batch_size; rng=Random.GLOBAL_RNG)
#     @assert batch_size <= size(er.buffer)[1]
#     return er[(end-batch_size+1):end]
# end

# warmup(::OnlineReplay, x) = x

# mutable struct WeightedExperienceReplay{CB<:CircularBuffer} <: AbstractWeightedReplay
#     buffer::CB
#     sumtree::SumTree
# end

# WeightedExperienceReplay(size, types, column_names) =
#     WeightedExperienceReplay(
#         CircularBuffer(size, types, column_names),
#         SumTree{Int}(size))

# @forward WeightedExperienceReplay.buffer getindex

# function Base.push!(er::WeightedExperienceReplay, experience, weight)
#     idx = push!(er.buffer, experience)
#     add!(er.sumtree, weight, idx)
#     return
# end

# function sample(er::WeightedExperienceReplay, batch_size; rng=Random.GLOBAL_RNG)
#     batch_idx, batch_priorities, idx = sample(er.sumtree, batch_size; rng=rng)
#     return er[idx]
# end





