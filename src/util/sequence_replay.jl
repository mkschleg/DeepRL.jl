


abstract type AbstractSequenceReplay <: AbstractReplay end

mutable struct SequenceReplay{CB} <: AbstractSequenceReplay
    buffer::CB
    place::Int64
end

function SequenceReplay(size, types, shapes, column_names) 
    cb = CircularBuffer(size, types, shapes, column_names)
    SequenceReplay(cb, 1)
end

Base.length(er::SequenceReplay) = length(er.buffer)
Base.getindex(er::SequenceReplay, idx) =
    er.buffer[idx]

Base.view(er::SequenceReplay, idx) =
    @view er.buffer[idx]

function Base.push!(er::SequenceReplay, experience)
    if er.buffer._full
        er.place = (er.place % capacity(er.buffer)) + 1
    end
    push!(er.buffer, experience)
end

sample(er::SequenceReplay, batch_size, seq_length) = sample(Random.GLOBAL_RNG, er, batch_size, seq_length)

function sample(rng::Random.AbstractRNG, er::SequenceReplay, batch_size, seq_length)
    start_inx = rand(rng, 1:(length(er) + 1 - seq_length), batch_size)
    e = [view(er, start_inx .+ (i-1)) for i ∈ 1:seq_length]
    start_inx, e
end

mutable struct EpisodicSequenceReplay{CB} <: AbstractSequenceReplay
    buffer::CB
    place::Int64
    terminal_locs::Vector{Int}
    terminal_symbol::Symbol
end

function EpisodicSequenceReplay(size, types, shapes, column_names; terminal_symbol = :t)
    cb = CircularBuffer(size, types, shapes, column_names)
    EpisodicSequenceReplay(cb, 1, Int[], terminal_symbol)
end


Base.length(er::EpisodicSequenceReplay) = length(er.buffer)
Base.getindex(er::EpisodicSequenceReplay, idx) =
    er.buffer[idx]

Base.view(er::EpisodicSequenceReplay, idx) =
    @view er.buffer[idx]

function Base.push!(er::EpisodicSequenceReplay, experience)
    if er.buffer._full
        er.place = (er.place % capacity(er.buffer)) + 1
    end
    push!(er.buffer, experience)
end

function get_episode_ends(er::EpisodicSequenceReplay)
    # TODO: n-computations. Maybe store in a cache?
    findall((exp)->exp::Bool, er.buffer._stg_tuple[er.terminal_symbol])
end

function get_valid_starting_range(s, e, seq_length)
    if e - seq_length <= s
        s:s
    else
        (s:e-seq_length)
    end
end


function get_valid_indicies(er::EpisodicSequenceReplay, seq_length)
    # episode_ends = get_episode_ends(er)

    1:(length(er) + 1 - seq_length)
    
end

function get_sequence(er::EpisodicSequenceReplay, start_ind, max_seq_length)
    # ret = [view(er, start_ind)]
    ret = [er[start_ind]]
    er_size = length(er)
    if ((start_ind + 1 - 1) % er_size) + 1 == er.place || ret[end][er.terminal_symbol][]::Bool
        return ret
    end
    for i ∈ 1:(max_seq_length-1)
        push!(ret, er[(((start_ind + i - 1) % er_size) + 1)])
        if ret[end][er.terminal_symbol][]::Bool || ((start_ind + i + 1 - 1) % er_size) + 1 == er.place 
            break
        end
    end
    ret
end

sample(er::EpisodicSequenceReplay,
       batch_size,
       max_seq_length) =
           sample(Random.GLOBAL_RNG, er, batch_size, max_seq_length)

function sample(rng::Random.AbstractRNG,
                er::EpisodicSequenceReplay,
                batch_size,
                max_seq_length)
    # get valid starting indicies
    valid_inx = get_valid_indicies(er, 1)
    start_inx = rand(rng, valid_inx, batch_size)
    exp = [get_sequence(er, si, max_seq_length) for si ∈ start_inx]
    start_inx, exp
    # padding and batching handled by agent.
end
