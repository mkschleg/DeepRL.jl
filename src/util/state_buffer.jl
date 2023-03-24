
# using Flux
# using CuArrays

"""
    StateBuffer(size::Int, state_size)

A cicular buffer for states. Typically used for images, can be used for state shapes up to 4d.
"""
mutable struct StateBuffer{T, N, S}
    buffer::Array{T, N}
    cur_idx::Int
    capacity::Int
    full::Bool
    state_size::S
end
StateBuffer{T}(size, state_size) where {T} = begin
    s_bfr = zeros(T, state_size..., size)
    StateBuffer(s_bfr,
                1,
                size,
                false,
                state_size)
end

ImageBuffer(size, img_shape) = StateBuffer{UInt8}(size, img_shape)

Base.isempty(sb::StateBuffer) = (sb.cur_idx == 1 && !sb.full)
isfull(sb::StateBuffer) = sb.full
capacity(sb::StateBuffer) = sb.capacity
Base.length(sb::StateBuffer) = isfull(sb) ? capacity(sb) : (sb.cur_idx - 1)
Base.eltype(sb::StateBuffer) = Base.eltype(sb.buffer)
hist_length(sb::StateBuffer) = 1

@inline Base.@propagate_inbounds function laststate(sb::StateBuffer)
    @boundscheck if length(sb) == 0
        throw(BoundsError(sb, 1))
    end
    
    if sb.cur_idx == 1
        capacity(sb)
    else
        sb.cur_idx - 1
    end
end

# TODO: Generate w/ macros.
@inline Base.@propagate_inbounds function Base.getindex(sb::StateBuffer, idx)
    if sb.state_size isa Int || sb.state_size isa Tuple{Int}
        sb.buffer[:, idx]
    elseif sb.state_size isa Tuple{Int, Int}
        sb.buffer[ :, :, idx]
    elseif sb.state_size isa Tuple{Int, Int, Int}
        sb.buffer[:, :, :, idx]
    elseif sb.state_size isa Tuple{Int, Int, Int, Int}
        sb.buffer[:, :, :, :, idx]
    end
end

@inline Base.@propagate_inbounds function Base.view(sb::StateBuffer, idx)
    if sb.state_size isa Int || sb.state_size isa Tuple{Int}
        view(sb.buffer, :, idx)
    elseif sb.state_size isa Tuple{Int, Int}
        view(sb.buffer, :, :, idx)
    elseif sb.state_size isa Tuple{Int, Int, Int}
        view(sb.buffer, :, :, :, idx)
    elseif sb.state_size isa Tuple{Int, Int, Int, Int}
        view(sb.buffer, :, :, :, :, idx)
    end
end

function Base.push!(sb::StateBuffer, state; new_episode=false)
    view(sb, sb.cur_idx) .= state
    sb.cur_idx += 1
    if sb.cur_idx > sb.capacity
        sb.cur_idx = 1
        sb.full = true
    end
    return sb
end

"""
    HistStateBuffer

A buffer which manages the state buffer, giving lastindex as a history buffer. This is useful when the user wants to deal w/ partial observability by using a short domain.
"""

mutable struct HistStateBuffer{T, N, S, SQ<:Val}
    buffer::StateBuffer{T, N, S}
    hist_length::Int
    hist::Array{Int, 1}
    squeeze::SQ
end

HistStateBuffer{T}(size, state_size, hist_length, squeeze=false) where {T} = if squeeze isa Val
    HistStateBuffer(StateBuffer{T}(size + hist_length, state_size), hist_length, ones(Int, hist_length), squeeze)
else
    HistStateBuffer(StateBuffer{T}(size + hist_length, state_size), hist_length, ones(Int, hist_length), Val(squeeze))
end

@forward HistStateBuffer.buffer Base.length, isfull, capacity, Base.isempty

hist_length(sb::HistStateBuffer) = sb.hist_length

@inline Base.@propagate_inbounds function laststate(sb::HistStateBuffer)
    @boundscheck if length(sb) == 0
        throw(BoundsError(sb, 1))
    end
    copy(sb.hist)
end

@inline Base.@propagate_inbounds function Base.getindex(sb::HistStateBuffer, idx::Array{<:Integer, 1})
    if sb.squeeze isa Val{true}
        # sb.buffer[idx]
        data = sb.buffer[idx]
        reshape(data, length(data))
    elseif sb.squeeze isa Val{false}
        sb.buffer[idx]
    end
end

@inline Base.@propagate_inbounds function Base.getindex(sb::HistStateBuffer, idx::Array{<:Integer, 2})
    if sb.squeeze isa Val{true}
        data = sb.buffer[idx]
        reshape(data, sum(Base.size(data)[1:end-1]), Base.size(data)[end])
    elseif sb.squeeze isa Val{false}
        sb.buffer[idx]
    end
end

function Base.push!(sb::HistStateBuffer, state; new_episode=false)
    push!(sb.buffer, state; new_episode=new_episode)
    hist_length = sb.hist_length
    if new_episode
        sb.hist .= laststate(sb.buffer)
    else
        sb.hist[1:(end - 1)] = sb.hist[2:end]
        sb.hist[end] = laststate(sb.buffer)
    end
    return sb
end
