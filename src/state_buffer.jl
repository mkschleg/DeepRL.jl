
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
@inline Base.@propagate_inbounds function Base.getindex(sb::StateBuffer, idx) where {A}
    if sb.state_size isa Int
        sb.buffer[:, idx]
    elseif sb.state_size isa Tuple{Int, Int}
        sb.buffer[ :, :, idx]
    elseif sb.state_size isa Tuple{Int, Int, Int}
        sb.buffer[:, :, :, idx]
    elseif sb.state_size isa Tuple{Int, Int, Int, Int}
        sb.buffer[:, :, :, :, idx]
    end
end

@inline Base.@propagate_inbounds function Base.view(sb::StateBuffer, idx) where {A}
    if sb.state_size isa Int
        view(sb.buffer, :, idx)
    elseif sb.state_size isa Tuple{Int, Int}
        view(sb.buffer, :, :, idx)
    elseif sb.state_size isa Tuple{Int, Int, Int}
        view(sb.buffer, :, :, :, idx)
    elseif sb.state_size isa Tuple{Int, Int, Int, Int}
        view(sb.buffer, :, :, :, :, idx)
    end
end

function Base.push!(sb::StateBuffer, state)
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

@inline Base.@propagate_inbounds function laststate(sb::HistStateBuffer)
    @boundscheck if length(sb) == 0
        throw(BoundsError(sb, 1))
    end
    sb.hist
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

function Base.push!(sb::HistStateBuffer, state)
    push!(sb.buffer, state)
    hist_length = sb.hist_length
    sb.hist[1:(hist_length - 1)] = sb.hist[2:end]
    sb.hist[end] = laststate(sb.buffer)
    return sb
end


# abstract type AbstractImageReplay <: AbstractReplay end

# """
#     HistImageReplay(size, img_size::Tuple{Int,Int}, image_manip, img_norm, hist, batchsize)

# This is a experience replay buffer designed for managing a history of images. Currently it only allows for images with a single channel (i.e. gray scale). Currently the batchsize is needed ahead of time, but this could change in the future.

# # Arguments
# - `image_mainp`: Funciton (or callable object) which takes the raw image and returns the processed image (w/o normalizing to Floats)
# - `image_norm`: Function (or callable object) which takes the processed image and returns the normalized image (type Array{Float32}).

# """
# struct HistImageReplay{ER<:AbstractReplay, IB<:ImageBuffer, IN} <: AbstractImageReplay
#     exp_replay::ER
#     image_buffer::IB
#     img_norm::IN
#     hist::Int
#     cur_state::Array{Int, 1}
#     s::Array{Float32, 4}
#     sp::Array{Float32, 4}
# end

# function HistImageReplay(size, img_size::Tuple{Int, Int}, img_manip, img_norm, hist, batchsize)
#     er = ExperienceReplay(size,
#                           (Array{Int, 1}, Int, Array{Int, 1}, Float32, Bool),
#                           (:s, :a, :sp, :r, :t))
#     imb = ImageBuffer(size + 2*hist, img_manip, img_size)
#     HistImageReplay(
#         er,
#         imb,
#         img_norm,
#         hist,
#         ones(Int64, hist),
#         zeros(Float32, img_size..., hist, batchsize),
#         zeros(Float32, img_size..., hist, batchsize))
# end


# size(er::HistImageReplay) = size(er.exp_replay)

# function add!(er::HistImageReplay, state::Array{UInt8})
#     idx = add!(er.image_buffer, state)
#     er.cur_state .= idx
# end

# """
#     add!(er::HistImageReplay, exp_tuple)

# # Arguments
# - `transition::Tuple`: A transition tuple (s, a, sp, r, t)
# """
# function add!(er::HistImageReplay, transition::TPL) where {TPL<:Tuple}
#     sp_img = transition[3]
#     idx = add!(er.image_buffer, sp_img)
#     s = copy(er.cur_state)
#     sp = [[idx]; er.cur_state[1:3]]
#     add!(er.exp_replay, (s, transition[2], sp, transition[4], transition[5]))
#     er.cur_state .= sp
# end

# function sample(er::HistImageReplay, batch_size; rng=Random.GLOBAL_RNG)

#     rows = sample(er.exp_replay, batch_size; rng=rng)
#     for i = 1:batch_size
#         er.s[:, :, :, i] .= er.img_norm(er.image_buffer[rows.s[i]])
#         er.sp[ :, :, :, i] .= er.img_norm(er.image_buffer[rows.sp[i]])
#     end

#     return (s=er.s, a=rows.a, sp=er.sp, r=rows.r, t=rows.t)
# end

