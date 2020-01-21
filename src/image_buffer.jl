
using Flux
using CuArrays


"""
    ImageBuffer(size::Int, image_manip, image_size)

A cicular buffer for images.

# Arguments
- `image_manip`: A function (or callable object) which takes the raw image and returns the pre-processed image (returning UInt8 arrays).

"""
mutable struct ImageBuffer{A<:AbstractArray{UInt8}, IM, TPL<:Tuple}
    step_buffer::Array{Int, 1}
    img_buffer::A
    image_manip::IM
    cur_idx::Int
    capacity::Int
    full::Bool
    img_size::TPL
end

ImageBuffer(size::Int, image_manip, img_size) = 
    ImageBuffer(
        zeros(Int, size),
        zeros(UInt8, img_size..., size),
        image_manip,
        1,
        size,
        false,
        img_size)

function add!(imb::ImageBuffer, img)

    ret = imb.cur_idx
    if imb.img_buffer isa CuArray
        view(imb, ret) .= imb.image_manip(img) |> gpu
    else
        view(imb, ret) .= imb.image_manip(img)
    end

    imb.cur_idx += 1
    if imb.cur_idx > imb.capacity
        imb.cur_idx = 1
        imb.full = true
    end

    return ret
end

function view_add!(imb::ImageBuffer, img)
    idx = add!(imb, img)
    return view(imb, idx)
end

getindex(imb::ImageBuffer{A, IM, Tuple{Int, Int, Int}}, idx) where {A, IM} = getindex(imb.img_buffer, :, :, :, idx)
Base.view(imb::ImageBuffer{A, IM, Tuple{Int, Int, Int}}, idx) where {A, IM} = view(imb.img_buffer, :, :, :, idx)

getindex(imb::ImageBuffer{A, IM, Tuple{Int, Int}}, idx) where {A, IM} = getindex(imb.img_buffer, :, :, idx)
Base.view(imb::ImageBuffer{A, IM, Tuple{Int, Int}}, idx) where {A, IM} = view(imb.img_buffer, :, :, idx)


abstract type AbstractImageReplay <: AbstractReplay end

"""
    HistImageReplay(size, img_size::Tuple{Int,Int}, image_manip, img_norm, hist, batchsize)

This is a experience replay buffer designed for managing a history of images. Currently it only allows for images with a single channel (i.e. gray scale). Currently the batchsize is needed ahead of time, but this could change in the future.

# Arguments
- `image_mainp`: Funciton (or callable object) which takes the raw image and returns the processed image (w/o normalizing to Floats)
- `image_norm`: Function (or callable object) which takes the processed image and returns the normalized image (type Array{Float32}).

"""
struct HistImageReplay{ER<:AbstractReplay, IB<:ImageBuffer, IN} <: AbstractImageReplay
    exp_replay::ER
    image_buffer::IB
    img_norm::IN
    hist::Int
    cur_state::Array{Int, 1}
    s::Array{Float32, 4}
    sp::Array{Float32, 4}
end

function HistImageReplay(size, img_size::Tuple{Int, Int}, img_manip, img_norm, hist, batchsize)
    er = ExperienceReplay(size,
                          (Array{Int, 1}, Int, Array{Int, 1}, Float32, Bool),
                          (:s, :a, :sp, :r, :t))
    imb = ImageBuffer(size + 2*hist, img_manip, img_size)
    HistImageReplay(
        er,
        imb,
        img_norm,
        hist,
        ones(Int64, hist),
        zeros(Float32, img_size..., hist, batchsize),
        zeros(Float32, img_size..., hist, batchsize))
end


size(er::HistImageReplay) = size(er.exp_replay)

function add!(er::HistImageReplay, state::Array{UInt8})
    idx = add!(er.image_buffer, state)
    er.cur_state .= idx
end

"""
    add!(er::HistImageReplay, exp_tuple)

# Arguments
- `transition::Tuple`: A transition tuple (s, a, sp, r, t)
"""
function add!(er::HistImageReplay, transition::TPL) where {TPL<:Tuple}
    sp_img = transition[3]
    idx = add!(er.image_buffer, sp_img)
    s = copy(er.cur_state)
    sp = [[idx]; er.cur_state[1:3]]
    add!(er.exp_replay, (s, transition[2], sp, transition[4], transition[5]))
    er.cur_state .= sp
end

function sample(er::HistImageReplay, batch_size; rng=Random.GLOBAL_RNG)
    # idx = rand(rng, 1:size(er), batch_size)
    # rows = getindex(er.exp_replay, idx)

    rows = sample(er.exp_replay, batch_size; rng=rng)
    for i = 1:batch_size
        er.s[:, :, :, i] .= er.img_norm(er.image_buffer[rows.s[i]])
        er.sp[ :, :, :, i] .= er.img_norm(er.image_buffer[rows.sp[i]])
    end
    # er.s .= cat([er.image_buffer[rows.s[i]]./256f0 for i = 1:batch_size]...;dims=4)
    # er.sp .= cat([er.image_buffer[rows.sp[i]]./256f0 for i = 1:batch_size]...;dims=4)
    return (s=er.s, a=rows.a, sp=er.sp, r=rows.r, t=rows.t)
end

