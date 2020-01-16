
using Flux
# using CuArrays


mutable struct ImageBuffer{A<:AbstractArray{UInt8}, IM, TPL<:Tuple}
    step_buffer::Array{Int, 1}
    img_buffer::A
    image_manip::IM
    cur_idx::Int
    capacity::Int
    full::Bool
    img_size::TPL
end

ImageBuffer(size::Int64, image_manip, img_size) = 
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
    # if imb.img_buffer isa CuArray
    #     view(imb, ret) .= imb.image_manip(img) |> CuArray
    # else
    view(imb, ret) .= imb.image_manip(img)
    # end

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


struct HistImageReplay{ER<:AbstractReplay, IB<:ImageBuffer} <: AbstractImageReplay
    exp_replay::ER
    image_buffer::IB
    hist::Int
    cur_state::Array{Int, 1}
end

function HistImageReplay(size, img_manip, img_size::Tuple{Int, Int}, hist)
    er = ExperienceReplay(size,
                          (Array{Int, 1}, Int, Array{Int, 1}, Float32, Bool),
                          (:s, :a, :sp, :r, :t))
    imb = ImageBuffer(size, img_manip, img_size)
    HistImageReplay(er, imb, hist, ones(Int64, hist))
end


size(er::HistImageReplay) = size(er.exp_replay)

function add!(er::HistImageReplay, state)
    idx = add!(er.image_buffer, state)
    er.cur_state .= idx
end

function add!(er::HistImageReplay, state_prime, action, reward, terminal)
    idx = add!(er.image_buffer, state_prime)
    s = copy(er.cur_state)
    sp = [[idx]; er.cur_state[1:3]]
    add!(er.exp_replay, (s, action, sp, reward, terminal))
    er.cur_state .= sp
end

function sample(er::HistImageReplay, batch_size; rng=Random.GLOBAL_RNG)
    idx = rand(rng, 1:size(er), batch_size)
    rows = getrow(er.exp_replay.buffer, idx)
    s_imgs = zeros(eltype(er.image_buffer.img_buffer), er.image_buffer.img_size..., er.hist, batch_size)
    sprime_imgs = zeros(eltype(er.image_buffer.img_buffer), er.image_buffer.img_size..., er.hist, batch_size)
    @simd for i = 1:batch_size
        @inbounds s_imgs[:, :, :, i] .= er.image_buffer[rows.s[i]]
        @inbounds sprime_imgs[ :, :, :, i] .= er.image_buffer[rows.sp[i]]
    end
    return (s=s_imgs, a=rows.a, sp=sprime_imgs, r=rows.r, t=rows.t)
end
