
abstract type AbstractImageManip end
function call(im::AbstractImageManip, img) end

struct NoManip <: AbstractImageManip end
(im::NoManip)(img) = img

mutable struct ImageBuffer{CB<:CircularBuffer, IM<:AbstractImageManip}
    buffer::CB
    image_manip::IM
    curr_step::Int64
end

ImageBuffer(size::Int64, im, T::Type) = 
    ImageBuffer(
        CircularBuffer(size, (Int64, T), (:step, :image)),
        im, 0)

# ImageBuffer(size) = ImageBuffer(size, NoManip())

function add!(imb::IMB, img) where {IMB<:ImageBuffer}
    imb.curr_step += 1
    add!(imb.buffer, (imb.curr_step, imb.image_manip(img)))
end

@forward ImageBuffer.buffer getindex
