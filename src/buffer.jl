using DataFrames
import Base.size, Base.getindex


"""
CircularBuffer

Maintains a buffer of fixed size w/o reallocating and deallocating memory through a circular queue data struct.
"""
mutable struct CircularBuffer
    """The structure the data is stored"""
    _data_frame::DataFrame
    """Current column."""
    _current_row::Int64
    """Max size"""
    _capacity::Int64
    """Whether the datastruct is full (i.e. has gone through a single rotation)"""
    _full::Bool
    """Data_types of the data stored in the buffer."""
    _data_types::Array{DataType, 1}
    function CircularBuffer(size, types; column_names=nothing)
        if column_names != nothing
            data_frame = DataFrame(types, column_names, size)
        else
            data_frame = DataFrame(types, size)
        end
        new(data_frame, 1, size, false, types)
    end
    function CircularBuffer(args...; kwargs...)
        data_frame = DataFrame(args...; kwargs...)
        new(data_frame, 1, size(data_frame)[1], false, eltypes(d))
    end
end


"""
    add!(buffer, data)

    Adds data to the buffer, where data is an array of collections of types defined in CircularBuffer._data_types

    returns row of data of added data
"""
function add!(buffer::CircularBuffer, data)
    ret = buffer._current_row
    for (idx, dat) in enumerate(data)
        buffer._data_frame[buffer._current_row, idx] = copy(dat)
    end
    buffer._current_row += 1
    if buffer._current_row > buffer._capacity
        buffer._current_row = 1
        buffer._full = true
    end
    return ret
end

"""
    size(buffer)

    Returns the current amount of data in the circular buffer.
    If the full flag is true then we return the size of the whole data frame.

"""
function size(buffer::CircularBuffer)
    if (buffer._full)
        size(buffer._data_frame)
    else
        (buffer._current_row-1, length(buffer._data_types))
    end
end

"""
    capacity(buffer)

    returns the max number of elements the buffer can store.
"""
capacity(buffer::CircularBuffer) = buffer._capacity
getindex(buffer::CircularBuffer, idx) = getindex(buffer._data_frame, idx)
getrow(buffer::CircularBuffer, idx) = buffer._data_frame[idx,:]
