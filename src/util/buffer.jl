"""
CircularBuffer

    Maintains a buffer of fixed size w/o reallocating and deallocating memory through a circular queue data struct.
"""
mutable struct CircularBuffer{TPL, TPS, NAMES}
    """The structure the data is stored"""
    _stg_tuple::TPL
    """Current column."""
    _current_row::Int
    """Max size"""
    _capacity::Int
    """Whether the datastruct is full (i.e. has gone through a single rotation)"""
    _full::Bool
    """Data_types of the data stored in the buffer."""
    _data_types::TPS
    """Names"""
    # _names::NTPL
    function CircularBuffer(size, types, shapes, column_names)
        @assert length(types) == length(shapes)
        @assert length(types) == length(column_names)
    
        data = NamedTuple{column_names}(shapes[i] == 1 ? zeros(types[i], size) : zeros(types[i], shapes[i]..., size) for i in 1:length(types))
        return new{typeof(data), typeof(types), column_names}(data, 1, size, false, types)
    end
end


_get_data_row(x::Array{T, 1}, idx) where {T} = x[idx]
_get_data_row(x::Array{T, 2}, idx) where {T} = x[:, idx]
_get_data_row(x::Array{T, 3}, idx) where {T} = x[:, :, idx]
_get_data_row(x::Array{T, 4}, idx) where {T} = x[:, :, :, idx]
_get_data_row(x::Array{T, 5}, idx) where {T} = x[:, :, :, :, idx]

_get_data_row_view(x::Array{T, 1}, idx) where {T} = @view x[idx]
_get_data_row_view(x::Array{T, 2}, idx) where {T} = @view x[:, idx]
_get_data_row_view(x::Array{T, 3}, idx) where {T} = @view x[:, :, idx]
_get_data_row_view(x::Array{T, 4}, idx) where {T} = @view x[:, :, :, idx]
_get_data_row_view(x::Array{T, 5}, idx) where {T} = @view x[:, :, :, :, idx]

_set_data_row!(x::Array{T, 1}, d::T, idx) where {T} = x[idx] = d
_set_data_row!(x::Array{T, 2}, d::Array{T, 1}, idx) where {T} = x[:, idx] .= d
_set_data_row!(x::Array{T, 3}, d::Array{T, 2}, idx) where {T} = x[:, :, idx] .= d
_set_data_row!(x::Array{T, 4}, d::Array{T, 3}, idx) where {T} = x[:, :, :, idx] .= d
_set_data_row!(x::Array{T, 5}, d::Array{T, 4}, idx) where {T} = x[:, :, :, :, idx] .= d


"""
    Base.push!(buffer, data)

Adds data to the buffer, where data is an array of collections of types defined in CircularBuffer._data_types

returns row of data of added d
"""
function Base.push!(buffer::CB, data_tpl) where {CB<:CircularBuffer}
    ret = buffer._current_row

    for (idx, data) in enumerate(data_tpl)
        _set_data_row!(buffer._stg_tuple[idx], data, buffer._current_row)
    end
    
    buffer._current_row += 1
    if buffer._current_row > buffer._capacity
        buffer._current_row = 1
        buffer._full = true
    end
    return ret
end


"""
    Base.length(buffer::CircularBuffer)

Returns the current amount of data in the circular buffer. If the full flag is true then we return the size of the whole data frame.

"""
function Base.length(buffer::CircularBuffer)
    if buffer._full
        capacity(buffer)
    else
        buffer._current_row - 1
    end
end


"""
    capacity(buffer)

returns the max number of elements the buffer can store.
"""
capacity(buffer::CircularBuffer) = buffer._capacity

Base.getindex(buffer::CircularBuffer{TPL, TPS, NAMES}, idx) where {TPL, TPS, NAMES} = 
    NamedTuple{NAMES}(_get_data_row(buffer._stg_tuple[i], idx) for i in 1:length(NAMES))

Base.view(buffer::CircularBuffer{TPL, TPS, NAMES}, idx) where {TPL, TPS, NAMES} = 
    NamedTuple{NAMES}(_get_data_row_view(buffer._stg_tuple[i], idx) for i in 1:length(NAMES))
