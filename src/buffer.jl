using TypedTables
import Base.size, Base.getindex


"""
CircularBuffer

Maintains a buffer of fixed size w/o reallocating and deallocating memory through a circular queue data struct.
"""
mutable struct CircularBuffer{TBL}
    """The structure the data is stored"""
    _table::TBL
    """Current column."""
    _current_row::Int64
    """Max size"""
    _capacity::Int64
    """Whether the datastruct is full (i.e. has gone through a single rotation)"""
    _full::Bool
    """Data_types of the data stored in the buffer."""
    _data_types::Array{DataType, 1}
    """Names"""
    _names::Array{Symbol, 1}
end

function CircularBuffer(size, types, column_names)

    d = Tuple(Array{T, 1}(undef, size) for T in types)
    table = Table(NamedTuple{Symbol.(column_names)}(d))
        
    CircularBuffer(table, 1, size, false, collect(types), collect(column_names))
end

"""
    add!(buffer, data)

    Adds data to the buffer, where data is an array of collections of types defined in CircularBuffer._data_types

    returns row of data of added data
"""
function add!(buffer::CircularBuffer, data)
    ret = buffer._current_row
    for (idx, dat) in enumerate(data)
        getproperty(buffer._table, buffer._names[idx])[buffer._current_row] = copy(dat)
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
    if buffer._full
        length(buffer._table)
    else
        buffer._current_row-1
    end
end

"""
    capacity(buffer)

    returns the max number of elements the buffer can store.
"""
capacity(buffer::CircularBuffer) = buffer._capacity
# getindex(buffer::CircularBuffer, idx) = getindex(buffer._table, idx)
getrow(buffer::CircularBuffer, idx) = buffer._table[idx]

function Base.show(io::IO, buffer::CircularBuffer)
    if !buffer._full
        Base.show(io, buffer._table[1:(buffer._current_row-1)])
    else
        Base.show(io, buffer._table)
    end
end
