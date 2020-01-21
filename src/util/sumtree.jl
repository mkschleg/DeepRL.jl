
import Random
import Base.getindex, Base.size, Base.eltype

mutable struct SumTree{T}
    capacity::Int64
    tree_struct::Array{Float64, 1}
    data::Array{T, 1}
    write_pos::Int64
    size::Int64
    tree_len::Int64
    SumTree{T}(capacity) where T = new{T}(
        capacity,
        zeros(2*capacity - 1),
        Array{T, 1}(undef, capacity),
        0,
        0,
        2*capacity - 1)
end

total(tree::SumTree) = tree.tree_struct[1]
size(tree::SumTree) = tree.size
capacity(tree::SumTree) = tree.capacity
eltype(tree::SumTree) = typeof(tree.data[1])
randInRange(a, b, size=1; rng=Random.GLOBAL_RNG) = (rand(rng, length(a), size).*(b-a)).+a

function update!(tree::SumTree, idx, weight)
    change = weight - tree.tree_struct[idx + 1]
    tree.tree_struct[idx + 1] = weight
    propagate!(tree, idx, change)
end

function propagate!(tree::SumTree, idx, change)
    parent = div((idx - 1), 2)
    tree.tree_struct[parent + 1] += change
    if parent != 0
        propagate!(tree, parent, change)
    end
end

function add!(tree::SumTree, weight, data)
    idx = tree.write_pos + tree.capacity - 1
    tree.data[tree.write_pos + 1] = data
    update!(tree, idx, weight)

    tree.write_pos += 1
    if tree.write_pos >= tree.capacity
        tree.write_pos = 0
    end

    if tree.size < tree.capacity
        tree.size += 1
    end
end

function get(tree::SumTree, s)
    idx = retrieve(tree, 0, s)
    data_idx = idx - tree.capacity + 1

    return idx + 1, tree.tree_struct[idx + 1], tree.data[data_idx + 1]
end

function getindex(tree::SumTree, idx::Int64)
    tree_idx = (idx - 1) + tree.capacity - 1
    return tree.tree_struct[tree_idx + 1], tree.data[idx]
end

function retrieve(tree::SumTree, idx, s)
    left = 2 * idx + 1
    if left >= tree.tree_len
        return idx
    end
    if s <= tree.tree_struct[left + 1]
        return retrieve(tree, left, s)
    else
        right = left + 1
        return retrieve(tree, right, s-tree.tree_struct[left + 1])
    end
end

function sample(tree::SumTree, batch_size; rng=Random.GLOBAL_RNG)
    batch_idx = zeros(Int64, batch_size)
    batch_priorities = zeros(Float64, batch_size)
    batch = Array{eltype(tree), 1}(undef, batch_size)
    segment = total(tree) / batch_size

    a = convert(Array, 0:(batch_size-1))*segment
    b = convert(Array, 1:batch_size)*segment
    s = randInRange(a, b; rng=rng)
    for i in 1:batch_size
        (x,y,z) = get(tree, s[i])
        batch_idx[i] = x
        batch_priorities[i] = y
        batch[i] = z
    end

    return batch_idx, batch_priorities, batch

end
