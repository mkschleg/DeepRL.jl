
using Flux
using LinearAlgebra
using CuArrays

using Flux.Zygote: dropgrad


get_cart_idx(a, l) = CartesianIndex.(a, 1:l)
Flux.Zygote.@nograd get_cart_idx


abstract type AbstractLearningUpdate end

function update!(model,
                 lu::T,
                 opt,
                 s_t::Array{<:AbstractArray, 1},
                 a_t,
                 s_tp1::Array{<:AbstractArray, 1},
                 r,
                 terminal,
                 args...) where {T<:AbstractLearningUpdate}
    update!(model, lu, opt, cat(s_t...; dims=2), a_t, cat(s_tp1...; dims=2), r, terminal, args...)
end


abstract type AbstractQLearning <: AbstractLearningUpdate end

struct QLearning <: AbstractQLearning
    γ::Float32
end

function loss(lu::QLearning, model, s_t, a_t, s_tp1, r, terminal, target_model)
    
    action_idx = get_cart_idx(a_t, length(terminal))
    q_tp1 = if target_model isa Nothing
        dropgrad(maximum(model(s_tp1); dims=1)[1, :])
    else
        dropgrad(maximum(target_model(s_tp1); dims=1)[1, :])
    end
    q_t = @view model(s_t)[action_idx]
    
    return Flux.mse(q_t, dropgrad(r .+ (1 .- terminal).*lu.γ.*q_tp1))
end


struct DoubleQLearning <: AbstractQLearning
    γ::Float32
end

function loss(lu::DoubleQLearning, model, s_t, a_t, s_tp1, r, terminal, target_model)
    γ = lu.γ.*(1 .- terminal)

    action_idx = get_cart_idx(a_t, length(terminal))

    q̃_tp1 = dropgrad(model(s_tp1))
    q̃_tp1_argmax = dropgrad(findmax(q̃_tp1; dims=1))
    action_tp1 = [q̃_tp1_argmax[2][i] for i in 1:length(terminal)]

    q_tp1 = if target_model isa Nothing
        dropgrad(model(s_tp1)[action_tp1])
    else
        dropgrad(target_model(s_tp1)[action_tp1])
    end
    
    target = dropgrad(r .+ γ.*q_tp1)
    q_t = model(s_t)[action_idx]
    
    return Flux.mse(q_t, target)
end


function update!(model,
                 lu::LU,
                 opt,
                 s_t::A,
                 a_t,
                 s_tp1::A,
                 r,
                 terminal,
                 target_model) where {LU<:AbstractQLearning, F<:AbstractFloat, A<:AbstractArray{F}}

    ps = params(model)
    l = 0.0f0
    gs = gradient(ps) do
        l = loss(lu, model, s_t, a_t, s_tp1, r, terminal, target_model)
    end
    Flux.Optimise.update!(opt, ps, gs)
    return l
end

struct TDLearning end

function loss(lu::TDLearning, model, s_t, a_t, s_tp1, r, terminal, target_model, horde::H) where {H<:Horde}

    # get GVF horde parameters

    p = [RLCore.get(horde, s_t[:,i], a_t, s_tp1[:,i]) for i in 1:length(terminal)]

    c = hcat(getindex.(p, 1)...)
    γ = hcat(getindex.(p, 2)...)
    π = hcat(getindex.(p, 3)...)

    v_t = model(s_t)
    v_tp1 = target_model(s_tp1)

    target = dropgrad(c .+ γ.*v_tp1) # -> Matrix (preds × batch_size)

    return sum((target .- v_t).^2) * (1 // length(terminal))
end

loss(lu::TDLearning, model, s_t, a_t, s_tp1, r, terminal, target_model::Nothing, horde::H) where {H<:Horde} = 
    loss(lu, model, s_t, a_t, s_tp1, r, terminal, (x)->Flux.data(model(x)), horde)


struct AuxQLearning{T<:AbstractQLearning} <: AbstractLearningUpdate
    β::Float32
    q_learning::T
    td_learning::TDLearning
end

function update!(
    model,
    lu::AQL,
    opt,
    s_t::Array{<:AbstractFloat, 2},
    a_t::Array{<:Integer, 1},
    s_tp1::Array{<:AbstractFloat, 2},
    r::Array{<:AbstractFloat, 1},
    terminal,
    target_model,
    horde::H) where {AQL<:AuxQLearning, H<:RLCore.Horde}

    num_gvfs = length(horde)
    ps = params(model)

    ℒ = 0.0f0

    
    gs = Flux.gradient(ps) do
        ℒ_q = loss(lu.q_learning, (x)->model(x)[1:(end-num_gvfs), :], s_t, a_t, s_tp1, r, terminal, (x)->target_model(x)[1:(end-num_gvfs), :])
        # @show ℒ_q
        ℒ_td = loss(lu.td_learning, (x)->model(x)[(end-num_gvfs+1):end, :], s_t, a_t, s_tp1, r, terminal, (x)->target_model(x)[(end-num_gvfs+1):end, :], horde)
        # @show ℒ_td
        ℒ = ℒ_q  + lu.β*ℒ_td
    end
    Flux.Optimise.update!(opt, ps, gs)
    return ℒ
end


function update!(
    model,
    lu::AuxQLearning,
    opt,
    s_t::Array{<:AbstractFloat, 2},
    a_t::Array{<:Integer, 1},
    s_tp1::Array{<:AbstractFloat, 2},
    r::Array{<:AbstractFloat, 1},
    terminal,
    target_model,
    horde::Nothing)
    update!(model, lu.q_learning, opt, s_t, a_t, s_tp1, r, terminal, target_model)
end
