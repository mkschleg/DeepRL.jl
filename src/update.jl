
abstract type AbstractLearningUpdate end


function update!(model,
                 lu::T,
                 opt,
                 s_t::Array{Array{AF, 1}, 1},
                 a_t::Array{<:Integer, 1},
                 s_tp1::Array{Array{AF, 1}, 1},
                 r::Array{AF, 1},
                 terminal,
                 args...) where {AF<:AbstractFloat, T<:AbstractLearningUpdate}
    update!(model, lu, opt, hcat(s_t...), a_t, hcat(s_tp1...), r, terminal, args...)
end

# function update!(model,
#                  lu::T,
#                  opt,
#                  s_t::Array{Array{AF, 1}, 1},
#                  a_t::Array{<:Integer, 1},
#                  s_tp1::Array{Array{AF, 1}, 1},
#                  r::Array{AF, 1},
#                  terminal) where {AF<:AbstractFloat, T<:AbstractLearningUpdate}
#     update!(model, lu, opt, hcat(s_t...), a_t, hcat(s_tp1...), r, terminal)
# end


# function update!(model::GVFNetwork,
#                  lu, opt, s_t, a_t, s_tp1, r, terminal)
#     update!(model.gvf_model, lu[1], opt, model.horde, s_t, a_t, s_tp1, r, terminal)
#     update!(model, lu[2], opt, s_t, a_t, s_tp1, r, terminal)
# end

abstract type AbstractQLearning <: AbstractLearningUpdate end


struct QLearning <: AbstractQLearning
    γ::Float32
end

function loss(lu::QLearning, model, s_t, a_t, s_tp1, r, terminal, target_model)
    γ = lu.γ.*(1 .- terminal)
    action_idx = [CartesianIndex(a_t[i], i) for i in 1:length(terminal)]

    q_tp1 = maximum(target_model(s_tp1); dims=1)[1,:]
    
    target = (r .+ γ.*q_tp1)
    q_t = model(s_t)[action_idx]
    return Flux.mse(target, q_t)
end

function loss(lu::QLearning, model, s_t, a_t, s_tp1, r, terminal, target_model::Nothing)
    γ = lu.γ.*(1 .- terminal)
    action_idx = [CartesianIndex(a_t[i], i) for i in 1:length(terminal)]

    q_tp1 = Flux.Tracker.data(maximum(model(s_tp1); dims=1)[1,:])

    target = (r .+ γ.*q_tp1)
    q_t = model(s_t)[action_idx]
    return Flux.mse(target, q_t)
end


struct DoubleQLearning <: AbstractQLearning
    γ::Float32
end

function loss(lu::DoubleQLearning, model, s_t, a_t, s_tp1, r, terminal, target_model)
    γ = lu.γ.*(1 .- terminal)
    action_idx = [CartesianIndex(a_t[i], i) for i in 1:length(terminal)]

    q̃_tp1 = Flux.data(model(s_tp1))
    q̃_tp1_argmax = findmax(q̃_tp1; dims=1)
    action_tp1 = [q̃_tp1_argmax[2][i] for i in 1:length(terminal)]
    q_tp1 = target_model(s_tp1)[action_tp1]

    target = (r .+ γ.*q_tp1)
    q_t = model(s_t)[action_idx]
    
    return Flux.mse(target, q_t)
end

function loss(lu::DoubleQLearning, model, s_t, a_t, s_tp1, r, terminal, target_model::Nothing)
    γ = lu.γ.*(1 .- terminal)
    action_idx = [CartesianIndex(a_t[i], i) for i in 1:length(terminal)]

    q̃_tp1 = Flux.data(model(s_tp1))
    q̃_tp1_argmax = findmax(q̃_tp1; dims=1)
    action_tp1 = [q̃_tp1_argmax[2][i] for i in 1:length(terminal)]
    q_tp1 = Flux.data(model(s_tp1)[action_tp1])

    target = (r .+ γ.*q_tp1)
    q_t = model(s_t)[action_idx]
    
    return Flux.mse(target, q_t)
end

function update!(model, lu::LU, opt,
                 s_t::Array{<:AbstractFloat, 2},
                 a_t::Array{<:Integer, 1},
                 s_tp1::Array{<:AbstractFloat, 2},
                 r::Array{<:AbstractFloat, 1},
                 terminal,
                 target_model) where {LU<:AbstractQLearning}

    ps = params(model)
    gs = Flux.gradient(ps) do
        loss(lu, model, s_t, a_t, s_tp1, r, terminal, target_model)
    end
    Flux.Optimise.update!(opt, ps, gs)
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

    target = c .+ γ.*v_tp1 # -> Matrix (preds × batch_size)

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

    gs = Flux.gradient(ps) do
        ℒ_q = loss(lu.q_learning, (x)->model(x)[1:(end-num_gvfs), :], s_t, a_t, s_tp1, r, terminal, (x)->target_model(x)[1:(end-num_gvfs), :])
        # @show ℒ_q
        ℒ_td = loss(lu.td_learning, (x)->model(x)[(end-num_gvfs+1):end, :], s_t, a_t, s_tp1, r, terminal, (x)->target_model(x)[(end-num_gvfs+1):end, :], horde)
        # @show ℒ_td
        return ℒ_q  + lu.β*ℒ_td
    end
    Flux.Optimise.update!(opt, ps, gs)
    
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
