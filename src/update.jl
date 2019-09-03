
abstract type LearningUpdate end


function update!(model,
                 lu::T,
                 opt,
                 s_t::Array{Array{AF, 1}, 1},
                 a_t::Array{<:Integer, 1},
                 s_tp1::Array{Array{AF, 1}, 1},
                 r::Array{AF, 1},
                 terminal::Array{Bool, 1},
                 target_model) where {AF<:AbstractFloat, T<:LearningUpdate}
    update!(model, lu, opt, hcat(s_t...), a_t, hcat(s_tp1...), r, terminal, target_model)
end

function update!(model,
                 lu::T,
                 opt,
                 s_t::Array{Array{AF, 1}, 1},
                 a_t::Array{<:Integer, 1},
                 s_tp1::Array{Array{AF, 1}, 1},
                 r::Array{AF, 1},
                 terminal::Array{Bool, 1}) where {AF<:AbstractFloat, T<:LearningUpdate}
    update!(model, lu, opt, hcat(s_t...), a_t, hcat(s_tp1...), r, terminal)
end


struct QLearning <: LearningUpdate
    γ::Float32
end

struct DoubleQLearning <: LearningUpdate
    γ::Float32
end

function update!(model, lu::QLearning, opt,
                 s_t::Array{<:AbstractFloat, 2},
                 a_t::Array{<:Integer, 1},
                 s_tp1::Array{<:AbstractFloat, 2},
                 r::Array{<:AbstractFloat, 1},
                 terminal::Array{Bool, 1},
                 target_model)

    ps = params(model)
    γ = lu.γ.*(1 .- terminal)
    action_idx = [CartesianIndex(a_t[i], i) for i in 1:length(terminal)]

    q_tp1 = maximum(target_model(s_tp1); dims=1)[1,:]
    
    target = (r .+ γ.*q_tp1)
    gs = Flux.gradient(ps) do
        q_t = model(s_t)[action_idx]
        return Flux.mse(target, q_t)
    end
    Flux.Optimise.update!(opt, ps, gs)
end

function update!(model, lu::QLearning, opt,
                 s_t::Array{<:AbstractFloat, 2},
                 a_t::Array{<:Integer, 1},
                 s_tp1::Array{<:AbstractFloat, 2},
                 r::Array{<:AbstractFloat, 1},
                 terminal::Array{Bool, 1})
    
    ps = params(model)
    
    γ = lu.γ.*(1 .- terminal)
    action_idx = [CartesianIndex(a_t[i], i) for i in 1:length(terminal)]

    q_tp1 = Flux.Tracker.data(maximum(model(s_tp1); dims=1)[1,:])
    
    target = (r .+ γ.*q_tp1)
    gs = Flux.gradient(ps) do
        q_t = model(s_t)[action_idx]
        return Flux.mse(target, q_t)
    end
    Flux.Optimise.update!(opt, ps, gs)

end

function update!(model, lu::DoubleQLearning, opt,
                 s_t::Array{<:AbstractFloat, 2},
                 a_t::Array{<:Integer, 1},
                 s_tp1::Array{<:AbstractFloat, 2},
                 r::Array{<:AbstractFloat, 1},
                 terminal::Array{Bool, 1})

    ps = params(model)
    
    γ = lu.γ.*(1 .- terminal)
    action_idx = [CartesianIndex(a_t[i], i) for i in 1:length(terminal)]

    q̃_tp1 = Flux.data(model(s_tp1))
    q̃_tp1_argmax = findmax(q̃_tp1; dims=1)
    action_tp1 = [q̃_tp1_argmax[2][i] for i in 1:length(terminal)]
    q_tp1 = target_model(s_tp1)[action_tp1]

    target = (r .+ γ.*q_tp1)
    gs = Flux.gradient(ps) do
        q_t = model(s_t)[action_idx]
        return Flux.mse(target, q_t)
    end
    Flux.Optimise.update!(opt, ps, gs)
end
