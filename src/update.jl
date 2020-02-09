
using Flux
using LinearAlgebra


abstract type AbstractLearningUpdate end

abstract type AbstractQLearning <: AbstractLearningUpdate end

struct QLearning{F} <: AbstractQLearning
    γ::Float32 # Discoount
    loss::F
end
QLearningHuberLoss(γ) = QLearning(Float32(γ), mean_huber_loss)


struct DoubleQLearning{F} <: AbstractQLearning
    γ::Float32
    loss::F
end

DoubleQLearningHuberLoss(γ) = DoubleQLearning(Float32(γ), mean_huber_loss)

include("loss.jl")

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
    ℒ = 0.0f0
    gs = gradient(ps) do
        ℒ = loss(lu, model, s_t, a_t, s_tp1, r, terminal, target_model)
    end
    Flux.Optimise.update!(opt, ps, gs)
    return ℒ
end



# struct AuxTDLearning end

# struct AuxHordeQLearning{T<:AbstractQLearning} <: AbstractLearningUpdate
#     β::Float32
#     q_learning::T
#     td_learning::AuxHordeTDLearning
# end

# function update!(
#     model,
#     lu::AQL,
#     opt,
#     s_t::Array{<:AbstractFloat, 2},
#     a_t::Array{<:Integer, 1},
#     s_tp1::Array{<:AbstractFloat, 2},
#     r::Array{<:AbstractFloat, 1},
#     terminal,
#     target_model,
#     horde::RLCore.Horde) where {AQL<:AuxQLearning, H<:RLCore.Horde}

#     num_gvfs = length(horde)
#     ps = params(model)

#     ℒ = 0.0f0
    
#     gs = Flux.gradient(ps) do
#         ℒ_q = loss(lu.q_learning, (x)->model(x)[1:(end-num_gvfs), :], s_t, a_t, s_tp1, r, terminal, (x)->target_model(x)[1:(end-num_gvfs), :])
#         # @show ℒ_q
#         ℒ_td = loss(lu.td_learning, (x)->model(x)[(end-num_gvfs+1):end, :], s_t, a_t, s_tp1, r, terminal, (x)->target_model(x)[(end-num_gvfs+1):end, :], horde)
#         # @show ℒ_td
#         ℒ = ℒ_q  + lu.β*ℒ_td
#     end
#     Flux.Optimise.update!(opt, ps, gs)
#     return ℒ
# end


# function update!(
#     model,
#     lu::AuxQLearning,
#     opt,
#     s_t::Array{<:AbstractFloat, 2},
#     a_t::Array{<:Integer, 1},
#     s_tp1::Array{<:AbstractFloat, 2},
#     r::Array{<:AbstractFloat, 1},
#     terminal,
#     target_model,
#     horde::Nothing)
#     update!(model, lu.q_learning, opt, s_t, a_t, s_tp1, r, terminal, target_model)
# end
