
using LinearAlgebra

using Flux
using Flux.Zygote: dropgrad

import Statistics: mean



"""
    smooth_l1_loss(y, fx; δ)

l1 loss w/ clipping. Also known as the huber loss. Only usable w/ numbers (see `huber_loss` for vector form)
"""
function smooth_l1_loss(y::Number, fx::Number, δ=1)
    α = abs(y - fx)
    α <= δ && return 0.5f0 * α ^ 2
    δ * α - (0.5f0 * δ ^ 2)
end

"""
    huber_loss(y, fx; δ)

Huber loss. Convenience for smoot_l1_loss
"""
function huber_loss(y, fx; δ=1)
    return smooth_l1_loss.(y, fx, δ)
end

"""
    mean_huber_loss(y, fx; δ)

mean huber loss.
"""
function mean_huber_loss(y, fx; δ=1)
    return mean(huber_loss(y, fx; δ=δ))
end


get_cart_idx(a, l) = CartesianIndex.(a, 1:l)
Flux.Zygote.@nograd get_cart_idx


function loss(lu::QLearning, model, s_t, a_t, s_tp1, r, terminal, target_model)
    
    action_idx = get_cart_idx(a_t, length(terminal))
    q_tp1 = if target_model isa Nothing
        dropgrad(maximum(model(s_tp1); dims=1)[1, :])
    else
        dropgrad(maximum(target_model(s_tp1); dims=1)[1, :])
    end
    q_t = @view model(s_t)[action_idx]
    targets = dropgrad(r .+ (1 .- terminal).*lu.γ.*q_tp1)
    return lu.loss(q_t, targets)
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
    
    return lu.loss(q_t, target)
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
