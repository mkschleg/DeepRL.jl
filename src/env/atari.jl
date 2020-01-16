import RLCore

using Plots
import Images
using Images: permutedims, permutedims!

import ArcadeLearningEnvironment
const ALE = ArcadeLearningEnvironment


"""
    Atari

An interface adapted from https://github.com/JuliaML/AtariAlgos.jl/blob/master/src/AtariAlgos.jl with a backend 
implemented by https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl. Because we want to have
some better fidelity with settings, reimplementing is easier than writting a wrapper around a wrapper around a wrapper...
"""
mutable struct Atari{S} <: RLCore.AbstractEnvironment
    ale::ALE.ALEPtr
    lives::Int
    died::Bool
    reward::Float64
    score::Float64
    nframes::Int
    width::Int
    height::Int
    rawscreen::Vector{Cuchar}  # raw screen data from the most recent frame
    state::Array{S, 3}  # the game state... raw screen data converted to Float64


end

function Atari{S}(gamename::AbstractString; seed=0, frameskip=1) where {S<:Number}
    ale = ALE.ALE_new()
    ALE.setInt(ale, "random_seed", seed)
    ALE.setInt(ale, "frame_skip", frameskip)
    ALE.loadROM(ale, gamename)
    w = ALE.getScreenWidth(ale)
    h = ALE.getScreenHeight(ale)
    rawscreen = Array{Cuchar}(undef, w * h * 3)
    state = fill(zero(S), h, w, 3)
    Atari{S}(ale, 0, false, 0., 0., 0, w, h, rawscreen, state)
end

Atari(gamename::AbstractString; kwargs...) = Atari{UInt8}(gamename; kwargs...)

function Base.close(env::Atari)
    env.state = typeof(env.state)(undef, 0, 0, 0)
    ALE.ALE_del(env.ale)
end

RLCore.get_actions(env::Atari) = ALE.getMinimalActionSet(env.ale)
valid_action(env::Atari, action) = action in RLCore.get_actions(env)

@recipe function f(env::Atari)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1

    if eltype(env.state) <: AbstractFloat
        Images.colorview(Images.RGB, permutedims(env.state, (3, 1, 2)))
    elseif eltype(env.state) <: Integer
        Images.colorview(Images.RGB{Images.N0f8}, permutedims(env.state, (3, 1, 2)))
    end
end

function update_state!(env::Atari)
    # get the raw screen data
    ALE.getScreenRGB!(env.ale, env.rawscreen)
    idx = 1
    if eltype(env.state) <: AbstractFloat
        permutedims!(env.state, reshape(env.rawscreen .// 256, (3, env.width, env.height)), (3,2,1)) 
    elseif eltype(env.state) <: Integer
        permutedims!(env.state, reshape(env.rawscreen, (3, env.width, env.height)), (3,2,1))
    end
    env.lives = ALE.lives(env.ale)
    return
end

# Set seed default to 0
function RLCore.reset!(env::Atari, rng::AbstractRNG; kwargs...)
    # ALE.setInt(env.ale, "random_seed", seed)
    ALE.reset_game(env.ale)
    env.lives = 0
    env.died = false
    env.reward = 0
    env.score = 0
    env.nframes = 0
    update_state!(env)
end

function RLCore.environment_step!(env::Atari, action; kwargs...)
    # act and get the reward and new state
    env.reward = ALE.act(env.ale, action)
    env.score += env.reward
    update_state!(env)
    return
end

RLCore.get_reward(env::Atari) = env.reward 
RLCore.is_terminal(env::Atari) = ALE.game_over(env.ale)
RLCore.get_state(env::Atari) = env.state

function image_manip_atari(s::Array{UInt8, 3})
    Images.colorview(RGB{Images.N0f8}, permutedims(s, (3, 1, 2))) |> # Colorview of array
        (img)->Gray{Images.N0f8}.(Images.imresize(img, (84,84))) |> # Resize image to 84x84
        (Images.rawview ∘ Images.channelview)
end

