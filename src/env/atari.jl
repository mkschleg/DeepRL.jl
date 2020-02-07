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
    reward_clip::Bool
    frameskip::Int
    color_averaging::Symbol
    gray_scale::Bool
    rawscreen::Vector{Cuchar}  # raw screen data from the most recent frame
    state::Array{S}  # the game state... 
    state_pre::Array{S} # the game state for doing max pooling...
end

function Atari{S}(gamename::AbstractString;
                  seed=0,
                  frameskip=5,
                  color_averaging=:none,
                  repeat_action_probability=0.25f0,
                  reward_clip=false,
                  gray_scale=false) where {S<:Number}

    @assert gamename ∈ ALE.getROMList()
    ale = ALE.ALE_new()
    ALE.setInt(ale, "random_seed", seed)
    # ALE.setInt(ale, "frame_skip", frameskip)
    ALE.setFloat(ale, "repeat_action_probability", repeat_action_probability)
    # ALE.setBool(ale, "color_averaging", color_averaging)
    ALE.loadROM(ale, gamename)
    w = ALE.getScreenWidth(ale)
    h = ALE.getScreenHeight(ale)
    
    rawscreen = if gray_scale
        Array{Cuchar}(undef, w * h)
    else
        Array{Cuchar}(undef, w * h * 3)
    end

    state = if gray_scale
        fill(zero(S), h, w)
    else
        fill(zero(S), h, w, 3)
    end
    
    Atari{S}(ale, 0, false, 0., 0., 0, w, h, reward_clip, frameskip, color_averaging, gray_scale, rawscreen, state, copy(state))
end

Atari(gamename::AbstractString; kwargs...) = Atari{UInt8}(gamename; kwargs...)

function Base.close(env::Atari)
    # env.state = typeof(env.state)(undef, 0, 0, 0)
    ALE.ALE_del(env.ale)
end

RLCore.get_actions(env::Atari) = ALE.getMinimalActionSet(env.ale)
valid_action(env::Atari, action) = action in RLCore.get_actions(env)

get_colorview(env::Atari{S}, gray_scale::Val{true}) where {S<:AbstractFloat} = 
    Images.colorview(Images.Gray, env.state)
get_colorview(env::Atari{S}, gray_scale::Val{true}) where {S<:Integer} = 
    Images.colorview(Images.Gray, env.state./255)


get_colorview(env::Atari{S}, gray_scale::Val{false}) where {S<:AbstractFloat} = 
    Images.colorview(Images.RGB, permutedims(env.state, (3, 1, 2)))
get_colorview(env::Atari{S}, gray_scale::Val{false}) where {S<:Integer} = 
    Images.colorview(Images.RGB, permutedims(env.state, (3, 1, 2))./255)


@recipe function f(env::Atari)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1

    get_colorview(env, Val(env.gray_scale))
end

# Horrible code reuse... HAHAHA.
function update_pre_state!(env::Atari, gray_scale::Val{false})
    # get the raw screen data
    ALE.getScreenRGB!(env.ale, env.rawscreen)
    idx = 1
    if eltype(env.state) <: AbstractFloat
        permutedims!(env.state_pre, reshape(env.rawscreen ./ eltype(env.state)(255), (3, env.width, env.height)), (3,2,1)) 
    elseif eltype(env.state) <: Integer
        permutedims!(env.state_pre, reshape(env.rawscreen, (3, env.width, env.height)), (3,2,1))
    end
    env.lives = ALE.lives(env.ale)
    return env.state
end

function update_pre_state!(env::Atari, gray_scale::Val{true})
    # get the raw screen data
    ALE.getScreenGrayscale!(env.ale, env.rawscreen)
    idx = 1
    if eltype(env.state) <: AbstractFloat
        permutedims!(env.state_pre, reshape(env.rawscreen ./ eltype(env.state)(255), (env.width, env.height)), (2,1)) 
    elseif eltype(env.state) <: Integer
        permutedims!(env.state_pre, reshape(env.rawscreen, (env.width, env.height)), (2,1))
    end
    env.lives = ALE.lives(env.ale)
    return env.state
end

function update_state!(env::Atari, gray_scale::Val{false})
    # get the raw screen data
    ALE.getScreenRGB!(env.ale, env.rawscreen)
    idx = 1
    if eltype(env.state) <: AbstractFloat
        permutedims!(env.state, reshape(env.rawscreen ./ eltype(env.state)(255), (3, env.width, env.height)), (3,2,1)) 
    elseif eltype(env.state) <: Integer
        permutedims!(env.state, reshape(env.rawscreen, (3, env.width, env.height)), (3,2,1))
    end
    env.lives = ALE.lives(env.ale)
    return env.state
end

function update_state!(env::Atari, gray_scale::Val{true})
    # get the raw screen data
    ALE.getScreenGrayscale!(env.ale, env.rawscreen)
    idx = 1
    if eltype(env.state) <: AbstractFloat
        permutedims!(env.state, reshape(env.rawscreen ./ eltype(env.state)(255), (env.width, env.height)), (2,1)) 
    elseif eltype(env.state) <: Integer
        permutedims!(env.state, reshape(env.rawscreen, (env.width, env.height)), (2,1))
    end
    env.lives = ALE.lives(env.ale)
    return env.state
end

# Set seed default to 0
function RLCore.start!(env::Atari; kwargs...)
    # ALE.setInt(env.ale, "random_seed", seed)
    ALE.reset_game(env.ale)
    env.lives = 0
    env.died = false
    env.reward = 0
    env.score = 0
    env.nframes = 0
    update_state!(env, Val(env.gray_scale))
end

RLCore.start!(env::Atari, rng::AbstractRNG; kwargs...) = RLCore.start!(env; kwargs...)

function RLCore.environment_step!(env::Atari{S}, action; kwargs...) where {S<:Number}
    # act and get the reward and new state

    # rew = zeros(Float32, env.frameskip)
    env.reward = 0.0f0
    for i ∈ 1:env.frameskip
        env.reward += ALE.act(env.ale, action)
        if i == (env.frameskip - 1)
            update_pre_state!(env, Val(env.gray_scale))
        elseif i == (env.frameskip)
            update_state!(env, Val(env.gray_scale))
        end
    end

    if env.color_averaging == :max
        env.state .= max.(env.state, env.state_pre)
    elseif env.color_averaging == :average
        env.state .= S.(round.(mean([env.state, env.state_pre])))
    end

    env.score += env.reward
end

RLCore.get_reward(env::Atari) = env.reward 
RLCore.is_terminal(env::Atari) = ALE.game_over(env.ale)
RLCore.get_state(env::Atari) = env.state

function image_manip_atari(s::Array{UInt8, 3})
    Images.colorview(RGB{Images.N0f8}, permutedims(s, (3, 1, 2))) |> # Colorview of array
        (img)->Gray{Images.N0f8}.(Images.imresize(img, (84,84))) |> # Resize image to 84x84
        (Images.rawview ∘ Images.channelview)
end

function image_manip_atari(s::Array{UInt8, 2})
    round.(Images.imresize(Float32.(s), (84,84)))
end

