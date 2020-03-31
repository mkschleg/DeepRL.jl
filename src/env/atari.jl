import MinimalRLCore

using RecipesBase
import Images
using Images: permutedims, permutedims!

import ArcadeLearningEnvironment
const ALE = ArcadeLearningEnvironment


"""
    Atari(gamename; seed, frameskip, color_averaging, repeat_action_probabiliyt, gray_scale)

An interface adapted from https://github.com/JuliaML/AtariAlgos.jl/blob/master/src/AtariAlgos.jl with a backend 
implemented by https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl. Because we want to have
some better fidelity with settings, reimplementing is easier than writting a wrapper around a wrapper around a wrapper...

Many current implementations of these utilities in Python use a compositional design pattern. I find this often 
obfuscates what exactly is going on and forces users to travers several classes and types to understand the code. 
Here we've decided to take a more procedural approach. The end user can still wrap this type to provide their own 
settings or use this type as a template for their own implementation.

Use cases found in the literature not yet implemented:
- Terminate on life lost
- 

## Arguments
- `gamename::AbstractString`: Name of game (should be found in rom list.)
- `seed::Int32`: The seed passed to the ALE environment. This needs to be a 32 bit integer for ALE.
- `frameskip`: The number of frames skipped between actions. If set to 1, `color_averaging` must be set to `:none`
- `color_averaging`: Can be any of the set `(:none, :max, :average)`
- `gray_scale`: If true this will return the gray scale images provided by the ALE library. If false the full color images will be returned.
"""
mutable struct Atari <: MinimalRLCore.AbstractEnvironment
    # Pointer to ALE
    ale::ALE.ALEPtr

    # Details of current episode
    lives::Int
    died::Bool
    reward::Float64
    score::Float64

    # settings
    gamename::String
    frameskip::Int
    color_averaging::Symbol
    gray_scale::Bool

    # details needed for constructing the base state.
    width::Int
    height::Int
    rawscreen::Vector{Cuchar}  # raw screen data from the most recent frame
    state_buffer::Array{Array{UInt8}, 1}
end

function Atari(gamename::AbstractString,
               seed,
               frameskip,
               color_averaging,
               repeat_action_probability,
               gray_scale,
               terminal_on_life=false)

    @assert gamename ∈ ALE.getROMList()
    @assert color_averaging ∈ (:none, :max, :average)
    @assert frameskip >= 1
    @assert !(frameskip == 1 && color_averaging == :none)

    ale = ALE.ALE_new()
    ALE.setInt(ale, "random_seed", UInt16(seed))
    ALE.setFloat(ale, "repeat_action_probability", Float32(repeat_action_probability))
    ALE.loadROM(ale, gamename)
    
    w = ALE.getScreenWidth(ale)
    h = ALE.getScreenHeight(ale)
    
    rawscreen = if gray_scale
        Array{Cuchar}(undef, w * h)
    else
        Array{Cuchar}(undef, w * h * 3)
    end

    state = if gray_scale
        [fill(zero(UInt8), h, w), fill(zero(UInt8), h, w)]
    else
        [fill(zero(UInt8), h, w, 3), fill(zero(UInt8), h, w, 3)]
    end
    
    Atari(ale, 0, false, 0.0, 0.0, gamename, frameskip, color_averaging, gray_scale, w, h, rawscreen, state)
end

Base.show(io::IO, env::Atari) =
    println(io, "$(env.gamename)(frameskip=$(env.frameskip), color_averaging=$(env.color_averaging), gray_scale=$(env.gray_scale))")


# Various default versions of Atari found in literature
"""
    RevisitingALEAtari(gamename, seed)

This is the settings used in "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents." by Marlos Machado et. al. See https://arxiv.org/abs/1709.06009.

"""
RevisitingALEAtari(gamename, seed) =
    Atari(gamename, seed, 5, :max, 0.25f0, true)



# Some utility functions
function Base.close(env::Atari)
    ALE.ALE_del(env.ale)
end

MinimalRLCore.get_actions(env::Atari) = ALE.getLegalActionSet(env.ale)
valid_action(env::Atari, action) = action in MinimalRLCore.get_actions(env)
get_minimal_actions(env::Atari) = ALE.getMinimalActionSet(env.ale)


function update_state!(env::Atari, t)
    # get the raw screen data

    if env.gray_scale
        ALE.getScreenGrayscale!(env.ale, env.rawscreen)
        permutedims!(env.state_buffer[t], reshape(env.rawscreen, (env.width, env.height)), (2,1))
    else
        ALE.getScreenRGB!(env.ale, env.rawscreen)
        permutedims!(env.state_buffer[t], reshape(env.rawscreen, (3, env.width, env.height)), (3,2,1))
    end
    env.lives = ALE.lives(env.ale)
    return env.state_buffer[t]
end

function _pool_state(env::Atari)
    if env.color_averaging == :max
        env.state_buffer[2] .= max.(env.state_buffer[1], env.state_buffer[2])
    elseif env.color_averaging == :average
        env.state_buffer[2] .= UInt8.(round.(mean([env.state_buffer[1], env.state_buffer[2]])))
    end
end

# Set seed default to 0
function MinimalRLCore.start!(env::Atari)
    ALE.reset_game(env.ale)
    env.lives = 0
    env.died = false
    env.reward = 0
    env.score = 0
    update_state!(env, 2)
end

MinimalRLCore.start!(env::Atari, rng::AbstractRNG) = MinimalRLCore.start!(env; kwargs...)

function MinimalRLCore.environment_step!(env::Atari, action)

    env.reward = 0.0f0
    for i ∈ 1:env.frameskip
        env.reward += ALE.act(env.ale, action)
        if (env.frameskip - i) < 2
            update_state!(env, env.frameskip - i + 1)
        end
    end

    env.score += env.reward
end

MinimalRLCore.get_reward(env::Atari) = env.reward 
MinimalRLCore.is_terminal(env::Atari) = ALE.game_over(env.ale)
MinimalRLCore.get_state(env::Atari) = env.state_buffer[2]


"""
    image_manip_atari(s)

Helper functions for doing image pre-processing. This does the standard thing of turing the image gray scale 
(if necessary) and resizes the image too 84*84.

As a note, Images.Gray does the conversion to Luminance as done in the original DQN paper (`0.299 * r + 0.587 * g + 0.114 * b`) 
but then rounds to fit in a UInt8. It is unclear if this effects the final performance of the DQN or not... (which is a bit dissapointing).

"""
function image_manip_atari(s::Array{UInt8, 3})
    Images.colorview(Images.RGB{Images.N0f8}, permutedims(s, (3, 1, 2))) |> # Colorview of array
        (img)->Images.Gray{Images.N0f8}.(Images.imresize(img, (84,84))) |> # Resize image to 84x84
        (Images.rawview ∘ Images.channelview)
end

function image_manip_atari(s::Array{UInt8, 2})
    Images.imresize(Images.Gray{Images.N0f8}.(s./UInt8(255)), (84,84)) |>
        (Images.rawview ∘ Images.channelview)
end

image_norm(img) = img./255f0

#####
# Visualization through RecipeBase.jl
#####

function get_colorview(env::Atari, gray_scale::Val{true})
    Images.colorview(Images.Gray, permutedims(reshape(env.rawscreen, (env.width, env.height)), (2,1))./255)
end

get_colorview(env::Atari, gray_scale::Val{false}) = 
    Images.colorview(Images.RGB, permutedims(reshape(env.rawscreen, (3, env.width, env.height)), (1,3,2))./255)

get_colorview_image_manip(env::Atari, gray_scale::Val{true}) =
    Images.Gray.(image_norm(image_manip_atari(MinimalRLCore.get_state(env))))

get_colorview_image_manip(env::Atari, gray_scale::Val{false}) = 
    Images.Gray.(image_norm(Images.colorview(Images.Gray, image_manip_atari(MinimalRLCore.get_state(env)))))

@recipe function f(env::Atari, resize=false)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    if resize
        get_colorview_image_manip(env, Val(env.gray_scale))
    else
        get_colorview(env, Val(env.gray_scale))
    end
end
