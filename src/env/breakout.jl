import JuliaRL

import ArcadeLearningEnvironment
const ALE = ArcadeLearningEnvironment


"""
    Atari

An interface adapted from https://github.com/JuliaML/AtariAlgos.jl/blob/master/src/AtariAlgos.jl with a backend implemented by https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl
"""
mutable struct Atari <: JuliaRL.AbstractEnvironment
    ale::ALE.ALEPtr
    lives::Int
    died::Bool
    reward::Float64
    score::Float64
    frame_skip::Int64
    nframes::Int
    width::Int
    height::Int
    rawscreen::Vector{Cuchar}  # raw screen data from the most recent frame
    state::Vector{Float64}  # the game state... raw screen data converted to Float64
    screen::Matrix{RGB{Float64}}

    function Atari(gamename::AbstractString; frame_skip=4)
        @assert frame_skip >= 1
        ale = ALE.ALE_new()
        ALE.loadROM(ale, gamename)
        w = ALE.getScreenWidth(ale)
        h = ALE.getScreenHeight(ale)
        rawscreen = Array{Cuchar}(undef, w * h * 3)
        state = similar(rawscreen, Float64)
        screen = fill(RGB{Float64}(0,0,0), h, w)
        new(ale, 0, false, 0., 0., frame_skip, 0, w, h, rawscreen, state, screen)
    end
end

function Base.close(game::Atari)
    game.state = Closed
    ALE.ALE_del(game.ale)
end

JuliaRL.get_actions(env::Atari) = ALE.getLegalActionSet(env.ale)
valid_action(env::Atari, action) = action in JuliaRL.get_actions(env)


function update_screen(game::Atari)
    idx = 1
    for i in 1:game.height, j in 1:game.width
        game.screen[i,j] = RGB{Float64}(game.state[idx], game.state[idx+1], game.state[idx+2])
        idx += 3
    end
    game.screen
end

@recipe function f(game::Atari)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1

    # convert to Image
    update_screen(game)
end

function update_state(game::Atari)
    # get the raw screen data
    ALE.getScreenRGB!(game.ale, game.rawscreen)
    for i in eachindex(game.rawscreen)
        game.state[i] = game.rawscreen[i] / 256
    end
    game.lives = ALE.lives(game.ale)
    game.state
end

function set_seed!(env::Atari, seed=0)
    ALE.setInt(env.ale, seed)
end

# Set seed default to 0
function JuliaRL.reset!(env::Atari; seed::Int64=0, kwargs...)
    ALE.reset_game(env.ale)
    ALE.setInt(env.ale, seed)
    env.lives = 0
    env.died = false
    env.reward = 0
    env.score = 0
    env.nframes = 0
    update_state(env)
end

function JuliaRL.environment_step!(env::Atari,
                                   action;
                                   rng=Random.GLOBAL_RNG, kwargs...)
    # act and get the reward and new state
    for 1:env.frame_skip
        game.reward = ALE.act(game.ale, a)
        game.score += game.reward
        if JuliaRL.is_terminal(env)
            break
        end
    end
    update_state(game)
    return
end

JuliaRL.get_reward(env::Atari) = env.reward 
JuliaRL.is_terminal(env::Atari) = ALE.game_over(env.ale)
JuliaRL.get_state(env::Atari) = game.state
