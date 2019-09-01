
module FluxUtils

using Flux
using Flux.Tracker
import Reproduce: ArgParseSettings, @add_arg_table

glorot_uniform(rng::Random.AbstractRNG, T::Type, dims...) = (rand(rng, T, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, T::Type, dims...) = randn(rng, T, dims...) .* sqrt(2.0f0/sum(dims))

glorot_uniform(rng::Random.AbstractRNG, dims::Vararg{Int64}) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))

function rnn_settings!(as::ArgParseSettings)
    @add_arg_table as begin
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
        "--cell"
        help="Cell"
        default="RNN"
        "--numhidden"
        help="Number of hidden units in cell"
        arg_type=Int64
        default=6
    end
end

function opt_settings!(as::ArgParseSettings, prefix::AbstractString="")
    add_arg_table(as,
                  "--$(prefix)opt",
                  Dict(:help=>"Optimizer",
                       :default=>"Descent"),
                  "--$(prefix)optparams",
                  Dict(:help=>"Parameters",
                       :arg_type=>Float64,
                       :default=>[],
                       :nargs=>'+'))
end

function get_optimizer(parsed::Dict)
    kt = keytype(parsed)
    get_optimizer(parsed[kt("opt")], parsed[kt("optparams")])
end

function get_optimizer(opt_string::AbstractString, params)
    opt_func = getproperty(Flux, Symbol(opt_string))
    return opt_func(params...)
end

function clip(a)
    clamp.(a, 0.0, 1.0)
end

function clip(a::TrackedArray)
    track(clip, a)
end
Flux.Tracker.@grad function clip(a)
    return clip(Flux.data(a)), Δ -> Tuple(Δ)
end

const CUSTOM_ACTIVATIONS = {
    "sigmoid": Flux.σ,
    "tanh": tanh,
    "linear": Flux.identity,
    "clip": clip,
    "relu": Flux.relu,
    "softplus": Flux.softplus}

function get_activation(act::AbstractString)
    if act ∈ keys(activations)
        return CUSTOM_ACTIVATIONS[act]
    else
        getproperty(Flux, Symbol(act))
    end
end

struct layer_spec
    name::String
    args
    kwargs
end

function construct_layer(l)
    l_func = getproperty(Flux, Symbol(l[1]))
    return l_func(l[2]...; l[3]...)
end

function construct_layer(l::layer_spec)
    l_func = getproperty(Flux, Symbol(l.name))
    return l_func(l.args...; l.kwargs...)
end

function construct_network(layer_list::Array{Dict{String, Any}})

    

    
end



end
