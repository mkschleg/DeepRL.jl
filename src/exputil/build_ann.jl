
import Flux: Flux, outputsize
import Flux: Dense

function get_activation(act::AbstractString)
    if act == "sigmoid_fast"
        return Flux.sigmoid_fast
    elseif act == "tanh_fast"
        return Flux.tanh_fast
    elseif act == "sigmoid"
        return Flux.sigmoid
    elseif act == "tanh"
        return Flux.tanh
    elseif act == "linear"
        return Flux.identity
    elseif act == "relu"
        return Flux.relu
    elseif act == "celu"
        return Flux.relu
    elseif act == "softplus"
        return Flux.softplus
    else
        throw("$(act) not known...")
    end
end

"""
    get_kwarg_config

A helper function which gets the kwarg associated with name.
"""
function get_kwarg_config(name::AbstractString, kwargs)
    sym_name = Symbol(name)
    @assert sym_name ∈ keys(kwargs) "Expected keyword argument \"$(name)\"."
    kwargs[sym_name]
end

function get_param_kwarg_or_config(name, kwargs, params_config)
    sym_name = Symbol(name)
    if sym_name ∈ keys(kwargs)
        kwargs[sym_name]
    elseif !isnothing(params_config) && name ∈ keys(params_config)
        params_config[name]
    else
        throw("Expected $(name) to be in kwargs or params_confg.")
    end
                      
end

function get_init_func(config, rng)
    init_name = config["init"]
    initialize_init_func(Val(Symbol(init_name)), rng)
end

function initialize_init_func(type, rng)
    throw("Implement initialize_init_func for $(typeof(type).parameters[1])")
end

function initialize_init_func(::Val{:glorot_uniform}, rng)
    Flux.glorot_uniform(rng)
end

function initialize_init_func(::Val{:glorot_normal}, rng)
    Flux.glorot_normal(rng)
end

function initialize_init_func(::Val{:kaiming_uniform}, rng)
    Flux.kaiming_uniform(rng)
end

function initialize_init_func(::Val{:kaiming_normal}, rng)
    Flux.kaiming_normal(rng)
end



function build_ann_from_config(in, nn_config, params_config; kwargs...)
    # The config will be a list.
    layers = []
    for cl in nn_config
        l = build_layer_from_config(in, cl; params_config=params_config, kwargs...)
        push!(layers, l)
        in = outputsize(l, in)
    end
    return Flux.Chain(layers...)
end

function build_layer_from_config(in, config; kwargs...)
    layer_type = config["type"]
    fw = Base.get(config, "framework", "Flux") #get init func?
    init = if !isnothing(Base.get(kwargs, :init, nothing)) && "init" ∈ keys(config)
        throw("Ambiguous constructions: Can't use both kwarg init and config init when building a layer.")
    elseif "init" ∈ keys(config)
        if :rng ∉ keys(kwargs)
            @warn "build_layer_from_confg: Creating init func with default rng for layer. Pass in rng to build_ann_from_config to avoid."
        end
        rng = get(kwargs, :rng, Random.default_rng())
        get_init_func(config, rng)
    elseif !isnothing(Base.get(kwargs, :init, nothing))
        kwargs[:init]
    else
        throw("Need to specify an init function either in the config or through a kwarg")
    end
    build_layer(Val(Symbol(fw)), Val(Symbol(layer_type)), in, config; kwargs...)
end


function build_layer(fw_name, layer_type, config; rng=Random.default_rng(), kwargs...)
    throw("Implement build_layer for combination: $(typeof(fw_name).parameters[1]), $(typeof(layer_type).parameters[1])")
end

function build_layer(::Val{:Flux}, ::Val{:Dense}, in, config; rng=Random.default_rng(), init=get_init_func(config, rng), params_config=nothing, kwargs...)
    @assert "bias" ∈ keys(config) "You need to specificy whether to use a bias unit for Dense"
    @assert "out" ∈ keys(config) "You need to specificy the out size for Dense"
    @assert "activation" ∈ keys(config) "You need to specificy an activation for Dense"

    out = if config["out"] isa String
        get_param_kwarg_or_config(config["out"], kwargs, params_config)
    else
        config["out"]
    end
    
    # Dense only cares about the first index of in.
    Dense(in[1]=>out, get_activation(config["activation"]); bias=config["bias"], init=init)
end



