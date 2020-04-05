
function DopamineDQNBaseline(env)
    # Replicate Dopamine's Results!
    
    γ=0.99
    batch_size=32
    buffer_size = 1000000
    tn_update_freq= 8000
    hist_length = 4
    update_freq = 4
    min_mem_size = 20000

    learning_rate = 0.00025
    momentum_term = 0.00
    squared_grad_term = 0.95
    min_grad_term = 1e-5

    example_state = MinimalRLCore.get_state(env)

    init_f = Flux.glorot_uniform
    
    model = Chain(
        Conv((8,8), hist_length=>32, relu, stride=4, init=init_f),
        Conv((4,4), 32=>64, relu, stride=2, init=init_f),
        Conv((3,3), 64=>64, relu, stride=1, init=init_f),
        (x)->reshape(x, :, size(x, 4)),
        Dense(3136, 512, relu, initW=init_f),
        Dense(512, length(get_actions(env)), identity, initW=init_f)) |> gpu

    target_network  = deepcopy(model)


    return DQNAgent(
        model,
        target_network,
        QLearningHuberLoss(γ),
        DeepRL.RMSPropTFCentered(learning_rate,
                                 squared_grad_term,
                                 momentum_term,
                                 min_grad_term),
        DeepRL.ϵGreedyDecay((1.0, 0.01), 250000, min_mem_size, get_actions(env)),
        buffer_size,
        hist_length,
        example_state,
        batch_size,
        tn_update_freq,
        update_freq,
        min_mem_size;
        hist_squeeze = Val{false}(),
        state_preproc = DeepRL.image_manip_atari,
        state_postproc = DeepRL.image_norm,
        rew_transform = (r)->clamp(Float32(r), -1.0f0, 1.0f0),
        device = Flux.use_cuda[] ? Val{:gpu}() : Val{:cpu}())
end

function RevisitingALEDQNBaseline(env)
    # Replicate Dopamine's Results!
    
    γ=0.99
    batch_size=32
    buffer_size = 1000000
    tn_update_freq= 10000
    hist_length = 4
    update_freq = 4
    min_mem_size = 50000

    learning_rate = 0.00025
    momentum_term = 0.00
    squared_grad_term = 0.95
    min_grad_term = 1e-5

    example_state = MinimalRLCore.get_state(env)

    init_f = Flux.glorot_uniform
    
    model = Chain(
        Conv((8,8), hist_length=>32, relu, stride=4, init=init_f),
        Conv((4,4), 32=>64, relu, stride=2, init=init_f),
        Conv((3,3), 64=>64, relu, stride=1, init=init_f),
        (x)->reshape(x, :, size(x, 4)),
        Dense(3136, 512, relu, initW=init_f),
        Dense(512, length(get_actions(env)), identity, initW=init_f)) |> gpu

    target_network  = deepcopy(model)


    return DQNAgent(
        model,
        target_network,
        QLearningHuberLoss(γ),
        DeepRL.RMSPropTFCentered(learning_rate,
                                 squared_grad_term,
                                 momentum_term,
                                 min_grad_term),
        DeepRL.ϵGreedyDecay((1.0, 0.01), 1000000, min_mem_size, get_actions(env)),
        buffer_size,
        hist_length,
        example_state,
        batch_size,
        tn_update_freq,
        update_freq,
        min_mem_size;
        hist_squeeze = Val{false}(),
        state_preproc = DeepRL.image_manip_atari,
        state_postproc = DeepRL.image_norm,
        rew_transform = (r)->clamp(Float32(r), -1.0f0, 1.0f0),
        device = Flux.use_cuda[] ? Val{:gpu}() : Val{:cpu}())
end
