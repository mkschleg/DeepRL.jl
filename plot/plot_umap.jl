
using Plots
using UMAP: umap
using DeepRL: FluxUtils
using Flux
using ProgressMeter
using BSON

function get_umap(model, data_range; kwargs...)

    features_map = collect.(collect(Iterators.product(data_range...)))
    features_map_mat = hcat(reshape(features_map, length(features_map), 1)...)

    activations = FluxUtils.get_activations(model, features_map_mat)
    mat_act = vcat(activations[2:end-1]...)

    # println(mat_act)

    umap(mat_act; kwargs...)
    
end



function get_umap_over_time(snapshot_loc, data_range, skip=1; kwargs...)

    model_files = readdir(snapshot_loc)
    filter!((f)->splitext(f)[2] == ".bson", model_files)
    num_models = length(model_files)

    model_order = sortperm(parse.(Int64, getindex.(split.(getindex.(splitext.(model_files), 1), "_"), 2)))
    model_files = model_files[model_order]
    
    p = ProgressMeter.Progress(
        num_models;
        dt=0.01,
        desc="Models Complete: ")

    mutex = Threads.Mutex()

    results = Vector{Matrix{Float64}}(undef, num_models)

    for m in 1:skip:num_models

        dict = BSON.load(joinpath(snapshot_loc, model_files[m]))
        results[m] = get_umap(Flux.mapleaves(Flux.data, dict[:model]), data_range; kwargs...)
        
        lock(mutex)
        next!(p)
        unlock(mutex)
        
    end

    results
end


get_umap_over_time_default(snapshot_loc, data_range) = 
    get_umap_over_time(snapshot_loc, data_range; n_neighbors=10, min_dist=0.001, n_epochs=200)


function save_umap_over_time(save_loc, snapshot_loc, data_range, skip=1; kwargs...)

    if !isdir(save_loc)
        mkdir(save_loc)
    end

    model_files = readdir(snapshot_loc)
    filter!((f)->splitext(f)[2] == ".bson", model_files)
    num_models = length(model_files)

    model_order = sortperm(parse.(Int64, getindex.(split.(getindex.(splitext.(model_files), 1), "_"), 2)))
    model_files = model_files[model_order]
    
    p = ProgressMeter.Progress(
        num_models;
        dt=0.01,
        desc="Models Complete: ")

    for m in 1:skip:num_models
        step = parse.(Int64, split(splitext(model_files[m])[1], "_")[2])
        if isfile(joinpath(save_loc, "umap_$(step).bson"))
            next!(p)
            continue
        end
        dict = BSON.load(joinpath(snapshot_loc, model_files[m]))
        res = get_umap(Flux.mapleaves(Flux.data, dict[:model]), data_range; kwargs...)

        bson(joinpath(save_loc, "umap_$(step).bson"), umap=res)

        next!(p)
    end

end
