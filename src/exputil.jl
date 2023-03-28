module ExpUtils


import ChoosyDataLoggers

include("macros.jl")
include("build_ann.jl")

ChoosyDataLoggers.construct_logger(; steps=nothing, extra_groups_and_names=[]) =
    ChoosyDataLoggers.construct_logger([[:EXP]; extra_groups_and_names]; steps=steps)


function prep_save_results(data, save_extras)
    save_results = copy(data[:EXP])
    for ex in save_extras
        if ex isa AbstractArray
            save_results[Symbol(ex[1]*"_"*ex[2])] = data[Symbol(ex[1])][Symbol(ex[2])]
        else
            for k in keys(data[Symbol(ex)])
                save_results[Symbol(ex * "_" * string(k))] = data[Symbol(ex)][Symbol(k)]
            end
        end
    end
    save_results
end


end
