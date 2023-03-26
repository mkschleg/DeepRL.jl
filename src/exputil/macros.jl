
module Macros

using MacroTools: prewalk, postwalk, @capture
import Markdown: Markdown, MD, @md_str
import TOML


struct InfoStr
    str::String
end

# export @help_str
macro help_str(str)
end

# export @info_str
macro info_str(str)
end

function get_args_and_order(default_config)

    arg_order = String[]
    args = Expr[]
    postwalk(default_config) do ex
        chk = @capture(ex, k_ => v_)
        # @show k
        if chk
            k_str = string(k)
            push!(arg_order, k_str)
            push!(args, :($k_str=>$v))
        end
        ex
    end
    args, arg_order
    
end

function get_help_str(default_config, __module__)
    start_str = "# Automatically generated docs for $(__module__) config."
    help_str_strg = InfoStr[
        InfoStr(start_str)
    ]
    postwalk(default_config) do expr
        expr_str = string(expr)
        if length(expr_str) > 5 && (expr_str[1:5] == "help\"" || expr_str[1:5] == "info\"")
            push!(help_str_strg, InfoStr(string(expr)[6:end-1]))
        end
        expr
    end
    md_strs = [Markdown.parse(hs.str) for hs in help_str_strg]
    join(md_strs, "\n")
end


macro generate_config_funcs(default_config)
    # println(default_config)

    func_name = :default_config
    help_func_name = :help
    create_toml_func_name = :create_toml_template
    mdstrings = String[]
    src_file = relpath(String(__source__.file))

    
    docs = get_help_str(default_config, __module__)
    args, arg_order = get_args_and_order(default_config)
    # @show args

    create_toml_docs = """
        create_toml_template(save_file=nothing; database=false)

    Used to create toml template. If save_file is nothing just return toml string. 
    If database is true, then generate using mysql backend otherwise generate using file backend.
    """
    quote
        @doc $(docs)
        function $(esc(func_name))()
            Dict{String, Any}(
                $(args...)
            )

        end

        function $(esc(help_func_name))()
            local docs = Markdown.parse($(docs))
            # InteractiveUtils.less(docs)
            display(docs)
        end

        function $(esc(create_toml_func_name))(save_file=nothing; database=false)
            local ao = filter((str)->str!="save_dir", $arg_order)
            cnfg = $(esc(func_name))()
            cnfg_filt = filter((p)->p.first != "save_dir", cnfg)
            sv_path = get(cnfg, "save_dir", "<<ADD_SAVE_DIR>>")

            mod = $__module__

            save_info = if database
                """
                save_backend="mysql" # mysql only database backend supported
                database="<<SET DATABASE NAME>>" # Database name
                save_dir="$(sv_path)" # Directory name for exceptions, settings, and more!"""
            else
                """
                save_backend="file" # file saving mode
                file_type = "jld2" # using JLD2 as save type
                save_dir="$(sv_path)" # save location"""
            end
            
            toml_str = """
            Config generated automatically from default_config. When you have finished 
            making changes to this config for your experiment comment out this line.

            info \"\"\"

            \"\"\"

            [config]
            $(save_info)
            exp_file = "$($src_file)"
            exp_module_name = "$(mod)"
            exp_func_name = "main_experiment"
            arg_iter_type = "iter"

            [static_args]
            """
            buf = IOBuffer()

            TOML.print(buf,
                cnfg_filt, sorted=true, by=(str)->findfirst((strinner)->str==strinner, ao)
                       )
            toml_str *= String(take!(buf))

            toml_str *= """\n[sweep_args]
            # Put args to sweep over here.
            """

            if save_file === nothing
                toml_str
            else
                open(save_file, "w") do io
                    write(io, toml_str)
                end
            end
            
        end
    end    
end

macro generate_working_function()
    quote
        """
            working_experiment

        Creates a wrapper experiment where the main experiment is called with progress=true, testing=true 
        and the config is the default_config with the addition of the keyword arguments.
        """
        function $(esc(:working_experiment))(progress=true;kwargs...)
            config = $__module__.default_config()
            for (n, v) in kwargs
                config[string(n)] = v
            end
            $__module__.main_experiment(config; progress=progress, testing=true)
        end
    end
end

macro generate_ann_size_helper(construct_env=:construct_env, construct_agent=:construct_agent)
    const_env_sym = construct_env
    quote
        """
            get_ann_size

        Helper function which constructs the environment and agent using default config and kwargs then returns
        the number of parameters in the model.
        """
        function $(esc(:get_ann_size))(;kwargs...)
            config = $__module__.default_config()
            for (k, v) in kwargs
                config[string(k)] = v
            end
            env = $(esc(const_env_sym))(config, $__module__.Random.GLOBAL_RNG)
            agent = $(esc(construct_agent))(env, config, $__module__.Random.GLOBAL_RNG)
            sum(length, $__module__.Flux.params($__module__.ActionRNNs.get_model(agent)))
        end
    end
end

end # module macros

