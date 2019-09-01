


function exp_settings!(as::ArgParseSettings)
    @add_arg_table as begin
        "--exp_loc"
        help="Location of experiment"
        arg_type=String
        default="tmp"
        "--seed"
        help="Seed of rng"
        arg_type=Int64
        default=0
        "--steps"
        help="number of steps"
        arg_type=Int64
        default=100
        "--prev_action_or_not"
        action=:store_true
        "--verbose"
        action=:store_true
        "--working"
        action=:store_true
        "--progress"
        action=:store_true
    end
end


