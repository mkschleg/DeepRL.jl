# DeepRL.jl
A repository with some Deep Reinforcement Learning baselines written in julia using Flux. Currently, this is being used to test the refactored version of JuliaRL.


## To Run

'''julia
include("experiment/mountain_car.jl")
# Get a single run with seed = 10
ret = MountainCarExperiment.main_experiment(10, 500)
using Plots
gr()
plot(ret[1])
'''



## Next Steps

- [ ] MonteCarlo Rollouts w/ Mountain Car
- [ ] Policy Gradients w/ continuous mountain car
