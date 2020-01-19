# DeepRL.jl

A repository with some Deep Reinforcement Learning baselines written in julia using Flux. Currently, this is being used to test the refactored version of JuliaRL.


## To Run

```Julia
include("experiment/mountain_car.jl")
#get single run of experiment with seed = 10.
ret = MountainCarExperiment.main_experiment(10, 500)
using Plots
gr()
plot(ret[1])
```


## Next Steps

- [ ] GPU support
- [ ] MonteCarlo Rollouts w/ Mountain Car
- [ ] Policy Gradients w/ continuous mountain car
