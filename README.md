# DeepRL.jl

A repository with some Deep Reinforcement Learning baselines written in julia using Flux. Currently, this is being used to test the refactored version of JuliaRL.

## Installing

Currently, this library is built to be used as a project and not as a package. To use ArcadeLearningEnvironment you need to install a version of cmake before instantiating this environment.

MacOS
```zsh
brew install cmake
```

Linux (apt)
```zsh
sudo apt-get install cmake
```

Then clone this repository and instantiate the environment
```zsh
git clone https://github.com/mkschleg/DeepRL.jl.git
cd DeepRL.jl
julia --project -e "using Pkg; Pkg.instantiate()"
```


## To Run

```Julia
include("experiment/mountain_car.jl")
#get single run of experiment with seed = 10.
ret = MountainCarExperiment.main_experiment(10, 500)
using Plots
gr()
plot(ret[1])
```


## Atari Experiments

In the process of developing this repository, I've implemented some very basic versions of a DQN and am benchmarking this implementation against the results and hyperparameters reported in [Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents by Machado et. al.](https://research.google/pubs/pub48769/). While we are only starting, we want to ensure our results follow closely to these and our API allows for more transparent experiments than previous implementations (i.e. in python). 



## Next Steps

- [x] GPU support
- [ ] MonteCarlo Rollouts w/ Mountain Car
- [ ] Policy Gradients w/ continuous mountain car
