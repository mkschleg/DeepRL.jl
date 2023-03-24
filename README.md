# DeepRL.jl

An opinionated repository with some Deep Reinforcement Learning baselines written in julia using Flux. Currently, this is being used to test the refactored version of JuliaRL.

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


## Prior Repositories

The [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl) package is another really nice project which works to implement core reinforcement learning concepts for Julia. Currently, their package is much more feature complete than the DeepRL package. While this is the case, the core design principle of the two packages is quite different and I believe the overall goals of the projects are quite different.

### Goals and Principles

  My goal for this project is to provide tools for reinforcement learning researchers to do good research and science. To achieve this goal I've decided on a few core design principles:
  1. There should be limited obfuscation between what is written and what runs. A core reason why I decided to do my PhD work in Julia is because of the transparancy of the tools and the absence of object orientation. I believe OOP is a central cause for mistakes in RL and ML empirical studies. Because of this, all functions should be as transparent as possible with minimal layers of composition.
  2. I believe it is the researchers responsibility to make sure their code is consistent. Thus, I often design functions which can use a user managed random number generator (an RNG other than the GLOBAL). This is never a requirement, but I often use this design principle when there is any probabilistic component of my code.
  3. The researcher should know how to use their code and the libraries they use. This means I often provide very little in the way of default agents and do very little in the way of fixing the users mistakes. This often results in more work for the researcher, but I think of this as a positive.
  
  TL;DR
  1. Limited obfuscation and layer abstraction
  2. Runtime consitency
  3. Loud errors and no free lunch.


