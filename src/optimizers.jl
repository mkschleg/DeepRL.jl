using Flux

"""
    RMSPropTF(η, ρ)
Implements the RMSProp algortihm as implemented in tensorflow. 
  - Learning Rate (η): Defaults to `0.001`.
  - Rho (ρ): Defaults to `0.9`.
  - Gamma (γ): Defaults to `0.0`.
  - Epsilon (ϵ): Defaults to `1e-6`
## Examples

## References
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
[Tensorflow RMSProp](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)
"""
mutable struct RMSPropTF
    eta::Float64
    rho::Float64
    gamma::Float64
    epsilon::Float64
    acc::IdDict
    mom::IdDict
end

RMSPropTF(η = 0.001, ρ = 0.9, γ = 0.0, ϵ = 1e-6) = RMSPropTF(η, ρ, γ, ϵ, IdDict(), IdDict())

function Flux.Optimise.apply!(o::RMSPropTF, x, Δ)
    η, ρ, γ, ϵ = o.eta, o.rho, o.gamma, o.epsilon
    acc = get!(o.acc, x, zero(x))::typeof(x)
    mom = get!(o.mom, x, zero(x))::typeof(x)
    @. acc = ρ * acc + (1 - ρ) * Δ^2
    @. mom = γ * mom + η * Δ/(sqrt(acc + ϵ))
    mom
end

"""
    RMSPropTFCentered(η, ρ)
Implements the Centered version of RMSProp algortihm as implemented in tensorflow. 
  - Learning Rate (η): Defaults to `0.001`.
  - Rho (ρ): Defaults to `0.9`.
  - Gamma (γ): Defaults to `0.0`.
  - Epsilon (ϵ): Defaults to `1e-6`
## Examples

## References
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
[Tensorflow RMSProp](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)
"""
mutable struct RMSPropTFCentered
    eta::Float64
    rho::Float64
    gamma::Float64
    epsilon::Float64
    grad::IdDict
    acc::IdDict
    mom::IdDict
end

RMSPropTFCentered(η = 0.001, ρ = 0.9, γ = 0.0, ϵ = 1e-6) =
    RMSPropTFCentered(η, ρ, γ, ϵ, IdDict(), IdDict(), IdDict())

function Flux.Optimise.apply!(o::RMSPropTFCentered, x, Δ)
    η, ρ, γ, ϵ = o.eta, o.rho, o.gamma, o.epsilon
    grad = get!(o.grad, x, zero(x))::typeof(x)
    acc = get!(o.acc, x, zero(x))::typeof(x)
    mom = get!(o.mom, x, zero(x))::typeof(x)
    @. grad = ρ * grad + (1 - ρ) * Δ
    @. acc = ρ * acc + (1 - ρ) * Δ^2
    @. mom = γ * mom + η * Δ/(sqrt(acc - grad^2 + ϵ))
    mom
end
