
import RLCore
import Flux

mutable struct GVFNetwork{M, H<:RLCore.AbstractHorde, N}
    gvf_model::M
    horde::H
    dqn_model::N
end


(m::GVFNetwork)(x) = m.dqn_model(Flux.data(m.gvf_model(x)))
