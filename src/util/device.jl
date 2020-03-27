
using Flux, CuArrays


to_host(x) = to_device(Val(:cpu), x)
to_device(::Val{:cpu}, x) = x




