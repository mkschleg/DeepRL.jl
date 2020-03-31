using Flux, CuArrays



to_device(use_cuda::Val{:cpu}, x) = to_host(x)
to_device(use_cuda::Val{:gpu}, x) = to_gpu(x)

to_host(x) = fmap(x -> adapt(Array, x), m)
to_host(x::Array) = x

to_gpu(x) = fmap(CuArrays.cu, x)
# gpu_if_avail(x, use_cuda::Val{true}) = fmap(CuArrays.cu, x)
