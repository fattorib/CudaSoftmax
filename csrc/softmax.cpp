#include <torch/extension.h>

torch::Tensor fusedSoftmax(torch::Tensor in);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused Softmax module",
    m.def("fusedSoftmax", torch::wrap_pybind_function(fusedSoftmax), "fusedSoftmax");
}