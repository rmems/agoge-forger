#include <torch/extension.h>

// CUDA forward declarations
void dummy_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c);

void dummy_add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(c.device().is_cuda(), "c must be a CUDA tensor");
    dummy_add_cuda(a, b, c);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dummy_add", &dummy_add, "Dummy Add (CUDA)");
}
