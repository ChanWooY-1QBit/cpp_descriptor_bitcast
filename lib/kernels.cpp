#include "kernels.h"
#include "kernel_helpers.h"

#include <iostream>

int main() {
    einsum::EinsumDescriptor descriptor = {
        "ab,b->c", (int[]){32,30}
    };
    std::cout << "before desc subs: " << descriptor.subscripts << "\n";
    std::cout << "before desc shape: " << descriptor.tensor_shape[0] << "\n";

    pybind11::bytes opaque = einsum::PackDescriptor(descriptor);

    const einsum::EinsumDescriptor &convDescriptor = *einsum::UnpackDescriptor(opaque, sizeof(opaque));
    std::cout << "conv desc subs: " << convDescriptor.subscripts << "\n";
    std::cout << "conv desc shape: " << convDescriptor.tensor_shape[0] << "\n";
}