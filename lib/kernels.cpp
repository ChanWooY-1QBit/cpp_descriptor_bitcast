#include "kernels.h"
#include "kernel_helpers.h"

#include <iostream>

int main() {
    std::cout <<"before init\n";
    int shape[2] = {32, 30};
    char *subs = "ab,b->c";
    einsum::EinsumDescriptor descriptor = {
        subs, shape
    };
    std::cout << "before desc subs: " << descriptor.subscripts << "\n";
    std::cout << "before desc shape: " << descriptor.tensor_shape[0] << "\n";

    std::string opaque_str = einsum::PackDescriptor(descriptor);
    //const char *opaque = reinterpret_cast<const char*> (opaque_byte);
    const char* opaque = opaque_str.c_str();

    const einsum::EinsumDescriptor &convDescriptor = *einsum::UnpackDescriptor<einsum::EinsumDescriptor>(opaque, sizeof(einsum::EinsumDescriptor));
    std::cout << "conv desc subs: " << convDescriptor.subscripts << "\n";
    std::cout << "conv desc shape: " << convDescriptor.tensor_shape[0] << "\n";
    return 0;
}
