#include<type_traits>
#include <iostream>
#include <pybind11.pybind11.h>

#include "absl/base/casts.h"

namespace einsum {
    template <typename T>
    pybind11::bytes PackDescriptor(const T& descriptor) {
        std::cout << "pybind_helper desc: " << descriptor.subscripts << "\n";
        return pybind11::bytes(PackDescriptorAsString(descriptor));
    }

    template <typename T>
    std::string PackDescriptorAsString(const T& descriptor) {
        std::cout << "kernel helpers desc: " << descriptor.subscripts << "\n";
        return std::string(absl::bit_cast<const char*>(&descriptor), sizeof(T));
    }
    
    template <typename T>
    const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
        if (opaque_len != sizeof(T)) {
            throw std::runtime_error("Invalid opaque object size");
        }
        std::cout << "kernel helpers opaque: " << opaque << "\n";
        std::cout << "kernel helpers opaque_len: " << opaque_len << "\n";
        return absl::bit_cast<const T*>(opaque);
    }
}