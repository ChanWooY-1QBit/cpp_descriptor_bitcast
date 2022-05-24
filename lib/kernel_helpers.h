#include<type_traits>
#include <iostream>
#include <pybind11/pybind11.h>

// #include "absl/base/casts.h"

namespace einsum {
    // https://en.cppreference.com/w/cpp/numeric/bit_cast
    template <class To, class From>
    typename std::enable_if<sizeof(To) == sizeof(From) && 
        std::is_trivially_copyable<From>::value && 
        std::is_trivially_copyable<To>::value, To>::type bit_cast(const From& src) 
        noexcept {
            static_assert(
                std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to be trivially constructible"
            );
            To dst;
            memcpy(&dst, &src, sizeof(To));
            return dst;
    }

    template <typename T>
    std::string PackDescriptor(const T& descriptor) {
        std::cout << "pybind_helper desc: " << descriptor.subscripts << "\n";
        return PackDescriptorAsString(descriptor);
    }

    template <typename T>
    std::string PackDescriptorAsString(const T& descriptor) {
        std::cout << "kernel helpers desc: " << descriptor.subscripts << "\n";
        return std::string(bit_cast<const char*>(&descriptor), sizeof(T));
    }
    
    template <typename T>
    const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
        if (opaque_len != sizeof(T)) {
            throw std::runtime_error("Invalid opaque object size");
        }
        std::cout << "kernel helpers opaque: " << opaque << "\n";
        std::cout << "kernel helpers opaque_len: " << opaque_len << "\n";
        return bit_cast<const T*>(opaque);
    }
}
