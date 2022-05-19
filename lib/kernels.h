namespace einsum {
    struct EinsumDescriptor {
        char *subscripts;
        int *tensor_shape;
    };
}