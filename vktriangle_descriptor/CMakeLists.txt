add_program(triangle_descriptor vktriangle_descriptor.cpp)
copy_shaders(triangle_descriptor
    passthrough.vert
    passthrough.vert.spv
    passthrough.frag
    passthrough.frag.spv
)
