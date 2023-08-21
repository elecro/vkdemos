add_program(triangle_vertex vktriangle_vertex.cpp)
copy_shaders(triangle_vertex
    passthrough.vert
    passthrough.vert.spv
    passthrough.frag
    passthrough.frag.spv
)
