add_program(triangle_glfw vktriangle_glfw.cpp)
copy_shaders(triangle_glfw
    passthrough.vert
    passthrough.vert.spv
    passthrough.frag
    passthrough.frag.spv
)
