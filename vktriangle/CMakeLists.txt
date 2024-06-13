add_program(triangle vktriangle.cpp)
copy_shaders(triangle 
    triangle.vert
    triangle.vert.spv
    passthrough.frag
    passthrough.frag.spv
)
