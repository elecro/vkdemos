add_program(triangle_subpass vktriangle_subpass.cpp)
copy_shaders(triangle_subpass
    passthrough.vert
    passthrough.vert.spv
    subpass_2_compose.vert
    subpass_2_compose.vert.spv
    subpass_0_colorizer.frag
    subpass_0_colorizer.frag.spv
    subpass_2_compose.frag
    subpass_2_compose.frag.spv
)
