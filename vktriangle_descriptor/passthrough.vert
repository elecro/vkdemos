#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPosition;

layout(location = 0) out vec3 fragColor;

layout (set=0, binding=0) uniform myUniformBuffer {
    vec4 colors[3];
};

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);

    fragColor = colors[gl_VertexIndex].xyz;
}
