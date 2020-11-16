#version 450

layout (input_attachment_index = 0, binding = 1) uniform subpassInput samplerRed;
layout (input_attachment_index = 1, binding = 2) uniform subpassInput samplerGreen;
layout (input_attachment_index = 2, binding = 3) uniform subpassInput samplerBlue;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

void main() {
    const vec3 target = vec3(0.01, 0.01, 0.01);

    float colorRed = subpassLoad(samplerRed).r;
    float colorGreen = subpassLoad(samplerGreen).g;
    float colorBlue = subpassLoad(samplerBlue).b;

    vec3 color = vec3(colorRed, colorGreen, colorBlue);

    if (any(lessThan(color, target))) {
        // outside of any colored parts
        color = vec3(0.1, 0.1, 0.1);

        color.rgb *= vec3(int(mod(gl_FragCoord.x, 40) < 20));
        color.rgb *= vec3(int(mod(gl_FragCoord.y, 40) < 20));
    }

    outColor = vec4(color, 1.0);
}

