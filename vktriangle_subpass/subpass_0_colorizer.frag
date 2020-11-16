#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in flat int instanceId;

//layout(location = 0) out vec4 outColor;

layout(location = 1) out vec4 colorRed;
layout(location = 2) out vec4 colorGreen;
layout(location = 3) out vec4 colorBlue;


void main() {
    vec3 color = vec3(0.0);
    color[instanceId] = fragColor[instanceId];

/*    switch (instanceId-1) {
        case 0: { colorRed = vec4(color, 1.0); break; }
        case 1: { colorGreen = vec4(color, 1.0); break; }
        case 2: { colorBlue = vec4(color, 1.0); break; }
    }*/

    switch(instanceId) {
        case 0: colorRed = vec4(fragColor.r, 0.0, 0.0, 1.0); break;
        case 1: colorGreen = vec4(0.0, fragColor.g, 0.0, 1.0); break;
        case 2: colorBlue = vec4(0.0, 0.0, fragColor.b, 1.0); break;
    }
}
