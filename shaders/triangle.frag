#version 410

layout(location = 0) in vec3 vertColor;
layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = vec4(vertColor, 1.0);
}
