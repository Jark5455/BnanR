#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec2 uv;

layout(location = 0) out vec4 v_color;

layout(set = 0, binding = 0) uniform GlobalUbo {
    mat4 projection;
    mat4 view;
} ubo;

void main() {
    gl_Position = ubo.projection * ubo.view * vec4(position, 1.0);
    v_color = vec4(1.0, 1.0, 1.0, 1.0);
}