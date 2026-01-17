#version 450

layout(location = 0) out vec4 FragColor;
layout(location = 0) in vec4 v_color;

void main()
{
    FragColor = v_color;
}