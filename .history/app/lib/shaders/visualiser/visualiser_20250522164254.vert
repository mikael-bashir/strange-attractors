precision highp float;
attribute vec3 position;

uniform mat4 worldViewProjection;

void main() {
  gl_Position = worldViewProjection * vec4(position.xy, 0.0, 1.0);
  gl_PointSize = 2.0;
}