precision highp float;
attribute vec3 position;
uniform mat4 worldViewProjection;

void main() {
  gl_Position = worldViewProjection * vec4(position, 1.0);
  gl_PointSize = 0.5;
}
