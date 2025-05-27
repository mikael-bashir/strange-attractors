// precision highp float;
// attribute vec3 position;
// attribute vec2 uv;
// uniform sampler2D orbitTex;
// uniform mat4 worldViewProjection;

// void main() {
//     vec4 pt = texture2D(orbitTex, uv);
//     gl_Position = worldViewProjection * vec4(pt.xy, 0.0, 1.0);
// }

precision highp float;
attribute vec3 position;
attribute vec2 uv;

uniform sampler2D orbitTex;
uniform mat4 worldViewProjection;

void main() {
    vec4 pt = texture2D(orbitTex, vec2(uv.x, 0.5));
    gl_Position = worldViewProjection * vec4(pt.xy, 0.0, 1.0);
}
