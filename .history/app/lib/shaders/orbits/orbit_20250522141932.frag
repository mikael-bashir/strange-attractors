#extension GL_EXT_frag_depth : enable
precision highp float;
varying vec2 vUV;
uniform float a, b, c, d;
uniform vec2 initial;
uniform float N;  // number of iterations

// map definition here
vec2 map(vec2 p) {
    float x = p.x;
    float y = p.y;
    return vec2(
    sin(a * y) + c * cos(a * x),
    sin(b * x) + d * cos(b * y)
    );
}

void main() {
    // which iteration index this pixel is
    float i = floor(vUV.x * N);
    vec2 p = initial;

    // iterate p with f, i+1 times
    for (int k = 0; k < 2000; k++) {
    if (k >= int(i)+1) break;
    p = map(p);
    }

    gl_FragColor = vec4(p, 0.0, 1.0);
}
