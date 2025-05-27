

// Writes the point of the orbit at this iteration
precision highp float;
uniform float a, b, c, d, N;
uniform vec2 initial;
varying vec2 vUV;

vec2 iterate(vec2 p) {
    // return vec2(sin(a * p.y) + c * cos(a * p.x),
    //     sin(b * p.x) + d * cos(b * p.y));
    return vec2(1 - a*p.x*p.x + p.y, b*p.x)
}

void main() {
    vec2 pt = initial;
    for (float i = 0.0; i < N; i++) {
        if (i > vUV.x * N) break;
        pt = iterate(pt);
    }

    // Store orbit point as vec4 (to make it visible)
    gl_FragColor = vec4(pt.xy, 0.0, 1.0);
}
