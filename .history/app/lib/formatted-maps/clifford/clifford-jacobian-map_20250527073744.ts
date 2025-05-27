import { LyapunovParams } from "../../types";

export function cliffordJacobian(
    x: number[],
    params: LyapunovParams
): number[][] {
    const [a, b, c, d] = [params.a, params.b, params.c, params.d];
    const [x0, y0] = x;

    // Partial derivatives
    const dxn_dx = -a * c * Math.sin(a * x0);
    const dxn_dy = a * Math.cos(a * y0);
    const dyn_dx = b * Math.cos(b * x0);
    const dyn_dy = -b * d * Math.sin(b * y0);

    return [
        [dxn_dx, dxn_dy],
        [dyn_dx, dyn_dy]
    ];
}
