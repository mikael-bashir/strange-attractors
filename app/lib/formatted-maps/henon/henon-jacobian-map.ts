import { LyapunovParams } from "../../types";

export function henonJacobian(
    x: number[],
    params: LyapunovParams
): number[][] {
    return [
        [-2 * params.a * x[0], 1],
        [params.b, 0]
    ];
}
