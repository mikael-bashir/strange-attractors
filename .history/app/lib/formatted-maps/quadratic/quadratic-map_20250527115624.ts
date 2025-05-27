import quadratic from "../../maps/quadratic";

import { LyapunovParams } from "../../types";

export function quadraticSystem(
    state: number[],
    params: LyapunovParams
): number[] {
    const [x, y] = state;
    return quadratic(x, y, params);
}
