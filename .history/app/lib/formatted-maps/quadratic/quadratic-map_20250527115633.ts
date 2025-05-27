import quadratic from "../../maps/quadratic";

import { LyapunovParams } from "../../types";
import { quadraticMapInterface } from "../../types";

export function quadraticSystem(
    state: number[],
    params: quadraticMapInterface
): number[] {
    const [x, y] = state;
    return quadratic(x, y, params);
}
