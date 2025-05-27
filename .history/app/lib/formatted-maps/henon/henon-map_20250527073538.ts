import henon from "../../maps/henon";
import { LyapunovParams } from "../../types";

export function cliffordSystem(
    state: number[],
    params: LyapunovParams
): number[] {
    const [x, y] = state;
    return henon(x, y, params.a, params.b);
}
