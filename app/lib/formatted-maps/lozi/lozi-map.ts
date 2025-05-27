import lozi from "../../maps/lozi";
import { LyapunovParams } from "../../types";

export function loziSystem(
    state: number[],
    params: LyapunovParams
): number[] {
    const [x, y] = state;
    return lozi(x, y, params.a, params.b);
}
