import zaslavskii from "../../maps/zaslavskii";
import { LyapunovParams } from "../../types";

export function zaslavskiiSystem(
    state: number[],
    params: LyapunovParams
): number[] {
    const [x, y] = state;
    return zaslavskii(x, y, params.a, params.b);
}
