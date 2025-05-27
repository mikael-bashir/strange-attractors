import { LyapunovParams } from "../../types";

export function zaslavskiiJacobian(
  state: number[],
  params: LyapunovParams
): number[][] {
  const x = state[0];
  const c = Math.cos(x);
  return [
    [1 + params.a * c, params.b],
    [    params.a * c, params.b]
  ];
}
