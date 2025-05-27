import { LyapunovParams } from "../../types";

export function loziJacobian(
  state: number[],
  params: LyapunovParams
): number[][] {
  const x = state[0];
  // sign(x): +1 if x>0, -1 if x<0; default to +1 at x=0
  const s = Math.sign(x) || 1;
  return [
    [ -params.a * s, params.b ],
    [  1,             0       ]
  ];
}
