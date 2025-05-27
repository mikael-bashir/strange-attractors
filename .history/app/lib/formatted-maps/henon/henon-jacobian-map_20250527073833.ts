import { LyapunovParams } from "../../types";

export function henonJacobian(
  x: number,
  y: number,
  params: LyapunovParams
): number[][] {
  return [
    [-2 * params.a * x, 1],
    [params.b, 0]
  ];
}
