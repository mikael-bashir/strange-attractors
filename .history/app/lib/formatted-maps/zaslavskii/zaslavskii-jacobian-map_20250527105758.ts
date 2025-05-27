// src/formatted-maps/zaslavskii/zaslavskii-jacobian-map.ts
import { LyapunovParams } from "../../types";

export function zaslavskiiJacobian(
  state: number[],
  params: LyapunovParams
): number[][] {
  const x = state[0];
  const twoPiX = 2 * Math.PI * x;
  const sin2πx = Math.sin(twoPiX);

  // ∂yₙ₊₁/∂x = -2π·sin(2πx),  ∂yₙ₊₁/∂y = 0
  const dy_dx = -2 * Math.PI * sin2πx;
  const dy_dy = 0;

  // ∂xₙ₊₁/∂x = 1 + a·(∂yₙ₊₁/∂x)
  // ∂xₙ₊₁/∂y = a·(∂yₙ₊₁/∂y) = 0
  const dx_dx = 1 + params.a * dy_dx;
  const dx_dy = 0;

  return [
    [dx_dx, dx_dy],
    [dy_dx, dy_dy]
  ];
}
