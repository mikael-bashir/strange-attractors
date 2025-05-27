import { quadraticMapInterface } from "../../types";

export function quadraticJacobian(
  state: number[],
  params: quadraticMapInterface
): number[][] {
  const [x, y] = state;
  const {
    a1, a2, a3, a4, a5,
    b1, b2, b3, b4, b5,
  } = params;

  // ∂X'/∂x = a1 + 2·a3·x + a4·y
  // ∂X'/∂y = a2 + a4·x + 2·a5·y
  const dXdX = a1 + 2 * a3 * x + a4 * y;
  const dXdY = a2 + a4 * x + 2 * a5 * y;

  // ∂Y'/∂x = b1 + 2·b3·x + b4·y
  // ∂Y'/∂y = b2 + b4·x + 2·b5·y
  const dYdX = b1 + 2 * b3 * x + b4 * y;
  const dYdY = b2 + b4 * x + 2 * b5 * y;

  return [
    [dXdX, dXdY],
    [dYdX, dYdY]
  ];
}
