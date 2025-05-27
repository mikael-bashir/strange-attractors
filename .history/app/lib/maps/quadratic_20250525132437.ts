import { quadraticMapInterface } from "../types";

export default function quadratic(x: number, y: number, params: quadraticMapInterface ) {
    const 
    {
    a0, a1, a2, a3, a4, a5,
    b0, b1, b2, b3, b4, b5,
    } 
    = params;
    const nextX = a0 + a1 * x + a2 * y + a3 * x * x + a4 * x * y + a5 * y * y;
    const nextY = b0 + b1 * x + b2 * y + b3 * x * x + b4 * x * y + b5 * y * y;
    return [nextX, nextY];
}
