// export default function zaslavskii(x: number, y: number, a: number, b: number) {
//     return [(((x + y + a * Math.sin(x)) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)  , b * y + a * Math.sin(x)];
// }

// src/maps/zaslavskii.ts
export default function zaslavskii(
  x: number,
  y: number,      // unused in this version, but kept for signature
  nu: number,     // ν
  a: number,      // coupling coefficient a
  r: number       // r
): [number, number] {
  // compute yₙ₊₁ first
  const y1 = Math.cos(2 * Math.PI * x) + Math.exp(-r);

  // then xₙ₊₁ = x + ν + a·yₙ₊₁  (mod 1)
  const x1 = ((x + nu + a * y1) % 1 + 1) % 1;

  return [x1, y1];
}
