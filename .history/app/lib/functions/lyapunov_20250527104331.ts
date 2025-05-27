import { Matrix, QrDecomposition } from "ml-matrix";
import clifford from "../maps/clifford";

import { LyapunovParams, LyapunovResult } from "../types";
import { cliffordSystem } from "../formatted-maps/clifford/clifford-map";
import { cliffordJacobian } from "../formatted-maps/clifford/clifford-jacobian-map";
import { henonSystem } from "../formatted-maps/henon/henon-map";
import { henonJacobian } from "../formatted-maps/henon/henon-jacobian-map";
import { loziSystem } from "../formatted-maps/lozi/lozi-map";
import { loziJacobian } from "../formatted-maps/lozi/lozi-jacobian-map";

export async function calculateLyapunov(
  f: (x: number[], params: LyapunovParams) => number[],
  J: (x: number[], params: LyapunovParams) => number[][],
  x0: number[],
  params: LyapunovParams,
  nIter: number,
  burnIn = 1_000
): Promise<LyapunovResult> {
  const exponents = await lyapunovSpectrum(f, J, x0, params, nIter, burnIn);
  const dimension = calculateDimension(exponents);
  return { exponents, dimension };2
}



export async function lyapunovSpectrum(
  f: (x: number[], params: LyapunovParams) => number[],
  J: (x: number[], params: LyapunovParams) => number[][],
  x0: number[],
  params: LyapunovParams,
  nIter: number,
  burnIn = 1_000,
  debug = true
): Promise<number[]> {
  const dim = x0.length;
  let x = x0.slice();
  let Q = Matrix.eye(dim);
  const sums = new Array(dim).fill(0);

  // Burn-in
  for (let i = 0; i < burnIn; i++) {
    x = f(x, params);
  }

  // How often to print status
  const checkpoints = 10;
  const interval = Math.floor(nIter / checkpoints) || 1;

  // Main loop
  for (let i = 1; i <= nIter; i++) {
    // 1) Apply Jacobian to each basis vector
    const Jx = new Matrix(J(x, params));
    const Z  = Jx.mmul(Q);
    const qr = new QrDecomposition(Z);

    // 2) Re-orthonormalize
    Q = qr.orthogonalMatrix;
    const R = qr.upperTriangularMatrix;

    // 3) Accumulate log of diagonal
    for (let k = 0; k < dim; k++) {
      const rkk = Math.max(Math.abs(R.get(k, k)), 1e-16);
      sums[k] += Math.log(rkk);
    }

    // 4) Step the trajectory
    x = f(x, params);

    // 5) Debug checkpoints
    if (debug && (i % interval === 0 || i === nIter)) {
      const N = i;
      const exps = sums.map(s => s / N);
      const sumExps = exps.reduce((a, b) => a + b, 0);

      // Partial convergence
      console.log(
        `Iter ${N}/${nIter} → λ ≈ [${exps.map(e => e.toFixed(4)).join(", ")}]`
      );

      // Orthogonality check: Qᵀ·Q should ≈ I
      const QtQ = Q.transpose().mmul(Q);
      let maxOff = 0;
      for (let r = 0; r < dim; r++) {
        for (let c = 0; c < dim; c++) {
          if (r !== c) {
            maxOff = Math.max(
              maxOff,
              Math.abs(QtQ.get(r, c))
            );
          }
        }
      }
      if (maxOff > 1e-6) {
        console.warn(`  ▶ QᵀQ off‐diagonal max = ${maxOff.toExponential()}`);
      }

      // Volume‐preservation check (sum of exponents)
      //   e.g. for an area‐preserving map expect sum ≈ 0
      if (Math.abs(sumExps) > 1e-2) {
        console.warn(`  ▶ ∑λ = ${sumExps.toFixed(4)} (should be ≈0 if volume‐preserving)`);
      }
    }
  }

  // Final normalized and sorted
  return sums
    .map(s => s / nIter)
    .sort((a, b) => b - a);
}


export function calculateDimension(exponents: number[]): number {
  let cumulative = 0;
  let dimension = 0;
  let k = 0;

  // Create a copy to avoid mutating original array
  const sortedExponents = [...exponents].sort((a, b) => b - a);

  for (; k < sortedExponents.length; k++) {
    cumulative += sortedExponents[k];
    if (cumulative < 0) break;
    
    if (k === sortedExponents.length - 1) {
      dimension = k + 1;
      break;
    }

    const lambdaNext = sortedExponents[k + 1];
    dimension = k + 1 + cumulative / Math.abs(lambdaNext);
  }

  return Math.max(0, dimension);
}

// 3. Calculate the metrics
export async function getCliffordMetrics(a: number, b: number, c: number, d: number) {
  const initialConditions = [0.0, 0.0]; // Common starting point
  const params = { a, b, c, d };

  // Use 1M iterations for good convergence
  const results = await calculateLyapunov(
    loziSystem,
    loziJacobian,
    // [0.631, 0.189],
    initialConditions,
    params,
    10_000_000,
    10_000
  );

  return {
    lyapunovExponents: results.exponents,
    fractalDimension: results.dimension,
    parameters: params
  };
}
