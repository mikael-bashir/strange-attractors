import { Matrix, QrDecomposition } from "ml-matrix";
import clifford from "../maps/clifford";

type LyapunovResult = {
  exponents: number[];
  dimension: number;
};

type LyapunovParams = {
  [key: string]: number;
};

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
  return { exponents, dimension };
}

export async function lyapunovSpectrum(
  f: (x: number[], params: LyapunovParams) => number[],
  J: (x: number[], params: LyapunovParams) => number[][],
  x0: number[],
  params: LyapunovParams,
  nIter: number,
  burnIn = 1_000
): Promise<number[]> {
  const dim = x0.length;
  let x = x0.slice();
  let Q = Matrix.eye(dim);
  const sums = new Array(dim).fill(0);

  // Burn-in to reach attractor
  for (let i = 0; i < burnIn; i++) x = f(x, params);

  // Main calculation loop
  for (let i = 0; i < nIter; i++) {
    const Jx = new Matrix(J(x, params));
    const Z = Jx.mmul(Q);
    const qr = new QrDecomposition(Z);
    
    // Corrected QR decomposition access
    Q = qr.orthogonalMatrix;
    const R = qr.upperTriangularMatrix;

    for (let k = 0; k < dim; k++) {
      const rkk = Math.max(Math.abs(R.get(k, k)), 1e-16);
      sums[k] += Math.log(rkk);
    }
    x = f(x, params);
  }

  // Normalize and sort descending
  return sums.map(s => s / nIter).sort((a, b) => b - a);
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

// 1. First create the Jacobian matrix function for the Clifford map
function cliffordJacobian(
  x: number[], 
  params: LyapunovParams
): number[][] {
  const [a, b, c, d] = [params.a, params.b, params.c, params.d];
  const [x0, y0] = x;

  // Partial derivatives
  const dxn_dx = -a * c * Math.sin(a * x0);
  const dxn_dy = a * Math.cos(a * y0);
  const dyn_dx = b * Math.cos(b * x0);
  const dyn_dy = -b * d * Math.sin(b * y0);

  return [
    [dxn_dx, dxn_dy],
    [dyn_dx, dyn_dy]
  ];
}

// 2. Wrap your clifford function in the expected format
function cliffordSystem(
  state: number[],
  params: LyapunovParams
): number[] {
  const [x, y] = state;
  return clifford(x, y, params.a, params.b, params.c, params.d);
}

// 3. Calculate the metrics
export async function getCliffordMetrics(a: number, b: number, c: number, d: number) {
  const initialConditions = [0.1, 0.1]; // Common starting point
  const params = { a, b, c, d };

  // Use 1M iterations for good convergence
  const results = await calculateLyapunov(
    cliffordSystem,
    cliffordJacobian,
    initialConditions,
    params,
    1_000_000,
    1_000
  );

  return {
    lyapunovExponents: results.exponents,
    fractalDimension: results.dimension,
    parameters: params
  };
}

// Example usage
// const metrics = await getCliffordMetrics(-1.4, 1.6, 1.0, 0.7);
// console.log('Lyapunov Exponents:', metrics.lyapunovExponents);
// console.log('Fractal Dimension:', metrics.fractalDimension);
