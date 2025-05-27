import { Matrix, QrDecomposition } from "ml-matrix";

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

async function lyapunovSpectrum(
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

function calculateDimension(exponents: number[]): number {
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