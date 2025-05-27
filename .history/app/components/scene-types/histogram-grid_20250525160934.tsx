import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight,
         RawTexture, Texture, MeshBuilder, StandardMaterial, Tools } from "@babylonjs/core";
import getZoomableOrthoCamera from "../scene/cameras/zoomable-orthographic-camera";
import { histogramGrid } from "@/app/lib/utils";
import clifford from "@/app/lib/maps/clifford";

async function renderHistogram(
  canvas: HTMLCanvasElement,
  histogram: number[][],
  gradientStops: { offset: number; color: number[] }[]
) {
  const width = histogram.length;
  const height = histogram[0].length;

  // 1) Compute min/max to normalize
  let min = Infinity, max = -Infinity;
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      const v = histogram[x][y];
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const range = max - min || 1;

  // 2) Gradient lookup helper
  function getColor(t: number): [number, number, number] {
    // clamp
    t = Math.max(0, Math.min(1, t));
    // find two stops
    let i = 0;
    while (i + 1 < gradientStops.length && gradientStops[i+1].offset < t) i++;
    const a = gradientStops[i];
    const b = gradientStops[Math.min(i+1, gradientStops.length-1)];
    const span = b.offset - a.offset || 1;
    const ft = (t - a.offset) / span;
    return [
      a.color[0] + (b.color[0] - a.color[0]) * ft,
      a.color[1] + (b.color[1] - a.color[1]) * ft,
      a.color[2] + (b.color[2] - a.color[2]) * ft,
    ];
  }

  // 3) Build RGBA buffer
  const data = new Uint8Array(width * height * 4);
  let ptr = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = (histogram[x][y] - min) / range;
      const [r, g, b] = getColor(v);
      data[ptr++] = Math.floor(r * 255);
      data[ptr++] = Math.floor(g * 255);
      data[ptr++] = Math.floor(b * 255);
      data[ptr++] = 255;               // opaque
    }
  }

  // 4) Babylon setup
  const engine = new Engine(canvas, true);
  const scene = new Scene(engine);
//   const camera = new ArcRotateCamera("cam", Math.PI / 2, Math.PI / 2, 1, Vector3.Zero(), scene);
//   camera.mode = camera.ORTHOGRAPHIC_CAMERA;
//   // match ortho to canvas aspect
//   const aspect = canvas.width / canvas.height;
  const half = 1;
//   camera.orthoLeft   = -half * aspect;
//   camera.orthoRight  =  half * aspect;
//   camera.orthoTop    =  half;
//   camera.orthoBottom = -half;
//   camera.attachControl(canvas, true);
  getZoomableOrthoCamera(canvas, scene);
  new HemisphericLight("light", new Vector3(0,1,0), scene);

  // 5) Create RawTexture from our buffer
  const rawTex = RawTexture.CreateRGBATexture(
    data, width, height, scene, false, false, Texture.NEAREST_SAMPLINGMODE
  );

  // 6) Fullscreen plane to show it
  const mat = new StandardMaterial("mat", scene);
  mat.disableLighting = true;
  mat.emissiveTexture = rawTex;
  const plane = MeshBuilder.CreatePlane("p", { size: 2 * half }, scene);
  plane.material = mat;

  // 7) Render once (or loop if you like)
  engine.runRenderLoop(() => scene.render());
}

// ----- Usage example -----
const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
const width = 1920, height = 1080;

// Suppose you’ve already filled this with your 10 million‐point histogram:
const histogram: number[][] = Array.from({length: width},
  () => new Array<number>(height).fill(0));

// Define a gradient (e.g., blue→cyan→yellow→red)
const gradientStops = [
  { offset: 0.0, color: [0, 0, 0.5] },   // dark blue
  { offset: 0.3, color: [0, 1, 1]   },   // cyan
  { offset: 0.6, color: [1, 1, 0]   },   // yellow
  { offset: 1.0, color: [1, 0, 0]   },   // red
];

renderHistogram(canvas, histogramGrid("high", clifford, 10_000_000), gradientStops);
