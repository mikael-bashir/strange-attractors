"use client";

import { useEffect, useRef } from "react";
import {
  Engine,
  Scene,
  Vector3,
  HemisphericLight,
  RawTexture,
  Texture,
  MeshBuilder,
  StandardMaterial
} from "@babylonjs/core";
import getZoomableOrthoCamera from "../scene/cameras/zoomable-orthographic-camera";
import { histogramGrid } from "@/app/lib/functions/histogram";
import clifford from "@/app/lib/maps/clifford";
import henon from "@/app/lib/maps/henon";

type GradientStop = { offset: number; color: number[] };

type Props = {
  a: number;
  b: number;
  c: number;
  d: number;
  width?: number;
  height?: number;
  iterations?: number;
};

export default function AttractorHistogram({
  a,
  b,
  c,
  d,
  width = 1920,
  height = 1080,
  iterations = 10_000_000,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const histogram = histogramGrid(
      "high",
      (x, y) => clifford(x, y, a, b, c, d),
    //   (x, y) => henon(x, y, a, b),
      iterations
    );

    // Black to white gradient
    const gradientStops: GradientStop[] = [
      { offset: 0.0, color: [0, 0, 0] },   // Black
      { offset: 1.0, color: [1, 1, 1] },   // White
    ];

    const renderHistogram = async () => {
      const histWidth = histogram.length;
      const histHeight = histogram[0].length;

      // Calculate log-scaled min/max
      let logMin = Infinity, logMax = -Infinity;
      for (let x = 0; x < histWidth; x++) {
        for (let y = 0; y < histHeight; y++) {
          const logValue = Math.log(histogram[x][y] + 1); // Add 1 to avoid log(0)
          if (logValue < logMin) logMin = logValue;
          if (logValue > logMax) logMax = logValue;
        }
      }
      const logRange = logMax - logMin || 1; // Prevent division by zero

      function getColor(t: number): [number, number, number] {
        t = Math.max(0, Math.min(1, t));
        let i = 0;
        while (i + 1 < gradientStops.length && gradientStops[i + 1].offset < t) i++;
        const a = gradientStops[i];
        const b = gradientStops[Math.min(i + 1, gradientStops.length - 1)];
        const span = b.offset - a.offset || 1;
        const ft = (t - a.offset) / span;
        return [
          a.color[0] + (b.color[0] - a.color[0]) * ft,
          a.color[1] + (b.color[1] - a.color[1]) * ft,
          a.color[2] + (b.color[2] - a.color[2]) * ft,
        ];
      }

      const data = new Uint8Array(histWidth * histHeight * 4);
      let ptr = 0;
      for (let y = 0; y < histHeight; y++) {
        for (let x = 0; x < histWidth; x++) {
          // Apply log scaling and normalize
          const logValue = Math.log(histogram[x][y] + 1);
          const t = (logValue - logMin) / logRange;
          const [r, g, b] = getColor(t);
          data[ptr++] = Math.floor(r * 255);
          data[ptr++] = Math.floor(g * 255);
          data[ptr++] = Math.floor(b * 255);
          data[ptr++] = 255;
        }
      }

      const engine = new Engine(canvas, true);
      const scene = new Scene(engine);
      scene.clearColor.set(0, 0, 0, 1); // Black background
      getZoomableOrthoCamera(canvas, scene);
      new HemisphericLight("light", new Vector3(0, 1, 0), scene);

      const rawTex = RawTexture.CreateRGBATexture(
        data,
        histWidth,
        histHeight,
        scene,
        false,
        false,
        Texture.NEAREST_SAMPLINGMODE
      );

      const mat = new StandardMaterial("mat", scene);
      mat.disableLighting = true;
      mat.emissiveTexture = rawTex;

      const plane = MeshBuilder.CreatePlane("p", { size: 2 }, scene);
      plane.material = mat;

      engine.runRenderLoop(() => scene.render());
    };

    renderHistogram();
  }, [a, b, c, d, width, height, iterations]);

  return (
    <canvas
      ref={canvasRef}
      id="renderCanvas"
      width={width}
      height={height}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}
