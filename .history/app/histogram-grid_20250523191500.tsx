'use client';

import { useEffect, useRef } from 'react';
import {
  Engine,
  Scene,
  ArcRotateCamera,
  Vector3,
  HemisphericLight,
  DynamicTexture,
  MeshBuilder,
  StandardMaterial,
  Tools,
} from '@babylonjs/core';
import { ComputeParams } from './lib/types';
import getOrthoCamera from './components/scene/cameras/orthographic-camera';
import getZoomableOrthoCamera from './components/scene/cameras/zoomable-orthographic-camera';

export default function AttractorDensity({ params }: { params: ComputeParams }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const engine = new Engine(canvas, true);
    const scene = new Scene(engine);
    scene.clearColor.set(0, 0, 0, 1); // Black background

    // Camera
    getZoomableOrthoCamera(canvas, scene);

    // Light, makes graph look better
    new HemisphericLight('light', new Vector3(0, 1, 0), scene);

    // Histogram setup
    // const width = 1024;
    // const height = 1024;
    const width = 8192;
    const height = 8192;
    const grid = new Float32Array(width * height);

    // Generate attractor
    const { a, b, c, d, x0, y0 } = params;
    let x = x0, y = y0;
    const iter = params.iterations * 100;

    for (let i = 0; i < iter; i++) {
      // Map to grid
      const xi = Math.floor(((x + 2) / 4) * width);
      const yi = Math.floor(((y + 2) / 4) * height);
      if (xi >= 0 && xi < width && yi >= 0 && yi < height) {
        grid[yi * width + xi] += 1;
      }


    //   const xn = Math.sin(a * y) + c * Math.cos(a * x);
    //   const yn = Math.sin(b * x) + d * Math.cos(b * y);
    const xn = 1 - a * x**2 + y;
    const yn = b * x;
      x = xn;
      y = yn;
    }

    // Normalize and create image data
    let max = 0;
    for (let i = 0; i < grid.length; i++) {
    if (grid[i] > max) max = grid[i];
    }
    const imageData = new Uint8Array(width * height * 4);

    // for (let i = 0; i < width * height; i++) {
    //   const v = Math.floor((grid[i] / max) * 255);
    //   imageData[i * 4 + 0] = v;
    //   imageData[i * 4 + 1] = v;
    //   imageData[i * 4 + 2] = v;
    //   imageData[i * 4 + 3] = 255;
    // }

    for (let i = 0; i < width * height; i++) {
        const val = Math.log1p(grid[i]) / Math.log1p(max); // log1p(x) = log(1 + x)
        const v = Math.floor(val * 255);
        imageData[i * 4 + 0] = v*1/2;
        imageData[i * 4 + 1] = v;
        imageData[i * 4 + 2] = v;
        imageData[i * 4 + 3] = 255;
    }

    // Babylon dynamic texture
    const texture = new DynamicTexture('densityTex', { width, height }, scene, false);
    const ctx = texture.getContext();
    const imageDataObj = new ImageData(
      new Uint8ClampedArray(imageData.buffer),
      width,
      height
    );
    
    ctx.putImageData(imageDataObj, 0, 0);
    texture.update();

    // Create plane
    const plane = MeshBuilder.CreatePlane('plane', { width: 4, height: 4 }, scene);
    const mat = new StandardMaterial('mat', scene);
    mat.diffuseTexture = texture;
    mat.emissiveTexture = texture;
    mat.specularColor.set(0, 0, 0);
    mat.backFaceCulling = false;
    plane.material = mat;

    engine.runRenderLoop(() => scene.render());
    return () => engine.dispose();
  }, [params]);

  return (
    <div className="w-full h-full relative">
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
}