'use client';
import { useEffect, useRef } from 'react';
import {
  Engine,
  Scene,
  ShaderMaterial,
  Mesh,
  VertexData,
  SceneLoader,
  Vector3,
} from '@babylonjs/core';
import { ComputeParams } from '../../lib/types';
import getOrthoCamera from '../scene/cameras/orthographic-camera';
import visualiserVertex from './lib/shaders/visualiser/visualiser.vert';
import visualiserFragment from './lib/shaders/visualiser/visualiser.frag';

export default function AttractorCompute({ params }: { params: ComputeParams }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let engine: Engine, scene: Scene;

    const generateOrbit = (
      { a, b, c, d, x0, y0, iterations }: ComputeParams,
      noiseLevel = 0
    ): Float32Array => {
      const orbit = new Float32Array(iterations * 3); // x, y, z
      let x = x0;
      let y = y0;

      for (let i = 0; i < iterations; i++) {
        orbit[3 * i] = x;
        orbit[3 * i + 1] = y;
        orbit[3 * i + 2] = 0; // z = 0 for 2D

        const xn = Math.sin(a * y) + c * Math.cos(a * x) + (Math.random() - 0.5) * noiseLevel;
        const yn = Math.sin(b * x) + d * Math.cos(b * y) + (Math.random() - 0.5) * noiseLevel;

        x = xn;
        y = yn;
      }

      return orbit;
    };

    (async () => {
      const canvas = canvasRef.current!;
      engine = new Engine(canvas, true);
      scene = new Scene(engine);
    //   scene.clearColor.set(0, 0, 0, 1); // Black background

      getOrthoCamera(canvas, scene);

      const N = params.iterations;
      const orbitData = generateOrbit(params);

      // Mesh with orbit positions
      const orbitMesh = new Mesh('orbitMesh', scene);
      const vertexData = new VertexData();
      vertexData.positions = Array.from(orbitData);
      vertexData.indices = [...Array(N).keys()]; // [0, 1, ..., N-1]
      vertexData.applyToMesh(orbitMesh, true);

      // Shader material (draws GL_POINTS)
      const visMat = new ShaderMaterial('visMat', scene, {
        vertexSource: visualiserVertex,
        fragmentSource: visualiserFragment
      }, {
        attributes: ['position'],
        uniforms: ['worldViewProjection']
      });

      visMat.pointsCloud = true; // ← THIS enables GL_POINTS
    orbitMesh.material = visMat;


      engine.runRenderLoop(() => {
        visMat.setMatrix('worldViewProjection', scene.getTransformMatrix());
        scene.render();
      });
    })();

    return () => engine?.dispose();
  }, [params]);

  return (
    <div className="w-full h-full relative">
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
}
