'use client';
import { useEffect, useRef } from 'react';
import {
  Engine,
  Scene,
  ShaderMaterial,
  RenderTargetTexture,
  MeshBuilder,
  Vector2,
  Effect,
  Constants,
  Texture
} from '@babylonjs/core';
import { ComputeParams } from './lib/types';
import getOrthoCamera from './components/scene/cameras/orthographic-camera';
import orbitVertex from './lib/shaders/orbits/orbit.vert'
import orbitFragment from './lib/shaders/orbits/orbit.frag'
import visualiserVertex from './lib/shaders/visualiser/visualiser.vert'
import visualiserFragment from './lib/shaders/visualiser/visualiser.frag'

export default function AttractorCompute({ params }: { params: ComputeParams }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let engine: Engine, scene: Scene;

    (async () => {
      const canvas = canvasRef.current!;
      engine = new Engine(canvas, true);
      scene = new Scene(engine);

      // 1) Orthographic camera
      getOrthoCamera(canvas, scene);

      // 2) Create a 1×N RTT to hold the entire orbit
      const N = params.iterations;
      const orbitRTT = new RenderTargetTexture(
        'orbitRTT',
        { width: N, height: 1 },
        scene,
        false,
        true,
        Constants.TEXTURETYPE_FLOAT
      );

    orbitRTT.wrapU = orbitRTT.wrapV = Texture.CLAMP_ADDRESSMODE;
    orbitRTT.refreshRate = 1;
    orbitRTT.samples = 1;
    (orbitRTT as any)._texture?.updateSamplingMode(Texture.NEAREST_NEAREST);


      // 4) Material + quad to render into orbitRTT
      const orbitMat = new ShaderMaterial('orbitMat', scene, {
        vertexSource: orbitVertex,
        fragmentSource: orbitFragment
      }, {
        attributes: ['position'],
        uniforms:   ['a','b', 'c', 'd', 'initial','N']
      });
      orbitMat.setFloat('a', params.a);
      orbitMat.setFloat('b', params.b);
      orbitMat.setFloat('c', params.c);
      orbitMat.setFloat('d', params.d);
      orbitMat.setVector2('initial', new Vector2(params.x0, params.y0));
      orbitMat.setFloat('N', N);

      const quad = MeshBuilder.CreatePlane('orbitQuad', { size: 2 }, scene);
      quad.material = orbitMat;
      orbitRTT.renderList = [quad];

      // 5) Now visualize: build a 1×N grid, one vertex per iteration
      const grid = MeshBuilder.CreateGround('grid', {
        width: 2, height: 2,
        subdivisions: N - 1
      }, scene);

      const visMat = new ShaderMaterial('visMat', scene, {
        vertexSource: visualiserVertex,
        fragmentSource: visualiserFragment
      }, {
        attributes: ['position','uv'],
        uniforms:   ['orbitTex','worldViewProjection']
      });
      visMat.setTexture('orbitTex', orbitRTT);

      grid.material = visMat;

      // 6) Kick off render once (no need per-frame unless params change)
        engine.runRenderLoop(() => {
        // 1) compute the orbit into the RTT
        orbitRTT.render(true);

        visMat.setMatrix(
            'worldViewProjection',
            scene.getTransformMatrix()
        );

        // 2) draw your grid of points
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
