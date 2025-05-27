import { ArcRotateCamera, Vector3, Camera, Scene } from "@babylonjs/core";

declare global {
  var camera: ArcRotateCamera | undefined;
}

export default function getOrthoCamera(canvas: HTMLCanvasElement, scene: Scene): ArcRotateCamera {
  if (!canvas) throw new Error("canvas is null. Check the ref for camera.");

  if (!global.camera || global.camera.getScene() !== scene) {
    // Create an ArcRotateCamera looking straight down from Z=2
    global.camera = new ArcRotateCamera(
      "cam",
      Math.PI / 2,    // alpha
      Math.PI / 2,    // beta
      2,              // radius
      Vector3.Zero(), // target
      scene
    );

    // Switch to orthographic projection
    global.camera.mode = Camera.ORTHOGRAPHIC_CAMERA;

    // Use your exact frustum bounds
    global.camera.orthoTop    =  1;
    global.camera.orthoBottom = -1;
    global.camera.orthoLeft   = -1;
    global.camera.orthoRight  =  1;

    // Enable user controls (pan & zoom only)
    global.camera.attachControl(canvas, true);

    scene.activeCamera = global.camera;
    console.log("Orthographic camera created");
  } else {
    scene.activeCamera = global.camera;
    console.log("Orthographic camera re-used");
  }

  return global.camera;
}
