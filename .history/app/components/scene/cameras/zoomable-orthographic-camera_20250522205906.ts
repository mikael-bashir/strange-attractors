import { ArcRotateCamera, Vector3, Camera, Scene } from "@babylonjs/core";

declare global {
  var camera: ArcRotateCamera | undefined;
}

export default function getZoomableOrthoCamera(canvas: HTMLCanvasElement, scene: Scene): ArcRotateCamera {
  if (!canvas) throw new Error("canvas is null. Check the ref for camera.");

  if (!global.camera || global.camera.getScene() !== scene) {
    const cam = new ArcRotateCamera(
      "cam",
      Math.PI / 2,    // alpha
      Math.PI / 2,    // beta
      2,              // radius (won’t matter in ortho)
      Vector3.Zero(), // target
      scene
    );

    cam.mode = Camera.ORTHOGRAPHIC_CAMERA;

    // Initial orthographic size
    let orthoSize = 2;
    setOrthoBounds(cam, orthoSize);

    // Lock radius (disable zoom via radius)
    cam.lowerRadiusLimit = cam.upperRadiusLimit = cam.radius;

    // Attach camera controls
    cam.attachControl(canvas, true);

    // ✅ Native DOM wheel event to control zoom
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();

      // Zoom sensitivity
      orthoSize += event.deltaY * 0.01;

      // Clamp values
      orthoSize = Math.min(Math.max(0.2, orthoSize), 20);

      setOrthoBounds(cam, orthoSize);
    };

    // Attach wheel listener
    canvas.addEventListener("wheel", onWheel, { passive: false });

    // Store camera + cleanup
    global.camera = cam;
    scene.activeCamera = cam;

    // Clean up when scene is disposed
    scene.onDisposeObservable.add(() => {
      canvas.removeEventListener("wheel", onWheel);
    });

    console.log("Orthographic camera created");
  } else {
    scene.activeCamera = global.camera;
    console.log("Orthographic camera re-used");
  }

  return global.camera;
}

function setOrthoBounds(cam: ArcRotateCamera, size: number) {
  cam.orthoLeft = -size;
  cam.orthoRight = size;
  cam.orthoTop = size;
  cam.orthoBottom = -size;
}
