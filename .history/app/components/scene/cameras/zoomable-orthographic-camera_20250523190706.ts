import { ArcRotateCamera, Vector3, Camera, Scene } from "@babylonjs/core";

declare global {
  var camera: ArcRotateCamera | undefined;
}

export default function getZoomableOrthoCamera(canvas: HTMLCanvasElement, scene: Scene): ArcRotateCamera {
  if (!canvas) throw new Error("canvas is null. Check the ref for camera.");

  if (!global.camera || global.camera.getScene() !== scene) {
    // Create camera
    global.camera = new ArcRotateCamera(
      "cam",
      Math.PI / 2,    // alpha
      Math.PI / 2,    // beta
      2,              // radius (unused in ortho)
      Vector3.Zero(), // target
      scene
    );

    // Orthographic setup
    const cam = global.camera;
    cam.mode = Camera.ORTHOGRAPHIC_CAMERA;
    
    // Initial zoom level
    let zoomLevel = 0.5;
    const minZoom = 0.1;
    const maxZoom = 20;

    // Set initial bounds
    const updateBounds = () => {
      cam.orthoLeft = -zoomLevel;
      cam.orthoRight = zoomLevel;
      cam.orthoTop = zoomLevel;
      cam.orthoBottom = -zoomLevel;
    };
    updateBounds();

    // Mouse wheel handler
    const wheelHandler = (e: WheelEvent) => {
      e.preventDefault();
      
      // Adjust zoom level exponentially for smooth scaling
      zoomLevel *= e.deltaY > 0 ? 0.9 : 1.1;
      
      // Clamp zoom between min/max
      zoomLevel = Math.max(minZoom, Math.min(maxZoom, zoomLevel));
      
      updateBounds();
    };

    // Add event listeners
    canvas.addEventListener('wheel', wheelHandler, { passive: false });
    cam.attachControl(canvas, true);

    // Cleanup
    cam.onDisposeObservable.add(() => {
      canvas.removeEventListener('wheel', wheelHandler);
    });

    scene.activeCamera = cam;
  }
  
  return global.camera;
}