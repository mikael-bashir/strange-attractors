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
      2,              // radius (orthographic projection ignores this)
      Vector3.Zero(), // target
      scene
    );

    cam.mode = Camera.ORTHOGRAPHIC_CAMERA;
    cam.lowerRadiusLimit = cam.upperRadiusLimit = cam.radius; // Lock radius

    // Get initial aspect ratio
    const aspect = canvas.width / canvas.height;
    let orthoHeight = 2;
    
    const setOrthoBounds = () => {
      const orthoWidth = orthoHeight * aspect;
      cam.orthoLeft = -orthoWidth;
      cam.orthoRight = orthoWidth;
      cam.orthoTop = orthoHeight;
      cam.orthoBottom = -orthoHeight;
    };

    // Initial setup
    setOrthoBounds();

    // Mouse wheel zoom handler
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      
      // Zoom with exponential scaling for smooth experience
      const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
      orthoHeight = Math.min(Math.max(0.2, orthoHeight * zoomFactor), 20);

      // Update bounds based on current aspect ratio
      setOrthoBounds();
    };

    // Handle window resize
    const onResize = () => {
      const newAspect = canvas.width / canvas.height;
      if (Math.abs(aspect - newAspect) > 0.01) {
        setOrthoBounds();
      }
    };

    // Attach event listeners
    canvas.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("resize", onResize);

    // Cleanup
    scene.onDisposeObservable.add(() => {
      canvas.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
    });

    cam.attachControl(canvas, true);
    global.camera = cam;
    scene.activeCamera = cam;
  } else {
    scene.activeCamera = global.camera;
  }

  return global.camera!;
}