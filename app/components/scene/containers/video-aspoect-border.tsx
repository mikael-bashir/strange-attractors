import type { ComponentType } from "react";

export default function VideoAspectFrameWithBorder({ SceneComponent }: { SceneComponent: ComponentType}) {
    return (
        <div className="pt-[11vh]">
            <div className="relative w-[800px] h-[450px] mx-auto bg-white border-1 border-white">
                <SceneComponent />
            </div>
        </div>
    )
}
