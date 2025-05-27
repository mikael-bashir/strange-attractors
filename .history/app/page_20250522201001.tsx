'use client'

import VideoAspectFrameWithBorder from "./components/scene/containers/video-aspoect-border";
import dynamic from "next/dynamic";

const AttractorCompute = dynamic(() => import('./scene'), {
    ssr: false
});

const AttractorDensity = dynamic(() => import('./histogram-grid'), {
    ssr: false
});

export default function Index() {
    return(
        <VideoAspectFrameWithBorder
            SceneComponent={() => (
                <AttractorDensity params=
                    {{
                        a: 1.6, 
                        b: -0.6,
                        c: -1.2,
                        d: 1.6, 
                        iterations: 100_000, 
                        x0: 0.1, 
                        y0: 0.1  
                    }}
                />
            )}
        />
    )
}
