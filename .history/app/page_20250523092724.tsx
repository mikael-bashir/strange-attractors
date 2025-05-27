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
                        a: 1.4, 
                        b: 0.3,
                        c: 1.0,
                        d: 0.7, 
                        iterations: 100_000, 
                        x0: 0.631, 
                        y0: 0.189  
                    }}
                />
            )}
        />
    )
}
