'use client'

import { useControls } from "leva";
import VideoAspectFrameWithBorder from "./components/scene/containers/video-aspoect-border";
import dynamic from "next/dynamic";

const AttractorCompute = dynamic(() => import('./scene'), {
    ssr: false
});

const AttractorDensity = dynamic(() => import('./histogram-grid'), {
    ssr: false
});

export default function Index() {
    const params = useControls({
        a: { value: 1.4, min: -2, max: 2, step: 0.01 },
        b: { value: 0.3, min: -2, max: 2, step: 0.01 },
        c: { value: 1.0, min: -2, max: 2, step: 0.01 },
        d: { value: 0.7, min: -2, max: 2, step: 0.01 },
        iterations: { value: 10000000, min: 10000, max: 10000000, step: 10000 },
        x0: { value: 0.631, min: -2, max: 2, step: 0.01 },
        y0: { value: 0.189, min: -2, max: 2, step: 0.01 }
    })
    
    return(
        <VideoAspectFrameWithBorder
            SceneComponent={() => (
                <AttractorDensity params=
                    {params}
                    
                    //                     {{
                    //     a: 1.4, 
                    //     b: 0.3,
                    //     c: 1.0,
                    //     d: 0.7, 
                    //     iterations: 100_000, 
                    //     x0: 0.631, 
                    //     y0: 0.189  
                    // }}
                />
            )}
        />
    )
}
