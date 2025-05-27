'use client'

import { useControls } from "leva";
import VideoAspectFrameWithBorder from "./components/scene/containers/video-aspoect-border";
import dynamic from "next/dynamic";

const AttractorCompute = dynamic(() => import('./components/scene-types/point-map'), {
    ssr: false
});

const AttractorDensity = dynamic(() => import('./components/scene-types/line-map'), {
    ssr: false
});

const AttractorHistogram = dynamic(() => import('./components/scene-types/histogram-grid'), {
    ssr: false
});

export default function Index() {
    // const params = useControls({
    //     a: { value: 1.4, min: -2, max: 2, step: 0.01 },
    //     b: { value: 0.3, min: -2, max: 2, step: 0.01 },
    //     c: { value: 1.0, min: -2, max: 2, step: 0.01 },
    //     d: { value: 0.7, min: -2, max: 2, step: 0.01 },
    //     iterations: { value: 10000000, min: 10000, max: 10000000, step: 10000 },
    //     x0: { value: 0.631, min: -2, max: 2, step: 0.01 },
    //     y0: { value: 0.189, min: -2, max: 2, step: 0.01 }
    // })

    const histogramParams = useControls({
        a: { value: -1.4, min: -2, max: 2, step: 0.01 },
        b: { value: 1.6, min: -2, max: 2, step: 0.01 },
        c: { value: 1.0, min: -2, max: 2, step: 0.01 },
        d: { value: 0.7, min: -2, max: 2, step: 0.01 }
    })
    
    return(
        <VideoAspectFrameWithBorder
            SceneComponent={() => (
                // <AttractorDensity params=
                //     {params}
                // />
                <AttractorHistogram 
                    {...histogramParams} 
                />
            )}
        />
    )
}
