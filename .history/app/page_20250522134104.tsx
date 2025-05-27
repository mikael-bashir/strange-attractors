'use client'

import VideoAspectFrameWithBorder from "./components/scene/containers/video-aspoect-border";
import dynamic from "next/dynamic";

const AttractorCompute = dynamic(() => import('./scene'), {
    ssr: false
});

export default function Index() {
    return(
        <VideoAspectFrameWithBorder
            SceneComponent={() => (
                <AttractorCompute params=
                    {{   
                        a: -1.4, 
                        b: 1.6,
                        c: 1.0,
                        d: 0.7, 
                        iterations: 1024, 
                        x0: 0.1, 
                        y0: 0.1  
                    }} 
                />
            )}
        />
    )
}