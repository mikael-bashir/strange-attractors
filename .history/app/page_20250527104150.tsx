'use client'

import { useControls } from "leva";
import VideoAspectFrameWithBorder from "./components/scene/containers/video-aspoect-border";
import dynamic from "next/dynamic";
import { getCliffordMetrics } from "./lib/functions/lyapunov";
import { useEffect, useState } from "react";

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

    // const params = useControls({
    //     a: { value: -1.4, min: -2, max: 2, step: 0.01 },
    //     b: { value: 1.6, min: -2, max: 2, step: 0.01 },
    //     c: { value: 1.0, min: -2, max: 2, step: 0.01 },
    //     d: { value: 0.7, min: -2, max: 2, step: 0.01 },
    // })

    // const params = useControls({
    //     a: { value: 1.4, min: -2, max: 2, step: 0.01 },
    //     b: { value: 0.3, min: -2, max: 2, step: 0.01 },
    //     c: { value: 1.0, min: -2, max: 2, step: 0.01 },
    //     d: { value: 0.7, min: -2, max: 2, step: 0.01 },
    // })

    const params = useControls({
        a: { value: 1.7, min: -2, max: 2, step: 0.01 },
        b: { value: 0.5, min: -2, max: 2, step: 0.01 },
        c: { value: 1.0, min: -2, max: 2, step: 0.01 },
        d: { value: 0.7, min: -2, max: 2, step: 0.01 },
    })

    const [lyapunovData, setLyapunovData] = useState<{
        exponents: number[];
        dimension: number;
    } | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const calculateMetrics = async () => {
            setLoading(true);
            try {
                const results = await getCliffordMetrics(
                    params.a,
                    params.b,
                    params.c,
                    params.d
                );
                setLyapunovData({ exponents: results.lyapunovExponents, dimension: results.fractalDimension });
            } catch (error) {
                console.error("Error calculating metrics:", error);
            } finally {
                setLoading(false);
            }
        };

        calculateMetrics();
    }, [params.a, params.b, params.c, params.d]);
    
    return(
        <div className="relative w-full h-full">
            <VideoAspectFrameWithBorder
                SceneComponent={() => (
                    // <AttractorDensity params=
                    //     {params}
                    // />
                    <AttractorHistogram 
                        {...params} width={3840} height={2160}
                    />
                    // <AttractorDensity
                    //     params={histogramParams}
                    // />
                    // <AttractorCompute
                    //     params={params}
                    // />
                )}
            />
            {/* Metrics overlay */}
            <div className="absolute top-4 right-4 bg-black/80 backdrop-blur-sm rounded-lg p-4 text-sm font-mono text-white shadow-xl">
                <h3 className="text-lg font-bold mb-2">System Metrics</h3>
                
                {loading ? (
                    <div className="animate-pulse">Calculating...</div>
                ) : lyapunovData ? (
                    <>
                        <div className="space-y-2">
                            <div className="flex justify-between gap-4">
                                <span>Lyapunov Exponents:</span>
                                <div className="text-right">
                                    {lyapunovData.exponents.map((exp, i) => (
                                        <div key={i} className="text-cyan-300">
                                            λ{i+1}: {exp.toFixed(3)}
                                        </div>
                                    ))}
                                </div>
                            </div>
                            
                            <div className="pt-2 border-t border-white/20">
                                <div className="flex justify-between">
                                    <span>Fractal Dimension:</span>
                                    <span className="text-purple-300">
                                        {lyapunovData.dimension.toFixed(3)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </>
                ) : (
                    <div className="text-red-400">Metrics unavailable</div>
                )}
            </div>
        </div>
    )
}
