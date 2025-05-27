'use client'

import { useControls, button } from 'leva'
import VideoAspectFrameWithBorder from "./components/scene/containers/video-aspoect-border"
import dynamic from "next/dynamic"
import { getCliffordMetrics } from "./lib/functions/lyapunov"
import { useEffect, useState } from "react"

const AttractorHistogram = dynamic(() => import('./components/scene-types/histogram-grid'), {
  ssr: false
})

export default function Index() {
  // Local state holds the “live” params used for compute
  const [submittedParams, setSubmittedParams] = useState({
    a: 400,
    b: 12.6695,
    c: 3,
    d: 0.7,
  })

  // Leva controls produce a temp object and an Update button
  const temp = useControls({
    a: { value: submittedParams.a },
    b: { value: submittedParams.b },
    c: { value: submittedParams.c },
    d: { value: submittedParams.d, min: -2, max: 2, step: 0.01 },
    Update: button(() => {
      // When you click “Update”, copy the temp values into submittedParams
      setSubmittedParams({
        a: temp.a,
        b: temp.b,
        c: temp.c,
        d: temp.d,
      })
    })
  })

  const [lyapunovData, setLyapunovData] = useState<{
    exponents: number[]
    dimension: number
  } | null>(null)
  const [loading, setLoading] = useState(false)

  // Only re-run when submittedParams changes
  useEffect(() => {
    const calculateMetrics = async () => {
      setLoading(true)
      try {
        const results = await getCliffordMetrics(
          submittedParams.a,
          submittedParams.b,
          submittedParams.c,
          submittedParams.d,
        )
        setLyapunovData({
          exponents: results.lyapunovExponents,
          dimension: results.fractalDimension,
        })
      } catch (error) {
        console.error("Error calculating metrics:", error)
      } finally {
        setLoading(false)
      }
    }

    calculateMetrics()
  }, [submittedParams])

  return (
    <div className="relative w-full h-full">
      <VideoAspectFrameWithBorder
        SceneComponent={() => (
          <AttractorHistogram
            {...submittedParams}
            width={3840}
            height={2160}
          />
        )}
      />

      {/* Metrics overlay */}
      <div className="absolute top-4 right-4 bg-black/80 backdrop-blur-sm rounded-lg p-4 text-sm font-mono text-white shadow-xl">
        <h3 className="text-lg font-bold mb-2">System Metrics</h3>

        {loading ? (
          <div className="animate-pulse">Calculating...</div>
        ) : lyapunovData ? (
          <div className="space-y-2">
            <div className="flex justify-between gap-4">
              <span>Lyapunov Exponents:</span>
              <div className="text-right">
                {lyapunovData.exponents.map((exp, i) => (
                  <div key={i} className="text-cyan-300">
                    λ{i + 1}: {exp.toFixed(3)}
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
        ) : (
          <div className="text-red-400">Metrics unavailable</div>
        )}
      </div>
    </div>
  )
}
