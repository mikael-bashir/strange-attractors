'use client'

import { useCreateStore, useControls, button } from 'leva'
import VideoAspectFrameWithBorder from "./components/scene/containers/video-aspoect-border"
import dynamic from "next/dynamic"
import { getCliffordMetrics } from "./lib/functions/lyapunov"
import { useEffect, useState } from "react"

const AttractorHistogram = dynamic(() => import('./components/scene-types/histogram-grid'), {
  ssr: false
})

// 1) Create a standalone Leva store
const levaStore = useCreateStore()

export default function Index() {
  // 2) Your simulation-driving state
  const [submittedParams, setSubmittedParams] = useState({
    a: 400,
    b: 12.6695,
    c: 3,
    d: 0.7,
  })

  // 3) Build controls bound to that store—but ignore their updates
  //    We *don't* destructure the returned values from useControls,
  //    so the component never re-renders on Leva changes.
  useControls(
    {
      a: { value: submittedParams.a },
      b: { value: submittedParams.b },
      c: { value: submittedParams.c },
      d: { value: submittedParams.d, min: -2, max: 2, step: 0.01 },
      Update: button(() => {
        // 4) On button press, *pull* the latest values from the store:
        const state = levaStore.getData() as Record<string, number>
        setSubmittedParams({
          a: state.a,
          b: state.b,
          c: state.c,
          d: state.d,
        })
      }),
    },
    { store: levaStore }
  )

  const [lyapunovData, setLyapunovData] = useState<{
    exponents: number[]
    dimension: number
  } | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    getCliffordMetrics(
      submittedParams.a,
      submittedParams.b,
      submittedParams.c,
      submittedParams.d,
    )
      .then(results => {
        setLyapunovData({
          exponents: results.lyapunovExponents,
          dimension: results.fractalDimension,
        })
      })
      .catch(console.error)
      .finally(() => setLoading(false))
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
