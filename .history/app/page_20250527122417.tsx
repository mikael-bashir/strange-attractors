'use client'

import { useEffect, useState } from 'react'
import VideoAspectFrameWithBorder from "./components/scene/containers/video-aspoect-border"
import dynamic from "next/dynamic"
import { getCliffordMetrics } from "./lib/functions/lyapunov"

const AttractorHistogram = dynamic(() => import('./components/scene-types/histogram-grid'), {
  ssr: false
})

export default function Index() {
  // Simulation parameters (only updated on button click)
  const [submittedParams, setSubmittedParams] = useState<{
    a: number,
    b: number,
    c: number,
    d: number
  } | null>(null)  // Start with null to prevent initial calculation

  // Form state (local edits not sent to simulator)
  const [form, setForm] = useState({
    a: 400,
    b: 12.6695,
    c: 3,
    d: 0.7,
  })

  const updateParam = (key: keyof typeof form, value: number) => {
    setForm(prev => ({ ...prev, [key]: value }))
  }

  const handleUpdate = () => {
    // Only update submitted params when button is clicked
    setSubmittedParams({ ...form })
  }

  const [lyapunovData, setLyapunovData] = useState<{
    exponents: number[]
    dimension: number
  } | null>(null)

  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const run = async () => {
      // Don't run if no submitted params
      if (!submittedParams) return

      setLoading(true)
      try {
        const result = await getCliffordMetrics(
          submittedParams.a,
          submittedParams.b,
          submittedParams.c,
          submittedParams.d
        )
        setLyapunovData({
          exponents: result.lyapunovExponents,
          dimension: result.fractalDimension,
        })
      } catch (err) {
        console.error("Metric error", err)
      } finally {
        setLoading(false)
      }
    }

    run()
  }, [submittedParams])  // Only trigger when submittedParams changes

  return (
    <div className="relative w-full h-full">
      <VideoAspectFrameWithBorder
        SceneComponent={() => submittedParams ? (
          <AttractorHistogram
            {...submittedParams}
            width={3840}
            height={2160}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-white/50">
            Click "Update Simulation" to start
          </div>
        )}
      />

      {/* Controls */}
      <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg p-4 text-sm font-mono text-white shadow-xl space-y-2">
        <h3 className="text-lg font-bold mb-2">Manual Controls</h3>

        {['a', 'b', 'c', 'd'].map(key => (
          <div key={key} className="flex justify-between items-center gap-2">
            <label htmlFor={key}>{key}:</label>
            <input
              id={key}
              type="number"
              step={key === 'd' ? 0.01 : 1}
              min={key === 'd' ? -2 : undefined}
              max={key === 'd' ? 2 : undefined}
              value={form[key as keyof typeof form]}
              onChange={e =>
                updateParam(key as keyof typeof form, parseFloat(e.target.value))
              }
              className="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded"
            />
          </div>
        ))}

        <button
          onClick={handleUpdate}
          className="mt-2 w-full bg-cyan-600 hover:bg-cyan-700 py-1 rounded font-bold"
        >
          Update Simulation
        </button>
      </div>

      {/* Metrics */}
      <div className="absolute top-4 right-4 bg-black/80 backdrop-blur-sm rounded-lg p-4 text-sm font-mono text-white shadow-xl">
        <h3 className="text-lg font-bold mb-2">System Metrics</h3>
        
        {!submittedParams ? (
          <div className="text-gray-400">No simulation data yet</div>
        ) : loading ? (
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
          // ... rest of metrics display remains the same
        )}
      </div>
    </div>
  )
}