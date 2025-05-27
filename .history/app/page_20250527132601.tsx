'use client'

import { useEffect, useState, useRef } from 'react'
import VideoAspectFrameWithBorder from "./components/scene/containers/video-aspoect-border"
import dynamic from "next/dynamic"
import { getCliffordMetrics } from "./lib/functions/lyapunov"

const AttractorHistogram = dynamic(() => import('./components/scene-types/histogram-grid'), {
  ssr: false
})

export default function Index() {
  // Simulation parameters
  const [a, setA] = useState(400)
  const [b, setB] = useState(12.6695)
  const [c, setC] = useState(3)
  const [d, setD] = useState(0.7)
  let submittedParams = { a: 400, b: 12.6695, c: 4, d: 0.7 }

  // Form states
  const [aForm, setAForm] = useState<number | null>(400)
  const [bForm, setBForm] = useState<number | null>(12.6695)
  const [cForm, setCForm] = useState<number | null>(3)
  const [dForm, setDForm] = useState<number | null>(0.7)

  const [lyapunovData, setLyapunovData] = useState<{
    exponents: number[],
    dimension: number
  } | null>(null)

  const [loading, setLoading] = useState(false)

  const handleUpdate = async () => {
    // Only update actual params if form values exist
    if (aForm == null || bForm == null || cForm == null || dForm == null) return;
    if (aForm !== null) setA(aForm);
    if (bForm !== null) setB(bForm);
    if (cForm !== null) setC(cForm);
    if (dForm !== null) setD(dForm);
    setLoading(true)
      try {
        const result = await getCliffordMetrics(aForm, bForm, cForm, dForm);
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

//   useEffect(() => {

//     const runSimulation = async () => {
//       setLoading(true)
//       try {
//         const result = await getCliffordMetrics(a, b, c, d)
//         setLyapunovData({
//           exponents: result.lyapunovExponents,
//           dimension: result.fractalDimension,
//         })
//       } catch (err) {
//         console.error("Metric error", err)
//       } finally {
//         setLoading(false)
//       }
//     }

//     runSimulation()
//   }, [a, b, c, d]) // Only runs when actual params change

  return (
    <div className="relative w-full h-full">
      <VideoAspectFrameWithBorder
        SceneComponent={() => (
          <AttractorHistogram
            a={a}
            b={b}
            c={c}
            d={d}
            width={3840}
            height={2160}
          />
        )}
      />

      {/* Controls */}
      <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg p-4 text-sm font-mono text-white shadow-xl space-y-2">
        <h3 className="text-lg font-bold mb-2">Manual Controls</h3>

        {[
          { key: 'a', form: aForm, setter: setAForm },
          { key: 'b', form: bForm, setter: setBForm },
          { key: 'c', form: cForm, setter: setCForm },
          { key: 'd', form: dForm, setter: setDForm }
        ].map(({ key, form, setter }) => (
          <div key={key} className="flex justify-between items-center gap-2">
            <label htmlFor={key}>{key}:</label>
            <input
              id={key}
              type="number"
              step={key === 'd' ? 0.01 : 1}
              min={key === 'd' ? -2 : undefined}
              max={key === 'd' ? 2 : undefined}
              value={form ?? ''}
              onChange={e => setter(parseFloat(e.target.value))}
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