// src/components/ProgressPanel.jsx
import React from 'react'

const STEPS = [
  { key: 'ingest', label: 'Ingesting report', icon: '📥' },
  { key: 'parse', label: 'Parsing PDF', icon: '📄' },
  { key: 'chunk', label: 'Chunking content', icon: '✂️' },
  { key: 'embed', label: 'Generating embeddings', icon: '🧠' },
  { key: 'extract', label: 'Extracting KPIs', icon: '📊' },
]

function detectStep(line) {
  const l = line.toLowerCase()
  if (l.includes('ingest')) return 'ingest'
  if (l.includes('parse')) return 'parse'
  if (l.includes('chunk')) return 'chunk'
  if (l.includes('embed') || l.includes('embedding')) return 'embed'
  if (l.includes('extract') || l.includes('kpi')) return 'extract'
  return null
}

export default function ProgressPanel({ company1, company2, progress1, progress2, error }) {
  return (
    <div className="progress-panel">
      {error && (
        <div className="alert alert-error" style={{ marginBottom: 20 }}>
          <span style={{ fontSize: 18 }}>⚠</span>
          <div>
            <strong>Error</strong>
            <p style={{ marginTop: 4, fontSize: 13, opacity: 0.9 }}>{error}</p>
          </div>
        </div>
      )}

      <div className="progress-cols">
        <ProgressCol label={company1} lines={progress1} accentColor="var(--brand)" />
        <ProgressCol label={company2} lines={progress2} accentColor="var(--accent-copper)" />
      </div>
    </div>
  )
}

function ProgressCol({ label, lines, accentColor }) {
  const containerRef = React.useRef(null)

  React.useEffect(() => {
    const el = containerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])

  const last = lines[lines.length - 1]
  const lastStep = last ? detectStep(last) : null

  // Determine completed steps
  const completedSteps = new Set()
  lines.forEach(line => {
    const step = detectStep(line)
    if (step) completedSteps.add(step)
  })

  return (
    <div>
      <div className="progress-col-label" style={{ color: accentColor }}>
        <span style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: accentColor,
          display: 'inline-block',
          marginRight: 8,
        }} />
        {label}
      </div>

      {/* Steps indicator */}
      <div className="progress-steps" style={{ marginTop: 12, marginBottom: 16 }}>
        {STEPS.map((step, i) => {
          const isDone = completedSteps.has(step.key)
          const isCurrent = lastStep === step.key
          const isPending = !isDone && !isCurrent

          let status = 'pending'
          if (isDone && (i === STEPS.length - 1 || !completedSteps.has(STEPS[i + 1]?.key))) {
            status = 'done'
          } else if (isCurrent) {
            status = 'running'
          } else if (isDone) {
            status = 'done'
          }

          return (
            <div key={step.key} className={`progress-step ${status}`}>
              <span className="progress-step-icon">
                {status === 'done' ? '✓' : status === 'running' ? '●' : '○'}
              </span>
              <span style={{ opacity: isPending ? 0.5 : 1 }}>{step.label}</span>
            </div>
          )
        })}
      </div>

      {/* Current status */}
      {last && (
        <div className="status-box status-info" style={{ marginBottom: 12 }}>
          <span className="spinner" style={{ width: 14, height: 14 }} />
          <span style={{ fontWeight: 500 }}>{last}</span>
        </div>
      )}

      {/* Log lines */}
      {lines.length > 0 && (
        <div
          ref={containerRef}
          style={{
            maxHeight: 180,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}
        >
          {lines.slice(-10).map((line, i) => (
            <div key={i} className="log-line" style={{
              fontSize: 10,
              padding: '3px 6px',
              opacity: i === lines.slice(-10).length - 1 ? 1 : 0.6
            }}>
              {line}
            </div>
          ))}
        </div>
      )}

      {lines.length === 0 && (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' }}>
          Waiting to start...
        </p>
      )}
    </div>
  )
}
