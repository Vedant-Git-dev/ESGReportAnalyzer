// src/components/ProgressPanel.jsx
import React from 'react'

/**
 * Shown while the pipeline is running.
 * progress1 / progress2 are arrays of log-line strings.
 */
export default function ProgressPanel({ company1, company2, progress1, progress2, error }) {
  return (
    <div>
      {error && (
        <div className="alert alert-error" style={{ marginBottom: 16 }}>
          ⚠️ {error}
        </div>
      )}

      <div className="progress-cols">
        <ProgressCol label={company1} lines={progress1} color="var(--ca)" />
        <ProgressCol label={company2} lines={progress2} color="var(--cb)" />
      </div>
    </div>
  )
}

function ProgressCol({ label, lines, color }) {
  const containerRef = React.useRef(null)

  // Auto-scroll to bottom as new lines arrive
  React.useEffect(() => {
    const el = containerRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])

  const last = lines[lines.length - 1]

  return (
    <div>
      <div className="progress-col-label" style={{ color }}>
        {lines.length > 0
          ? <><span className="spinner" style={{ width: 12, height: 12, marginRight: 6 }} />{label}</>
          : <span style={{ color: 'var(--sub)' }}>⏳ {label}</span>
        }
      </div>

      {last && (
        <div className="status-box status-info" style={{ marginBottom: 8 }}>
          <span className="spinner" style={{ width: 14, height: 14, flexShrink: 0 }} />
          <span>{last}</span>
        </div>
      )}

      {lines.length > 0 && (
        <div
          ref={containerRef}
          style={{ maxHeight: 220, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2 }}
        >
          {lines.map((line, i) => (
            <div key={i} className="log-line">{line}</div>
          ))}
        </div>
      )}
    </div>
  )
}
