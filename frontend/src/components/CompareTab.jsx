// src/components/CompareTab.jsx
import React, { useState } from 'react'
import KPICard from './KPICard'
import ProgressPanel from './ProgressPanel'
import { api } from '../lib/api'

const GROUPS = ['Environmental', 'Social', 'Governance']
const GROUP_COLORS = {
  Environmental: 'var(--green)',
  Social:        'var(--blue)',
  Governance:    'var(--amber)',
}

export default function CompareTab({ compareState, form }) {
  const { running, progress1, progress2, result, error } = compareState

  // ── Empty state ─────────────────────────────────────────────────────────
  if (!running && !result && !error) {
    return (
      <div className="hero">
        <div className="hero-icon">🌿</div>
        <div className="hero-title">ESG Competitive Intelligence</div>
        <div className="hero-sub">
          Enter two company names in the sidebar and click Compare to benchmark
          their ESG performance across Environmental, Social, and Governance metrics.
        </div>
        <div className="badge-row">
          <span className="badge badge-blue">Environmental</span>
          <span className="badge badge-green">Social</span>
          <span className="badge badge-amber">Governance</span>
        </div>
      </div>
    )
  }

  // ── Running ──────────────────────────────────────────────────────────────
  if (running && !result) {
    return (
      <div>
        <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>
          Comparing {form.company1} FY{form.fy1} vs {form.company2} FY{form.fy2}
        </h2>
        <ProgressPanel
          company1={form.company1}
          company2={form.company2}
          progress1={progress1}
          progress2={progress2}
          error={error}
        />
      </div>
    )
  }

  // ── Error ────────────────────────────────────────────────────────────────
  if (error && !result) {
    return (
      <div>
        <div className="alert alert-error">⚠️ {error}</div>
        <p style={{ fontSize: 13, color: 'var(--sub)', marginTop: 8 }}>
          Try uploading the report PDFs directly using the Upload PDF tab.
        </p>
      </div>
    )
  }

  // ── Results ──────────────────────────────────────────────────────────────
  if (!result) return null

  const { comparisons = [], summary, label_a, label_b, company1, company2 } = result

  return (
    <ResultsView
      comparisons={comparisons}
      summary={summary}
      labelA={label_a}
      labelB={label_b}
      company1={result.company1}
      company2={result.company2}
      form={form}
    />
  )
}

/* ── Full results view ── */
function ResultsView({ comparisons, summary, labelA, labelB, company1, company2, form }) {
  const [downloading, setDownloading] = useState(false)
  const [logOpen,     setLogOpen]     = useState(false)

  if (!comparisons.length) {
    return (
      <div className="alert alert-warn">
        No comparable ESG metrics found. Reports may use different formats.
      </div>
    )
  }

  async function handleExport() {
    setDownloading(true)
    try {
      const blob = await api.exportPdf({
        company1: labelA.split(' FY')[0],
        fy1:      parseInt(labelA.split(' FY')[1]),
        company2: labelB.split(' FY')[0],
        fy2:      parseInt(labelB.split(' FY')[1]),
        sector:   form.sector,
        summary,
        comparisons: comparisons.map(c => ({
          kpi_name: c.kpi_name,
          entries:  c.entries,
          pct_gap:  c.pct_gap,
          winner:   c.winner,
        })),
      })
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `ESG_${labelA.split(' FY')[0]}_vs_${labelB.split(' FY')[0]}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert('Export failed: ' + e.message)
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div>
      {/* Header */}
      <div className="compare-header">
        <div>
          <div className="compare-title">
            {labelA.split(' FY')[0]}
            <span style={{ color: 'var(--sub)', fontWeight: 400, margin: '0 10px' }}>vs</span>
            {labelB.split(' FY')[0]}
          </div>
          <div className="compare-meta">
            <span className="badge badge-blue">{form.sector}</span>
            &nbsp;&nbsp;{labelA.split(' FY')[1] && `FY${labelA.split(' FY')[1]}`}
            {' · '}
            {labelB.split(' FY')[1] && `FY${labelB.split(' FY')[1]}`}
          </div>
        </div>
        <div className="compare-actions">
          <button
            className="btn btn-ghost btn-sm"
            onClick={handleExport}
            disabled={downloading}
          >
            {downloading ? 'Generating…' : '⬇ Download PDF'}
          </button>
        </div>
      </div>

      {/* KPI groups */}
      {GROUPS.map(group => {
        const groupComps = comparisons.filter(c => c.group === group)
        if (!groupComps.length) return null
        return (
          <div key={group}>
            <div className="sec" style={{ borderBottomColor: GROUP_COLORS[group] }}>
              {group}
            </div>
            {groupComps.map(comp => (
              <KPICard key={comp.kpi_name} comp={comp} labelA={labelA} labelB={labelB} />
            ))}
          </div>
        )
      })}

      {/* Summary */}
      {summary && (
        <>
          <div className="sec" style={{ marginTop: 28 }}>Summary</div>
          <div className="summary-box">{summary}</div>
        </>
      )}

      {/* Pipeline log */}
      <details
        className="log-expander"
        style={{ marginTop: 20 }}
        open={logOpen}
        onToggle={e => setLogOpen(e.target.open)}
      >
        <summary>Processing details</summary>
        <div className="log-body">
          <LogCol label={labelA.split(' FY')[0]} lines={company1?.log || []} />
          <LogCol label={labelB.split(' FY')[0]} lines={company2?.log || []} />
        </div>
      </details>
    </div>
  )
}

function LogCol({ label, lines }) {
  return (
    <div>
      <div className="log-col-label">{label}</div>
      {lines.map((line, i) => (
        <div key={i} className="log-line">{line}</div>
      ))}
    </div>
  )
}
