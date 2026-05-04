// src/components/CompareTab.jsx
import React, { useState } from 'react'
import KPICard from './KPICard'
import ProgressPanel from './ProgressPanel'
import { api } from '../lib/api'

const GROUPS = [
  { key: 'environmental', label: 'Environmental' },
  { key: 'social', label: 'Social' },
  { key: 'governance', label: 'Governance' },
]
const GROUP_COLORS = {
  environmental: 'var(--green)',
  social:        'var(--blue)',
  governance:    'var(--amber)',
}

const groupKey = (value) => String(value || '').trim().toLowerCase()

export default function CompareTab({ compareState, form }) {
  const { running, progress1, progress2, result, error } = compareState

  // ── Empty state ────────────────────────────────────────────────────────────
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

  // ── Running ────────────────────────────────────────────────────────────────
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

  // ── Error ──────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// ResultsView — hosts the toggle and passes mode to every KPICard
// ─────────────────────────────────────────────────────────────────────────────
function ResultsView({ comparisons, summary, labelA, labelB, company1, company2, form }) {
  const [normalized,  setNormalized]  = useState(true)   // false = Absolute, true = Normalized
  const [downloading, setDownloading] = useState(false)
  const [logOpen,     setLogOpen]     = useState(false)

  if (!comparisons.length) {
    return (
      <div className="alert alert-warn">
        No comparable ESG metrics found. Reports may use different formats.
      </div>
    )
  }

  // Raw absolute kpi_records keyed by kpi_name for each company
  // Shape: { kpi_name: { value, unit, method, confidence } }
  const rawA = company1?.kpi_records || {}
  const rawB = company2?.kpi_records || {}

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
      {/* ── Header row ── */}
      <div className="compare-header">
        <div>
          <div className="compare-title">
            {labelA.split(' FY')[0]}
            <span style={{ color: 'var(--sub)', fontWeight: 400, margin: '0 10px' }}>vs</span>
            {labelB.split(' FY')[0]}
          </div>
          <div className="compare-meta">
            <span className="badge badge-blue">{form.sector}</span>
            &nbsp;&nbsp;
            {labelA.split(' FY')[1] && `FY${labelA.split(' FY')[1]}`}
            {' · '}
            {labelB.split(' FY')[1] && `FY${labelB.split(' FY')[1]}`}
          </div>
        </div>

        <div className="compare-actions">
          {/* ── Mode Toggle ── */}
          <ModeToggle normalized={normalized} onChange={setNormalized} />

          <button
            className="btn btn-ghost btn-sm"
            onClick={handleExport}
            disabled={downloading}
          >
            {downloading ? 'Generating…' : '⬇ Download PDF'}
          </button>
        </div>
      </div>

      {/* ── Mode explanation banner ── */}
      <ModeBanner normalized={normalized} />

      {/* ── KPI groups ── */}
      {GROUPS.map(group => {
        const groupComps = comparisons.filter(c => groupKey(c.group) === group.key)
        if (!groupComps.length) return null
        return (
          <div key={group.key}>
            <div className="sec" style={{ borderBottomColor: GROUP_COLORS[group.key] }}>
              {group.label}
            </div>
            {groupComps.map(comp => (
              <KPICard
                key={comp.kpi_name}
                comp={comp}
                labelA={labelA}
                labelB={labelB}
                normalized={normalized}
                rawA={rawA[comp.kpi_name] || null}
                rawB={rawB[comp.kpi_name] || null}
              />
            ))}
          </div>
        )
      })}

      {/* ── Summary ── */}
      {summary && (
        <>
          <div className="sec" style={{ marginTop: 28 }}>Summary</div>
          <div className="summary-box">{summary}</div>
        </>
      )}

      {/* ── Pipeline log ── */}
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

// ─────────────────────────────────────────────────────────────────────────────
// ModeToggle — pill-style switch between Absolute and Normalized
// ─────────────────────────────────────────────────────────────────────────────
function ModeToggle({ normalized, onChange }) {
  return (
    <div className="mode-toggle-wrap">
      <span className={`mode-label ${!normalized ? 'mode-label-active' : ''}`}>
        Absolute
      </span>

      {/* The actual toggle track + thumb */}
      <button
        className={`toggle-track ${normalized ? 'toggle-on' : ''}`}
        onClick={() => onChange(!normalized)}
        role="switch"
        aria-checked={normalized}
        aria-label="Switch between absolute and normalized comparison"
      >
        <span className="toggle-thumb" />
      </button>

      <span className={`mode-label ${normalized ? 'mode-label-active' : ''}`}>
        Normalized
      </span>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ModeBanner — contextual explanation below the header
// ─────────────────────────────────────────────────────────────────────────────
function ModeBanner({ normalized }) {
  if (normalized) {
    return (
      <div className="mode-banner mode-banner-norm">
        <span className="mode-banner-icon">📊</span>
        <div>
          <strong>Normalized view</strong> — KPI values divided by annual revenue (INR Crore).
          Enables fair comparison between companies of different sizes.
          Units shown as <em>value / Crore</em>.
        </div>
      </div>
    )
  }
  return (
    <div className="mode-banner mode-banner-abs">
      <span className="mode-banner-icon">🔢</span>
      <div>
        <strong>Absolute view</strong> — Raw reported values in original units
        (tCO2e, GJ, KL, MT, %, headcount). Direct comparison of reported figures.
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// LogCol
// ─────────────────────────────────────────────────────────────────────────────
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