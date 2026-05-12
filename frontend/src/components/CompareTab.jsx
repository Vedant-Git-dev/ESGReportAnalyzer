// src/components/CompareTab.jsx
import React, { useState } from 'react'
import KPICard from './KPICard'
import ProgressPanel from './ProgressPanel'
import { api } from '../lib/api'

const GROUPS = [
  { key: 'financial', label: 'Financial', icon: '💰' },
  { key: 'environmental', label: 'Environmental', icon: '🌱' },
  { key: 'social', label: 'Social', icon: '👥' },
  { key: 'governance', label: 'Governance', icon: '⚖' },
]

const FINANCIAL_KPI_NAMES = ['revenue_from_operations', 'net_revenue']

const CONTENT_TABS = [
  { key: 'kpis', label: 'KPIs', icon: '📊' },
  { key: 'summary', label: 'Summary', icon: '📝' },
  { key: 'recommendations', label: 'Recommendations', icon: '💡' },
]

const GROUP_STYLES = {
  financial: { color: 'var(--fin-color)', bg: 'rgba(75, 107, 138, 0.1)' },
  environmental: { color: 'var(--env-color)', bg: 'rgba(26, 77, 64, 0.1)' },
  social: { color: 'var(--soc-color)', bg: 'rgba(46, 107, 138, 0.1)' },
  governance: { color: 'var(--gov-color)', bg: 'rgba(123, 94, 60, 0.1)' },
}

const groupKey = (value) => String(value || '').trim().toLowerCase()

export default function CompareTab({ compareState, form }) {
  const { running, progress1, progress2, result, error } = compareState

  // Empty state
  if (!running && !result && !error) {
    return <EmptyState />
  }

  // Running
  if (running && !result) {
    return (
      <div>
        <h2 style={{ fontSize: 20, fontWeight: 600, marginBottom: 20, color: 'var(--text-primary)' }}>
          Comparing {form.company1} (FY{form.fy1}) vs {form.company2} (FY{form.fy2})
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

  // Error
  if (error && !result) {
    return (
      <div>
        <div className="alert alert-error">
          <span style={{ fontSize: 18 }}>⚠</span>
          <div>
            <strong>Analysis Failed</strong>
            <p style={{ marginTop: 4, fontSize: 13, opacity: 0.9 }}>{error}</p>
          </div>
        </div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 12 }}>
          Try uploading the report PDFs directly using the Upload tab, or check that the company names are correct.
        </p>
      </div>
    )
  }

  if (!result) return null

  return (
    <ResultsView
      comparisons={result.comparisons || []}
      summary={result.summary}
      recommendation={result.recommendation}
      labelA={result.label_a}
      labelB={result.label_b}
      company1={result.company1}
      company2={result.company2}
      form={form}
    />
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty State
// ─────────────────────────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div className="hero">
      <div className="hero-illustration">
        <div className="hero-illustration-icon">📊</div>
      </div>
      <h1 className="hero-title">ESG Competitive Intelligence</h1>
      <p className="hero-subtitle">
        Benchmark companies on their Environmental, Social, and Governance performance.
        Select two companies from the sidebar and click Compare to begin analysis.
      </p>
      <div className="hero-badges">
        <span className="esg-badge env">
          <span className="esg-badge-icon">🌱</span>
          Environmental
        </span>
        <span className="esg-badge soc">
          <span className="esg-badge-icon">👥</span>
          Social
        </span>
        <span className="esg-badge gov">
          <span className="esg-badge-icon">⚖</span>
          Governance
        </span>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Results View
// ─────────────────────────────────────────────────────────────────────────────
function ResultsView({ comparisons, summary, recommendation, labelA, labelB, company1, company2, form }) {
  const [normalized, setNormalized] = useState(true)
  const [downloading, setDownloading] = useState(false)
  const [logOpen, setLogOpen] = useState(false)
  const [selectedGroup, setSelectedGroup] = useState('environmental')
  const [selectedContentTab, setSelectedContentTab] = useState('kpis')

  if (!comparisons.length) {
    return (
      <div className="alert alert-warn">
        <span style={{ fontSize: 18 }}>📋</span>
        <div>
          <strong>No comparable metrics found</strong>
          <p style={{ marginTop: 4, fontSize: 13, opacity: 0.9 }}>
            Reports may use different formats or KPI definitions. Try uploading source PDFs.
          </p>
        </div>
      </div>
    )
  }

  const rawA = company1?.kpi_records || {}
  const rawB = company2?.kpi_records || {}

  async function handleExport() {
    setDownloading(true)
    try {
      const blob = await api.exportPdf({
        company1: labelA.split(' FY')[0],
        fy1: parseInt(labelA.split(' FY')[1]),
        company2: labelB.split(' FY')[0],
        fy2: parseInt(labelB.split(' FY')[1]),
        sector: form.sector,
        summary,
        comparisons: comparisons.map(c => ({
          kpi_name: c.kpi_name,
          entries: c.entries,
          pct_gap: c.pct_gap,
          winner: c.winner,
        })),
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `ESG_${labelA.split(' FY')[0]}_vs_${labelB.split(' FY')[0]}.pdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert('Export failed: ' + e.message)
    } finally {
      setDownloading(false)
    }
  }

  const groupComps = comparisons.filter(c => groupKey(c.group) === selectedGroup)
  // For Financial tab, only show revenue_from_operations and net_revenue
  const filteredGroupComps = selectedGroup === 'financial'
    ? groupComps.filter(c => FINANCIAL_KPI_NAMES.includes(c.kpi_name))
    : groupComps
  const selectedGroupObj = GROUPS.find(g => g.key === selectedGroup)
  const groupStyle = GROUP_STYLES[selectedGroup]
  const activeComps = selectedGroup === 'financial' ? filteredGroupComps : groupComps

  return (
    <div>
      {/* Header */}
      <div className="compare-header">
        <div>
          <div className="compare-title-block">
            <span className="company-title">{labelA.split(' FY')[0]}</span>
            <span className="vs-label">vs</span>
            <span className="company-title">{labelB.split(' FY')[0]}</span>
          </div>
          <div className="compare-meta">
            <span className="sector-badge">{form.sector}</span>
            <span className="fy-label">
              FY{labelA.split(' FY')[1]} · FY{labelB.split(' FY')[1]}
            </span>
          </div>
        </div>

        <div className="compare-actions">
          <ModeToggle normalized={normalized} onChange={setNormalized} />
          <button
            className="btn btn-ghost btn-sm"
            onClick={handleExport}
            disabled={downloading}
          >
            {downloading ? '⏳ Generating...' : '⬇ Export PDF'}
          </button>
        </div>
      </div>

      {/* Mode Banner */}
      <ModeBanner normalized={normalized} selectedGroup={selectedGroup} />

      {/* Content Tabs */}
      <ContentTabs
        selectedTab={selectedContentTab}
        onSelectTab={setSelectedContentTab}
        summary={summary}
        recommendation={recommendation}
      />

      {/* KPI Content - only show when KPIs tab is selected */}
      {selectedContentTab === 'kpis' && (
        <>
          {/* ESG Tabs */}
          <ESGTabs
            selectedGroup={selectedGroup}
            onSelectGroup={setSelectedGroup}
            comparisons={comparisons}
          />

          {/* KPI Cards */}
          {activeComps.length > 0 ? (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              {activeComps.map(comp => (
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
          ) : (
            <div className="alert alert-info">
              <span style={{ fontSize: 18 }}>📊</span>
              <div>
                <strong>No {selectedGroupObj?.label} KPIs found</strong>
                <p style={{ marginTop: 4, fontSize: 13, opacity: 0.9 }}>
                  {selectedGroup === 'financial'
                    ? 'Revenue data is being loaded. Try comparing companies.'
                    : 'Try switching to a different ESG category above.'}
                </p>
              </div>
            </div>
          )}
        </>
      )}

      {/* Summary Content - only show when Summary tab is selected */}
      {selectedContentTab === 'summary' && summary && (
        <>
          <div className="sec">Executive Summary</div>
          <div className="summary-box">{summary}</div>
        </>
      )}

      {/* Recommendations Content - only show when Recommendations tab is selected */}
      {selectedContentTab === 'recommendations' && recommendation && (
        <>
          <div className="sec">Investment Recommendations</div>
          <div className="summary-box">{recommendation}</div>
        </>
      )}

      {/* Pipeline Logs */}
      <details
        className="log-expander"
        style={{ marginTop: 24 }}
        open={logOpen}
        onToggle={e => setLogOpen(e.target.open)}
      >
        <summary>Processing Details</summary>
        <div className="log-body">
          <LogCol label={labelA.split(' FY')[0]} lines={company1?.log || []} />
          <LogCol label={labelB.split(' FY')[0]} lines={company2?.log || []} />
        </div>
      </details>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ESGTabs
// ─────────────────────────────────────────────────────────────────────────────
function ESGTabs({ selectedGroup, onSelectGroup, comparisons }) {
  return (
    <div className="esg-tabs-container">
      <div className="esg-tabs">
        {GROUPS.map(group => {
          const groupComps = comparisons.filter(c => groupKey(c.group) === group.key)
          const count = groupComps.length
          const isSelected = selectedGroup === group.key

          return (
            <button
              key={group.key}
              className={`esg-tab ${group.key.substring(0, 3)} ${isSelected ? 'esg-tab-active' : ''}`}
              onClick={() => onSelectGroup(group.key)}
              style={isSelected ? {
                color: GROUP_STYLES[group.key].color,
              } : {}}
            >
              <span>{group.icon}</span>
              <span>{group.label}</span>
              <span className="esg-tab-count">{count}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ContentTabs (Summary, Recommendations, KPIs)
// ─────────────────────────────────────────────────────────────────────────────
function ContentTabs({ selectedTab, onSelectTab, summary, recommendation }) {
  return (
    <div className="content-tabs-container" style={{ marginBottom: 20 }}>
      <div className="content-tabs">
        {CONTENT_TABS.map(tab => {
          // Hide tab if no content
          if (tab.key === 'summary' && !summary) return null
          if (tab.key === 'recommendations' && !recommendation) return null

          const isSelected = selectedTab === tab.key

          return (
            <button
              key={tab.key}
              className={`content-tab ${isSelected ? 'content-tab-active' : ''}`}
              onClick={() => onSelectTab(tab.key)}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// ModeToggle
// ─────────────────────────────────────────────────────────────────────────────
function ModeToggle({ normalized, onChange }) {
  return (
    <div className="mode-toggle-wrap">
      <span className={`mode-label ${!normalized ? 'mode-label-active' : ''}`}>
        Absolute
      </span>
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
// ModeBanner
// ─────────────────────────────────────────────────────────────────────────────
function ModeBanner({ normalized, selectedGroup }) {
  if (normalized) {
    return (
      <div className="mode-banner norm">
        <span className="mode-banner-icon">📊</span>
        <div>
          <strong>Normalized View</strong> — {selectedGroup === 'financial'
            ? 'Revenue divided by employee count (INR Crore/employee). Enables fair comparison of productivity across companies of different sizes.'
            : 'Values normalized by annual revenue (INR Crore). Enables fair comparison across companies of different sizes. Units displayed as value / Crore.'}
        </div>
      </div>
    )
  }
  return (
    <div className="mode-banner abs">
      <span className="mode-banner-icon">🔢</span>
      <div>
        <strong>Absolute View</strong> — Raw reported values in original units
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
      {lines.length === 0 && (
        <div className="log-line" style={{ fontStyle: 'italic', opacity: 0.5 }}>No processing details</div>
      )}
    </div>
  )
}
