// src/components/KPICard.jsx
import React from 'react'
import MiniBarChart from './MiniBarChart'

const CA = '#0D6EFD'
const CB = '#198754'

function fmt(v) {
  if (v === undefined || v === null) return 'N/A'
  if (Math.abs(v) < 0.001) return v.toExponential(3)
  if (Math.abs(v) < 1)     return v.toFixed(4)
  return v.toLocaleString('en-IN', { maximumFractionDigits: 2 })
}

export default function KPICard({ comp, labelA, labelB }) {
  const { kpi_name, display_name, unit, entries, winner, pct_gap, meta = {} } = comp

  const valMap = Object.fromEntries((entries || []).map(e => [e.company_label, e.value]))
  const va = valMap[labelA]
  const vb = valMap[labelB]

  const winnerName = winner?.split(' FY')[0]
  const aWins      = winner === labelA

  return (
    <div className="card">
      {/* Header row */}
      <div className="kpi-card-header">
        <div>
          <div className="kpi-card-title">{meta.label || display_name}</div>
          <div className="kpi-card-desc">
            {meta.desc || ''}
            {meta.ratio_unit ? ` · ${meta.ratio_unit}` : ''}
          </div>
        </div>
        <div className="kpi-card-meta">
          <span className="badge badge-green">Leader: {winnerName}</span>
          <div className="kpi-gap">{pct_gap?.toFixed(1)}% gap</div>
        </div>
      </div>

      {/* Bar chart */}
      <MiniBarChart
        entries={entries}
        labelA={labelA}
        labelB={labelB}
        unit={meta.ratio_unit || unit}
        winner={winner}
      />

      {/* Value panels */}
      <div className="kpi-vals">
        <ValPanel
          name={labelA.split(' FY')[0]}
          value={va}
          unit={meta.ratio_unit || unit}
          color={CA}
          isWinner={aWins}
        />
        <ValPanel
          name={labelB.split(' FY')[0]}
          value={vb}
          unit={meta.ratio_unit || unit}
          color={CB}
          isWinner={!aWins}
        />
      </div>
    </div>
  )
}

function ValPanel({ name, value, unit, color, isWinner }) {
  return (
    <div
      className="kpi-val-panel"
      style={{ borderLeft: `3px solid ${color}` }}
    >
      <div className="kpi-val-company" style={{ color }}>
        {name}{isWinner ? ' (leader)' : ''}
      </div>
      <div className="kpi-val-number">
        {fmt(value)} <span style={{ fontSize: 11, color: 'var(--sub)' }}>{unit}</span>
      </div>
    </div>
  )
}
