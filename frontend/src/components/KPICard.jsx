// src/components/KPICard.jsx
import React from 'react'
import MiniBarChart from './MiniBarChart'

const CA = '#237A5F'
const CB = '#C47D3F'

// Number formatting
function fmt(v, forLabel = false) {
  if (v === undefined || v === null || isNaN(v)) return 'N/A'
  const abs = Math.abs(v)
  if (abs === 0) return '0'
  if (abs < 0.0001) return v.toExponential(2)
  if (abs < 0.01) return v.toFixed(4)
  if (abs < 1) return v.toFixed(3)
  if (abs >= 1e9) return (v / 1e9).toFixed(2) + 'B'
  if (abs >= 1e6) return (v / 1e6).toFixed(2) + 'M'
  if (abs >= 1e4) return v.toLocaleString('en-IN', { maximumFractionDigits: 0 })
  return v.toLocaleString('en-IN', { maximumFractionDigits: 2 })
}

// Derive absolute-mode winner
function absWinner(va, vb, labelA, labelB, higherIsBetter) {
  if (va == null || vb == null) return { winner: null, pctGap: 0 }
  const aWins = higherIsBetter ? va >= vb : va <= vb
  const winner = aWins ? labelA : labelB
  const larger = Math.max(va, vb)
  const pctGap = larger > 0 ? (Math.abs(va - vb) / larger) * 100 : 0
  return { winner, pctGap }
}

// ─────────────────────────────────────────────────────────────────────────────
// KPICard
// ─────────────────────────────────────────────────────────────────────────────
export default function KPICard({ comp, labelA, labelB, normalized, rawA, rawB }) {
  const { kpi_name, display_name, unit, entries, winner, pct_gap, meta = {} } = comp

  const higherIsBetter = meta.higher_is_better ?? false
  const ratioDenominator = meta.ratio_denominator || 'revenue'
  const isRatioless = ratioDenominator === 'none'

  // Normalized mode values
  const normMap = Object.fromEntries((entries || []).map(e => [e.company_label, e.value]))
  const normA = normMap[labelA]
  const normB = normMap[labelB]
  const normUnit = isRatioless ? (meta.unit || unit) : (meta.ratio_unit || unit)

  // Absolute mode values
  const absA = rawA?.value ?? null
  const absB = rawB?.value ?? null
  const absUnit = rawA?.unit || rawB?.unit || meta.unit || unit

  // Pick display values
  const showNorm = normalized && !isRatioless
  const displayA = showNorm ? normA : (isRatioless ? normA : (absA ?? normA))
  const displayB = showNorm ? normB : (isRatioless ? normB : (absB ?? normB))
  const displayUnit = showNorm
    ? normUnit
    : (isRatioless ? normUnit : ((absA != null || absB != null) ? absUnit : normUnit))

  // Winner & gap
  let displayWinner, displayGap
  if (showNorm || isRatioless) {
    displayWinner = winner
    displayGap = pct_gap
  } else {
    const { winner: w, pctGap } = absWinner(absA, absB, labelA, labelB, higherIsBetter)
    displayWinner = w
    displayGap = pctGap
  }

  const winnerName = displayWinner?.split(' FY')[0] || '—'
  const aWins = displayWinner === labelA
  const noData = displayA == null && displayB == null

  const chartData = [
    { label: labelA.split(' FY')[0], value: displayA, isWinner: aWins },
    { label: labelB.split(' FY')[0], value: displayB, isWinner: !aWins },
  ].filter(d => d.value != null)

  return (
    <div className="card">
      {/* Header */}
      <div className="kpi-card-header">
        <div className="kpi-card-info">
          <div className="kpi-card-title">{meta.label || display_name}</div>
          <div className="kpi-card-desc">
            {showNorm
              ? (meta.desc || `${meta.label || display_name} per INR Crore revenue`)
              : (isRatioless
                  ? (meta.desc || '')
                  : `Absolute reported value · ${displayUnit}`)
            }
          </div>
        </div>
        <div className="kpi-card-meta">
          {noData ? (
            <span className="badge badge-amber" style={{ fontSize: 10 }}>No data</span>
          ) : (
            <>
              <span className="leader-badge">Leader: {winnerName}</span>
              {displayGap != null && displayGap > 0 && (
                <div className="kpi-gap">{displayGap.toFixed(1)}% gap</div>
              )}
            </>
          )}
          {isRatioless && <div className="kpi-same-badge">Same in both modes</div>}
        </div>
      </div>

      {/* Chart */}
      {!noData && (
        <MiniBarChart
          entries={[
            { company_label: labelA, value: displayA },
            { company_label: labelB, value: displayB },
          ].filter(d => d.value != null)}
          unit={displayUnit}
          winner={displayWinner}
        />
      )}

      {noData && (
        <div className="kpi-no-data">
          No extracted data available for this metric
        </div>
      )}

      {/* Value Panels */}
      <div className="kpi-vals">
        <ValPanel
          name={labelA.split(' FY')[0]}
          value={displayA}
          unit={displayUnit}
          color={CA}
          isWinner={aWins && !noData}
          normalized={showNorm}
          isRatioless={isRatioless}
        />
        <ValPanel
          name={labelB.split(' FY')[0]}
          value={displayB}
          unit={displayUnit}
          color={CB}
          isWinner={!aWins && !noData}
          normalized={showNorm}
          isRatioless={isRatioless}
        />
      </div>
    </div>
  )
}

// ValPanel
function ValPanel({ name, value, unit, color, isWinner, normalized, isRatioless }) {
  const modeTag = isRatioless
    ? null
    : normalized
      ? <span className="val-mode-tag val-mode-norm">normalized</span>
      : <span className="val-mode-tag val-mode-abs">absolute</span>

  return (
    <div className="kpi-val-panel" style={{ borderLeftColor: color }}>
      <div className="kpi-val-company" style={{ color }}>
        {name}
        {isWinner && <span style={{ fontWeight: 700 }}> (leader)</span>}
        {modeTag}
      </div>
      <div className="kpi-val-number">
        {value != null ? (
          <>
            {fmt(value)}
            <span className="kpi-val-unit">{unit}</span>
          </>
        ) : (
          <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: 14 }}>No data</span>
        )}
      </div>
    </div>
  )
}
