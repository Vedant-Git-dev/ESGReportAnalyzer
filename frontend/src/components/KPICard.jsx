// src/components/KPICard.jsx
import React from 'react'
import MiniBarChart from './MiniBarChart'

const CA = '#237A5F'
const CB = '#C47D3F'

// Financial KPIs that normalize by employee count instead of revenue
const FINANCIAL_NORMALIZED_KPIS = ['revenue_from_operations', 'net_revenue']

// Number formatting
function fmt(v, forLabel = false) {
  if (v === undefined || v === null || isNaN(v)) return 'N/A'
  return Number(v).toLocaleString('en-IN', { maximumFractionDigits: 2 })
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
  const isFinancialNorm = FINANCIAL_NORMALIZED_KPIS.includes(kpi_name)

  // Normalized mode values
  const normMap = Object.fromEntries((entries || []).map(e => [e.company_label, e.value]))
  const normA = normMap[labelA]
  const normB = normMap[labelB]
  const normUnit = isRatioless ? (meta.unit || unit) : (meta.ratio_unit || unit)

  // Absolute mode values
  const absA = rawA?.value ?? null
  const absB = rawB?.value ?? null
  const absUnit = rawA?.unit || rawB?.unit || meta.unit || unit

  // Compute per-employee normalized values for Financial KPIs
  const empA = rawA && rawA.value != null && rawA.employee_count
    ? rawA.value / rawA.employee_count
    : (rawA?.value_per_employee ?? null)
  const empB = rawB && rawB.value != null && rawB.employee_count
    ? rawB.value / rawB.employee_count
    : (rawB?.value_per_employee ?? null)

  // Pick display values
  const showNorm = normalized && !isRatioless && !isFinancialNorm

  let displayA, displayB, displayUnit
  if (isFinancialNorm && normalized) {
    // Normalized: per employee view for Financial KPIs
    const empNormA = rawA?.value_per_employee ?? empA
    const empNormB = rawB?.value_per_employee ?? empB
    displayA = empNormA
    displayB = empNormB
    displayUnit = 'INR_Crore/employee'
  } else if (isFinancialNorm) {
    // Absolute: show raw INR Crore values
    displayA = absA
    displayB = absB
    displayUnit = 'INR_Crore'
  } else if (showNorm || isRatioless) {
    displayA = normA
    displayB = normB
    displayUnit = normUnit
  } else {
    displayA = absA ?? normA
    displayB = absB ?? normB
    displayUnit = (absA != null || absB != null) ? absUnit : normUnit
  }

  // Winner & gap
  let displayWinner, displayGap
  if (isFinancialNorm && normalized) {
    // Financial normalized: use per-employee values to compute winner
    const { winner: w, pctGap } = absWinner(displayA, displayB, labelA, labelB, higherIsBetter)
    displayWinner = w
    displayGap = pctGap
  } else if (isFinancialNorm) {
    // Financial absolute: use raw values to compute winner
    const { winner: w, pctGap } = absWinner(absA, absB, labelA, labelB, higherIsBetter)
    displayWinner = w
    displayGap = pctGap
  } else if (showNorm || isRatioless) {
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

  return (
    <div className="card">
      {/* Header */}
      <div className="kpi-card-header">
        <div className="kpi-card-info">
          <div className="kpi-card-title">{meta.label || display_name}</div>
          <div className="kpi-card-desc">
            {isFinancialNorm
              ? (normalized ? 'Revenue per employee (INR Crore/employee)' : 'Absolute revenue (INR Crore)')
              : showNorm
                ? (meta.desc || `${meta.label || display_name} per INR Crore revenue`)
                : (isRatioless
                    ? (meta.desc || '')
                    : `Absolute reported value - ${displayUnit}`)}
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
          normalized={isFinancialNorm ? normalized : showNorm}
          isRatioless={isRatioless}
        />
        <ValPanel
          name={labelB.split(' FY')[0]}
          value={displayB}
          unit={displayUnit}
          color={CB}
          isWinner={!aWins && !noData}
          normalized={isFinancialNorm ? normalized : showNorm}
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