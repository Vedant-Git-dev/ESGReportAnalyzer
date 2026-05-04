// src/components/MiniBarChart.jsx
import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList } from 'recharts'

const CA = '#0D6EFD'
const CB = '#198754'

function fmt(v) {
  if (v === undefined || v === null) return 'N/A'
  if (Math.abs(v) < 0.001) return v.toExponential(2)
  if (Math.abs(v) < 1)     return v.toFixed(4)
  if (Math.abs(v) >= 1e6)  return (v / 1e6).toFixed(2) + 'M'
  if (Math.abs(v) >= 1e3)  return v.toLocaleString('en-IN', { maximumFractionDigits: 1 })
  return v.toLocaleString('en-IN', { maximumFractionDigits: 2 })
}

const CustomTooltip = ({ active, payload, unit }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#fff', border: '1px solid var(--border)',
      borderRadius: 6, padding: '6px 10px', fontSize: 12,
    }}>
      <strong>{payload[0].payload.label}</strong>
      <br />
      {fmt(payload[0].value)} {unit}
    </div>
  )
}

export default function MiniBarChart({ entries, labelA, labelB, unit, winner }) {
  const data = (entries || [])
    .map((entry) => {
      const companyLabel = Array.isArray(entry)
        ? entry[0]
        : entry?.company_label ?? entry?.label ?? ''
      const rawValue = Array.isArray(entry)
        ? entry[1]
        : entry?.value ?? entry?.ratio_value ?? entry?.normalized_value
      const value = Number(rawValue)

      if (!companyLabel || Number.isNaN(value)) return null

      return {
        label: String(companyLabel).split(' FY')[0],
        companyLabel: String(companyLabel),
        value,
        isWinner: String(companyLabel) === String(winner),
      }
    })
    .filter(Boolean)

  if (!data.length) return null

  return (
    <div className="mini-chart">
      <ResponsiveContainer width="100%" height={80}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 56, bottom: 4, left: 4 }}
          barCategoryGap="30%"
        >
          <XAxis type="number" hide domain={[0, 'dataMax']} />
          <YAxis
            type="category"
            dataKey="label"
            width={82}
            tick={{ fontSize: 11, fill: 'var(--sub)', fontFamily: 'var(--font-sans)' }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<CustomTooltip unit={unit} />} cursor={{ fill: 'var(--surface)' }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
            <LabelList
              dataKey="value"
              position="right"
              formatter={(v) => fmt(v)}
              style={{ fontSize: 11, fill: 'var(--sub)', fontFamily: 'var(--font-mono)' }}
            />
            {data.map((entry, i) => (
              <Cell key={i} fill={i === 0 ? CA : CB} opacity={entry.isWinner ? 1 : 0.55} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}