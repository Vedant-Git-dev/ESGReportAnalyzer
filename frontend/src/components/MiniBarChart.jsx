// src/components/MiniBarChart.jsx
import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

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
  if (!entries?.length) return null

  const data = entries.map(e => ({
    label: e.company_label.split(' FY')[0],
    value: e.value,
    isWinner: e.company_label === winner,
  }))

  return (
    <div className="mini-chart">
      <ResponsiveContainer width="100%" height={72}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 48, bottom: 0, left: 0 }}
        >
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="label" width={0} hide />
          <Tooltip content={<CustomTooltip unit={unit} />} cursor={{ fill: 'var(--surface)' }} />
          <Bar dataKey="value" radius={4} barSize={18} label={labelProps(unit)}>
            {data.map((entry, i) => (
              <Cell key={i} fill={i === 0 ? CA : CB} opacity={entry.isWinner ? 1 : 0.55} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function labelProps(unit) {
  return {
    position: 'right',
    style: { fontSize: 11, fill: 'var(--sub)', fontFamily: 'var(--font-mono)' },
    formatter: (v) => `${fmt(v)}`,
  }
}