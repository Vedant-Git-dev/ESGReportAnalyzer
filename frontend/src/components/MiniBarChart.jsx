// src/components/MiniBarChart.jsx
import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList } from 'recharts'

const CA = '#237A5F'
const CB = '#C47D3F'

function fmt(v) {
  if (v === undefined || v === null) return 'N/A'
  return Number(v).toLocaleString('en-IN', { maximumFractionDigits: 2 })
}

const CustomTooltip = ({ active, payload, unit }) => {
  if (!active || !payload?.length) return null
  const data = payload[0].payload
  return (
    <div style={{
      background: '#1A2320',
      border: 'none',
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 12,
      boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
    }}>
      <strong style={{ color: '#F5F7F5' }}>{data.label}</strong>
      <div style={{ color: '#8B9691', marginTop: 4 }}>
        {fmt(data.value)} <span style={{ opacity: 0.7 }}>{unit}</span>
      </div>
    </div>
  )
}

export default function MiniBarChart({ entries, unit, winner }) {
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
      <ResponsiveContainer width="100%" height={70}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 60, bottom: 4, left: 4 }}
          barCategoryGap="35%"
        >
          <XAxis type="number" hide domain={[0, 'dataMax']} />
          <YAxis
            type="category"
            dataKey="label"
            width={90}
            tick={{ fontSize: 12, fill: '#5A6662', fontFamily: 'var(--font-sans)', fontWeight: 500 }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<CustomTooltip unit={unit} />} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={24}>
            <LabelList
              dataKey="value"
              position="right"
              formatter={(v) => fmt(v)}
              style={{ fontSize: 11, fill: '#8B9691', fontFamily: 'var(--font-mono)', fontWeight: 500 }}
            />
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={i === 0 ? CA : CB}
                opacity={entry.isWinner ? 1 : 0.5}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
