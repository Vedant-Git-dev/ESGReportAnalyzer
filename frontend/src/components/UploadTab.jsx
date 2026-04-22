// src/components/UploadTab.jsx
import React, { useRef, useState } from 'react'
import { useUpload } from '../hooks/useApi'

const MAX_MB = 50

function fmt(v) {
  if (v === undefined || v === null) return '—'
  return Number(v).toLocaleString('en-IN', { maximumFractionDigits: 2 })
}

export default function UploadTab({ metadata, health }) {
  const { sectors = [], report_types = [] } = metadata
  const fileRef = useRef(null)

  const [form, setForm] = useState({
    company:     '',
    fy:          2024,
    sector:      sectors[0] || 'Information Technology',
    report_type: 'BRSR',
    file:        null,
  })
  const [fileErr, setFileErr] = useState('')

  const { upload, reset, state } = useUpload()

  function setField(k, v) { setForm(prev => ({ ...prev, [k]: v })) }

  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    if (f.size > MAX_MB * 1e6) {
      setFileErr(`File is ${(f.size / 1e6).toFixed(1)} MB — max ${MAX_MB} MB.`)
      return
    }
    setFileErr('')
    setField('file', f)
  }

  async function handleSubmit() {
    if (!form.file || !form.company) return
    reset()
    const fd = new FormData()
    fd.append('file',        form.file)
    fd.append('company',     form.company)
    fd.append('fy',          form.fy)
    fd.append('sector',      form.sector)
    fd.append('report_type', form.report_type)
    await upload(fd)
  }

  const ready = Boolean(form.file && form.company?.trim() && health.db_online)

  const { uploading, result, error } = state

  return (
    <div className="tab-content">
      <div className="sec" style={{ marginTop: 0 }}>Upload ESG / BRSR Report PDF</div>
      <p style={{ fontSize: 13, color: 'var(--sub)', marginBottom: 20 }}>
        Upload a PDF to extract and store ESG metrics. Processed reports are
        immediately available in the Comparison tab.
      </p>

      {!health.db_online && (
        <div className="alert alert-warn" style={{ marginBottom: 16 }}>
          Database is offline. Uploads require a database connection.
        </div>
      )}

      {/* Form */}
      <div className="upload-form">
        <div className="upload-form-grid">
          <div>
            <div className="field-label">Company name</div>
            <input
              type="text"
              placeholder="e.g. Wipro"
              value={form.company}
              onChange={e => setField('company', e.target.value)}
            />
          </div>
          <div>
            <div className="field-label">Fiscal year</div>
            <input
              type="number"
              min={2010} max={2030}
              value={form.fy}
              onChange={e => setField('fy', Number(e.target.value))}
            />
          </div>
          <div>
            <div className="field-label">Sector</div>
            <select value={form.sector} onChange={e => setField('sector', e.target.value)}>
              {sectors.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <div className="field-label">Report type</div>
            <select value={form.report_type} onChange={e => setField('report_type', e.target.value)}>
              {report_types.map(t => <option key={t}>{t}</option>)}
            </select>
          </div>
        </div>

        {/* File picker */}
        <div className="field-label">PDF file</div>
        <div
          className="upload-zone"
          onClick={() => fileRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => {
            e.preventDefault()
            const f = e.dataTransfer.files?.[0]
            if (f?.type === 'application/pdf') {
              handleFile({ target: { files: [f] } })
            }
          }}
        >
          <input type="file" accept=".pdf" ref={fileRef} onChange={handleFile} />
          {form.file
            ? <div className="upload-name">
                ✅ {form.file.name} ({(form.file.size / 1e6).toFixed(1)} MB)
              </div>
            : <div className="upload-zone-text">
                Click or drag a PDF here
              </div>
          }
        </div>
        {fileErr && <div className="alert alert-error" style={{ marginTop: 6 }}>{fileErr}</div>}

        <button
          className="btn btn-primary"
          style={{ marginTop: 14 }}
          disabled={!ready || uploading}
          onClick={handleSubmit}
        >
          {uploading
            ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Processing…</>
            : 'Process Upload'
          }
        </button>
      </div>

      {/* Results */}
      {error && (
        <div className="alert alert-error" style={{ marginTop: 16 }}>
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 20, maxWidth: 680 }}>
          <div className={`alert ${Object.keys(result.kpi_records || {}).length ? 'alert-ok' : 'alert-warn'}`}>
            {result.message}
          </div>

          {result.revenue && (
            <p style={{ fontSize: 13, color: 'var(--sub)', marginTop: 8 }}>
              Revenue: INR {Number(result.revenue.value_cr).toLocaleString('en-IN', { maximumFractionDigits: 0 })} Crore
              [{result.revenue.pattern_name}, {Math.round(result.revenue.confidence * 100)}% confidence]
            </p>
          )}

          {Object.keys(result.kpi_records || {}).length > 0 && (
            <KPIResultTable kpiRecords={result.kpi_records} />
          )}

          {/* Log */}
          {result.log?.length > 0 && (
            <details className="log-expander" style={{ marginTop: 16 }}>
              <summary>Processing details ({result.log.length} steps)</summary>
              <div style={{ marginTop: 8 }}>
                {result.log.map((line, i) => (
                  <div key={i} className="log-line">{line}</div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

/* ── KPI results table ── */
const KPI_META = {
  scope_1_emissions:             { label: 'Scope 1 GHG',         group: 'Environmental' },
  scope_2_emissions:             { label: 'Scope 2 GHG',         group: 'Environmental' },
  scope_3_emissions:             { label: 'Scope 3 GHG',         group: 'Environmental' },
  energy_consumption:            { label: 'Energy Intensity',    group: 'Environmental' },
  water_consumption:             { label: 'Water Intensity',     group: 'Environmental' },
  waste_generated:               { label: 'Waste Intensity',     group: 'Environmental' },
  renewable_energy_percentage:   { label: 'Renewable Energy',    group: 'Environmental' },
  employee_count:                { label: 'Workforce',           group: 'Social'        },
  women_in_workforce_percentage: { label: 'Women in Workforce',  group: 'Social'        },
  complaints_filed:              { label: 'Complaints Filed',    group: 'Governance'    },
  complaints_pending:            { label: 'Complaints Pending',  group: 'Governance'    },
}

function KPIResultTable({ kpiRecords }) {
  return (
    <table className="upload-result-table">
      <thead>
        <tr>
          <th>Group</th>
          <th>Metric</th>
          <th>Value</th>
          <th>Unit</th>
          <th>Method</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(kpiRecords).map(([name, rec]) => {
          const meta = KPI_META[name] || { label: name, group: '—' }
          return (
            <tr key={name}>
              <td>{meta.group}</td>
              <td>{meta.label}</td>
              <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                {fmt(rec.value)}
              </td>
              <td>{rec.unit}</td>
              <td>{rec.method}</td>
              <td>{Math.round((rec.confidence || 0) * 100)}%</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
