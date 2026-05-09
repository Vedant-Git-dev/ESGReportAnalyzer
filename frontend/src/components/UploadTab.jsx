// src/components/UploadTab.jsx
import React, { useRef, useState } from 'react'
import { useUpload } from '../hooks/useApi'

const MAX_MB = 50

function fmt(v) {
  if (v === undefined || v === null) return '—'
  return Number(v).toLocaleString('en-IN', { maximumFractionDigits: 2 })
}

export default function UploadTab({ metadata, health, onUploadComplete }) {
  const { sectors = [], report_types = [] } = metadata
  const fileRef = useRef(null)

  const [form, setForm] = useState({
    company: '',
    fy: 2024,
    sector: sectors[0] || 'Information Technology',
    report_type: 'BRSR',
    file: null,
  })
  const [fileErr, setFileErr] = useState('')
  const [dragOver, setDragOver] = useState(false)

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
    fd.append('file', form.file)
    fd.append('company', form.company)
    fd.append('fy', form.fy)
    fd.append('sector', form.sector)
    fd.append('report_type', form.report_type)
    await upload(fd)
    onUploadComplete?.()
  }

  const ready = Boolean(form.file && form.company?.trim() && health.db_online)
  const { uploading, result, error } = state

  return (
    <div className="upload-tab-container">
      <div className="upload-header">
        <h2 className="upload-title">Upload ESG Report</h2>
        <p className="upload-description">
          Upload a PDF to extract and store ESG metrics. Processed reports are
          immediately available for comparison in the platform.
        </p>
      </div>

      {!health.db_online && (
        <div className="alert alert-warn">
          <span className="alert-icon">⚠</span>
          <div>
            <strong>Database Offline</strong>
            <p className="alert-text">Uploads require an active database connection.</p>
          </div>
        </div>
      )}

      {/* Form Card */}
      <div className="upload-card">
        <div className="upload-form-grid">
          <div className="field-group">
            <label className="field-label">Company Name</label>
            <input
              type="text"
              placeholder="e.g. Wipro Ltd"
              value={form.company}
              onChange={e => setField('company', e.target.value)}
            />
          </div>
          <div className="field-group">
            <label className="field-label">Fiscal Year</label>
            <input
              type="number"
              min={2010}
              max={2030}
              value={form.fy}
              onChange={e => setField('fy', Number(e.target.value))}
            />
          </div>
          <div className="field-group">
            <label className="field-label">Sector</label>
            <select value={form.sector} onChange={e => setField('sector', e.target.value)}>
              {sectors.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div className="field-group">
            <label className="field-label">Report Type</label>
            <select value={form.report_type} onChange={e => setField('report_type', e.target.value)}>
              {report_types.map(t => <option key={t}>{t}</option>)}
            </select>
          </div>
        </div>

        {/* File Drop Zone */}
        <div className="field-group">
          <label className="field-label">PDF Report</label>
          <div
            className={`upload-zone ${dragOver ? 'drag-over' : ''} ${form.file ? 'has-file' : ''}`}
            onClick={() => fileRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={e => {
              e.preventDefault()
              setDragOver(false)
              const f = e.dataTransfer.files?.[0]
              if (f?.type === 'application/pdf') {
                handleFile({ target: { files: [f] } })
              }
            }}
          >
            <input type="file" accept=".pdf" ref={fileRef} onChange={handleFile} />

            {form.file ? (
              <div className="upload-file-info">
                <div className="upload-file-icon">📄</div>
                <div className="upload-file-details">
                  <div className="upload-name">{form.file.name}</div>
                  <div className="upload-zone-hint">
                    {(form.file.size / 1e6).toFixed(1)} MB • Click to change file
                  </div>
                </div>
              </div>
            ) : (
              <>
                <span className="upload-zone-icon">📁</span>
                <div className="upload-zone-text">Drop PDF here or click to browse</div>
                <div className="upload-zone-hint">Supported format: PDF • Max size: {MAX_MB} MB</div>
              </>
            )}
          </div>
          {fileErr && (
            <div className="alert alert-error" style={{ marginTop: 8 }}>
              <span className="alert-icon">⚠</span>
              <span>{fileErr}</span>
            </div>
          )}
        </div>

        <div className="upload-actions">
          <button
            className="btn-upload"
            disabled={!ready || uploading}
            onClick={handleSubmit}
          >
            {uploading ? (
              <>
                <span className="spinner spinner-sm" />
                Processing Report...
              </>
            ) : (
              <>
                <span className="btn-icon">⚡</span>
                Extract ESG Metrics
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {error && (
        <div className="alert alert-error">
          <span className="alert-icon">⚠</span>
          <div>
            <strong>Upload Failed</strong>
            <p className="alert-text">{error}</p>
          </div>
        </div>
      )}

      {result && (
        <div className="upload-results">
          <div className={`alert ${Object.keys(result.kpi_records || {}).length ? 'alert-ok' : 'alert-warn'}`}>
            <span className="alert-icon">{Object.keys(result.kpi_records || {}).length ? '✓' : '📋'}</span>
            <div>
              <strong>{result.message}</strong>
            </div>
          </div>

          {result.revenue && (
            <div className="revenue-info">
              <span className="revenue-label">Revenue extracted:</span>
              <strong className="revenue-value">INR {Number(result.revenue.value_cr).toLocaleString('en-IN', { maximumFractionDigits: 0 })} Crore</strong>
              <span className="revenue-meta">
                {result.revenue.pattern_name} • {Math.round(result.revenue.confidence * 100)}% confidence
              </span>
            </div>
          )}

          {Object.keys(result.kpi_records || {}).length > 0 && (
            <KPIResultTable kpiRecords={result.kpi_records} />
          )}

          {result.log?.length > 0 && (
            <details className="log-expander">
              <summary>Processing details ({result.log.length} steps)</summary>
              <div className="log-content">
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

/* KPI meta mapping */
const KPI_META = {
  scope_1_emissions: { label: 'Scope 1 GHG Emissions', group: 'Environmental' },
  scope_2_emissions: { label: 'Scope 2 GHG Emissions', group: 'Environmental' },
  scope_3_emissions: { label: 'Scope 3 GHG Emissions', group: 'Environmental' },
  energy_consumption: { label: 'Energy Consumption', group: 'Environmental' },
  water_consumption: { label: 'Water Consumption', group: 'Environmental' },
  waste_generated: { label: 'Waste Generated', group: 'Environmental' },
  renewable_energy_percentage: { label: 'Renewable Energy %', group: 'Environmental' },
  employee_count: { label: 'Total Workforce', group: 'Social' },
  women_in_workforce_percentage: { label: 'Women in Workforce %', group: 'Social' },
  complaints_filed: { label: 'Complaints Filed', group: 'Governance' },
  complaints_pending: { label: 'Complaints Pending', group: 'Governance' },
}

const GROUP_COLORS = {
  Environmental: 'var(--env-color)',
  Social: 'var(--soc-color)',
  Governance: 'var(--gov-color)',
}

function KPIResultTable({ kpiRecords }) {
  return (
    <div className="kpi-results">
      <div className="sec">Extracted KPIs ({Object.keys(kpiRecords).length})</div>
      <div className="table-wrapper">
        <table className="upload-result-table">
          <thead>
            <tr>
              <th>Category</th>
              <th>Metric</th>
              <th>Value</th>
              <th>Unit</th>
              <th>Method</th>
              <th>Conf.</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(kpiRecords).map(([name, rec]) => {
              const meta = KPI_META[name] || { label: name, group: 'Other' }
              const groupColor = GROUP_COLORS[meta.group] || 'var(--text-secondary)'
              return (
                <tr key={name}>
                  <td>
                    <span className="category-badge" style={{ color: groupColor, background: `${groupColor}12` }}>
                      {meta.group}
                    </span>
                  </td>
                  <td className="metric-name">{meta.label}</td>
                  <td className="metric-value">{fmt(rec.value)}</td>
                  <td className="metric-unit">{rec.unit}</td>
                  <td className="metric-method">{rec.method}</td>
                  <td>
                    <span
                      className="confidence-badge"
                      style={{ color: (rec.confidence || 0) > 0.7 ? 'var(--success)' : 'var(--warning)' }}
                    >
                      {Math.round((rec.confidence || 0) * 100)}%
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
