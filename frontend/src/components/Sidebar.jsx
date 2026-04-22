// src/components/Sidebar.jsx
import React, { useRef } from 'react'

export default function Sidebar({
  health,
  metadata,
  form,
  setForm,
  onCompare,
  comparing,
}) {
  const { sectors = [], report_types = [] } = metadata
  const file1Ref = useRef(null)
  const file2Ref = useRef(null)

  const ready = Boolean(form.company1?.trim() && form.company2?.trim())

  function setField(key, val) {
    setForm(prev => ({ ...prev, [key]: val }))
  }

  function handleFile(key, e) {
    const f = e.target.files?.[0]
    if (f) setField(key, f)
  }

  function handleDrop(key, e) {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0]
    if (f?.type === 'application/pdf') setField(key, f)
  }

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div>
        <div className="sidebar-logo">ESG Intelligence</div>
        <div className="sidebar-sub">Multi-Source Benchmarking</div>
      </div>

      {/* Status pills */}
      <div className="status-pills">
        <span className={`pill ${health.db_online ? 'pill-green' : 'pill-red'}`}>
          ● {health.db_online ? 'DB Online' : 'DB Offline'}
        </span>
        <span className={`pill ${health.llm_ready ? 'pill-green' : 'pill-amber'}`}>
          ● {health.llm_ready ? 'AI Ready' : 'No AI Key'}
        </span>
      </div>

      <hr className="divider" />

      {/* Sector */}
      <div className="field-label">Sector</div>
      <select value={form.sector} onChange={e => setField('sector', e.target.value)}>
        {sectors.map(s => <option key={s}>{s}</option>)}
      </select>

      <hr className="divider" style={{ marginTop: 14 }} />

      {/* Company 1 */}
      <CompanyBlock
        label="Company 1"
        colorClass="ca"
        company={form.company1}
        fy={form.fy1}
        file={form.file1}
        reportType={form.rtype1}
        reportTypes={report_types}
        fileRef={file1Ref}
        onCompany={v => setField('company1', v)}
        onFy={v => setField('fy1', Number(v))}
        onFile={e => handleFile('file1', e)}
        onDrop={e => handleDrop('file1', e)}
        onRtype={v => setField('rtype1', v)}
        placeholder="e.g. Infosys"
      />

      <hr className="divider" />

      {/* Company 2 */}
      <CompanyBlock
        label="Company 2"
        colorClass="cb"
        company={form.company2}
        fy={form.fy2}
        file={form.file2}
        reportType={form.rtype2}
        reportTypes={report_types}
        fileRef={file2Ref}
        onCompany={v => setField('company2', v)}
        onFy={v => setField('fy2', Number(v))}
        onFile={e => handleFile('file2', e)}
        onDrop={e => handleDrop('file2', e)}
        onRtype={v => setField('rtype2', v)}
        placeholder="e.g. TCS"
      />

      <hr className="divider" />

      <button
        className="btn btn-primary"
        disabled={!ready || comparing}
        onClick={onCompare}
        style={{ marginTop: 4 }}
      >
        {comparing
          ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Comparing…</>
          : 'Compare'}
      </button>

      {!ready && (
        <p style={{ fontSize: 11, color: 'var(--sub)', textAlign: 'center', marginTop: 8 }}>
          Enter both company names to compare
        </p>
      )}
    </aside>
  )
}

/* ── Per-company block ── */
function CompanyBlock({
  label, colorClass, company, fy, file, reportType, reportTypes,
  fileRef, onCompany, onFy, onFile, onDrop, onRtype, placeholder,
}) {
  return (
    <div style={{ marginBottom: 4 }}>
      <div className={`field-label ${colorClass}`}>{label}</div>

      <div className="input-row" style={{ marginBottom: 6 }}>
        <input
          className="grow"
          type="text"
          placeholder={placeholder}
          value={company}
          onChange={e => onCompany(e.target.value)}
        />
        <input
          className="fy-field"
          type="number"
          min={2015}
          max={2030}
          value={fy}
          onChange={e => onFy(e.target.value)}
        />
      </div>

      {/* PDF drop zone */}
      <div
        className={`upload-zone ${file ? 'drag-over' : ''}`}
        onDragOver={e => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
      >
        <input type="file" accept=".pdf" ref={fileRef} onChange={onFile} />
        {file
          ? <div className="upload-name">📄 {file.name} ({(file.size / 1e6).toFixed(1)} MB)</div>
          : <div className="upload-zone-text">Upload PDF (optional)</div>
        }
      </div>

      {file && (
        <div style={{ marginTop: 4 }}>
          <select value={reportType} onChange={e => onRtype(e.target.value)}>
            {reportTypes.map(t => <option key={t}>{t}</option>)}
          </select>
        </div>
      )}
    </div>
  )
}
