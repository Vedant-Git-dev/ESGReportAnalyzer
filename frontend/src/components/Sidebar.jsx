// src/components/Sidebar.jsx
import React, { useRef, useEffect, useState } from 'react'

const OTHER = '__other__'

export default function Sidebar({
  health,
  metadata,
  form,
  setForm,
  onCompare,
  comparing,
  getCompaniesBySector,
}) {
  const { sectors = [], report_types = [] } = metadata
  const file1Ref = useRef(null)
  const file2Ref = useRef(null)

  const companiesInSector = getCompaniesBySector(form.sector)
  const ready = Boolean(form.company1?.trim() && form.company2?.trim())

  function setField(key, val) {
    setForm(prev => ({ ...prev, [key]: val }))
  }

  function handleSectorChange(newSector) {
    // Reset both company selections when sector changes
    setForm(prev => ({ ...prev, sector: newSector, company1: '', company2: '' }))
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
      <select value={form.sector} onChange={e => handleSectorChange(e.target.value)}>
        {sectors.map(s => <option key={s}>{s}</option>)}
      </select>

      {companiesInSector.length > 0 ? (
        <div className="sector-hint">
          {companiesInSector.length} compan{companiesInSector.length === 1 ? 'y' : 'ies'} in DB
        </div>
      ) : (
        <div className="sector-hint sector-hint-empty">
          No companies in this sector — use "Other" to enter a new one
        </div>
      )}

      <hr className="divider" style={{ marginTop: 10 }} />

      {/* Company 1 */}
      <CompanyBlock
        label="Company 1"
        colorClass="ca"
        sector={form.sector}
        company={form.company1}
        fy={form.fy1}
        file={form.file1}
        reportType={form.rtype1}
        reportTypes={report_types}
        fileRef={file1Ref}
        companiesInSector={companiesInSector}
        onCompany={v => setField('company1', v)}
        onFy={v => setField('fy1', Number(v))}
        onFile={e => handleFile('file1', e)}
        onDrop={e => handleDrop('file1', e)}
        onRtype={v => setField('rtype1', v)}
      />

      <hr className="divider" />

      {/* Company 2 */}
      <CompanyBlock
        label="Company 2"
        colorClass="cb"
        sector={form.sector}
        company={form.company2}
        fy={form.fy2}
        file={form.file2}
        reportType={form.rtype2}
        reportTypes={report_types}
        fileRef={file2Ref}
        companiesInSector={companiesInSector}
        onCompany={v => setField('company2', v)}
        onFy={v => setField('fy2', Number(v))}
        onFile={e => handleFile('file2', e)}
        onDrop={e => handleDrop('file2', e)}
        onRtype={v => setField('rtype2', v)}
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
          : 'Compare'
        }
      </button>

      {!ready && (
        <p style={{ fontSize: 11, color: 'var(--sub)', textAlign: 'center', marginTop: 8 }}>
          {form.company1?.trim()
            ? 'Select or enter Company 2'
            : form.company2?.trim()
              ? 'Select or enter Company 1'
              : 'Select both companies to compare'
          }
        </p>
      )}
    </aside>
  )
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* CompanyBlock                                                                */
/*                                                                             */
/* FIX: "Other" mode is tracked in LOCAL STATE (`isOtherMode`), completely    */
/* independent from the `company` string value. Previously, clicking "Other"  */
/* called onCompany('') which set company='' which made selectValue='' which  */
/* made showOtherInput=false — the input never appeared.                      */
/*                                                                             */
/* Now: isOtherMode=true is set the moment the user picks "Other" from the    */
/* dropdown, and stays true until they either pick a known company or the      */
/* sector/company resets from outside.                                        */
/* ─────────────────────────────────────────────────────────────────────────── */
function CompanyBlock({
  label,
  colorClass,
  sector,
  company,
  fy,
  file,
  reportType,
  reportTypes,
  fileRef,
  companiesInSector,
  onCompany,
  onFy,
  onFile,
  onDrop,
  onRtype,
}) {
  // LOCAL state: tracks whether the user has chosen "Other" from the dropdown.
  // This is separate from `company` (which holds the typed string).
  const [isOtherMode, setIsOtherMode] = useState(false)
  const otherInputRef = useRef(null)

  const knownNames = companiesInSector.map(c => c.name)

  // Exit Other mode when the actual sector changes.
  useEffect(() => {
    setIsOtherMode(false)
  }, [sector])

  // Auto-focus when Other mode activates
  useEffect(() => {
    if (isOtherMode) {
      // Small delay so the DOM element is rendered first
      setTimeout(() => otherInputRef.current?.focus(), 30)
    }
  }, [isOtherMode])

  // What the <select> currently shows
  const selectValue = isOtherMode
    ? OTHER
    : knownNames.includes(company)
      ? company
      : ''

  function handleSelectChange(e) {
    const val = e.target.value
    if (val === OTHER) {
      setIsOtherMode(true)
      // Clear company so user types fresh — do NOT call onCompany(OTHER)
      onCompany('')
    } else {
      setIsOtherMode(false)
      onCompany(val)
    }
  }

  function handleOtherInput(e) {
    onCompany(e.target.value)
  }

  function handleClearOther() {
    setIsOtherMode(false)
    onCompany('')
  }

  const isKnownSelected = !isOtherMode && selectValue !== '' && selectValue !== OTHER

  return (
    <div style={{ marginBottom: 4 }}>
      <div className={`field-label ${colorClass}`}>{label}</div>

      {/* Dropdown + FY */}
      <div className="company-row">
        <select
          className="company-select"
          value={selectValue}
          onChange={handleSelectChange}
        >
          <option value="" disabled>Select company…</option>

          {companiesInSector.length > 0
            ? companiesInSector.map(c => (
                <option key={c.id} value={c.name}>{c.name}</option>
              ))
            : <option value="" disabled>— no companies in this sector —</option>
          }

          <option disabled>──────────────</option>
          <option value={OTHER}>✏ Other — type new name…</option>
        </select>

        <input
          className="fy-input"
          type="number"
          min={2015}
          max={2030}
          value={fy}
          onChange={e => onFy(e.target.value)}
          title="Fiscal year end"
        />
      </div>

      {/* "Other" text input — shown when isOtherMode is true */}
      {isOtherMode && (
        <div className="other-input-wrap">
          <div style={{ position: 'relative' }}>
            <input
              ref={otherInputRef}
              type="text"
              className="other-input"
              placeholder="Type company name…"
              value={company}
              onChange={handleOtherInput}
            />
            {/* ✕ button to go back to dropdown */}
            <button
              onClick={handleClearOther}
              title="Go back to dropdown"
              style={{
                position: 'absolute', right: 8, top: '50%',
                transform: 'translateY(-50%)',
                background: 'none', border: 'none',
                cursor: 'pointer', color: 'var(--sub)',
                fontSize: 14, lineHeight: 1, padding: 2,
              }}
            >✕</button>
          </div>
          <div className="other-input-hint">
            {company.trim()
              ? 'ℹ If not in DB, the full pipeline will run and store results for next time.'
              : 'Start typing a company name…'
            }
          </div>
        </div>
      )}

      {/* Green badge when a known company is selected */}
      {isKnownSelected && (
        <div className="selected-company-badge">✓ {company}</div>
      )}

      {/* PDF upload zone */}
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