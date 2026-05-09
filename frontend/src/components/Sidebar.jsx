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
  const { sectors = [] } = metadata

  const companiesInSector = getCompaniesBySector(form.sector)
  const ready = Boolean(form.company1?.trim() && form.company2?.trim())

  function setField(key, val) {
    setForm(prev => ({ ...prev, [key]: val }))
  }

  function handleSectorChange(newSector) {
    setForm(prev => ({ ...prev, sector: newSector, company1: '', company2: '' }))
  }

  return (
    <aside className="sidebar">
      {/* Sidebar Header */}
      <div className="sidebar-header">
        <div className="sidebar-logo">ESG Intelligence</div>
        <div className="sidebar-tagline">BRSR Benchmarking</div>

        <div className="status-bar">
          <span className={`status-dot ${health.db_online ? 'online' : 'offline'}`}>
            DB
          </span>
          <span className={`status-dot ${health.llm_ready ? 'online' : 'pending'}`}>
            AI
          </span>
        </div>
      </div>

      {/* Sidebar Content */}
      <div className="sidebar-content">
        {/* Sector Selection */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">Sector</div>
          <select value={form.sector} onChange={e => handleSectorChange(e.target.value)}>
            {sectors.map(s => <option key={s}>{s}</option>)}
          </select>
          {companiesInSector.length > 0 ? (
            <div className="sector-hint">
              ✓ {companiesInSector.length} compan{companiesInSector.length === 1 ? 'y' : 'ies'} in database
            </div>
          ) : (
            <div className="sector-hint sector-hint-empty">
              No companies — select "Other" below
            </div>
          )}
        </div>

        <div className="sidebar-divider" />

        {/* Company A */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">Company A</div>
          <CompanyInput
            company={form.company1}
            fy={form.fy1}
            companiesInSector={companiesInSector}
            sector={form.sector}
            onCompany={v => setField('company1', v)}
            onFy={v => setField('fy1', Number(v))}
          />
        </div>

        <div className="sidebar-divider" />

        {/* Company B */}
        <div className="sidebar-section">
          <div className="sidebar-section-title">Company B</div>
          <CompanyInput
            company={form.company2}
            fy={form.fy2}
            companiesInSector={companiesInSector}
            sector={form.sector}
            onCompany={v => setField('company2', v)}
            onFy={v => setField('fy2', Number(v))}
          />
        </div>
      </div>

      {/* Action Button */}
      <div className="sidebar-action">
        <button
          className="btn-compare"
          disabled={!ready || comparing}
          onClick={onCompare}
        >
          {comparing ? (
            <>
              <span className="spinner spinner-sm" />
              Analyzing...
            </>
          ) : (
            <>⚡ Compare Companies</>
          )}
        </button>

        {!ready && (
          <p className="btn-hint">
            Select both companies to compare
          </p>
        )}
      </div>
    </aside>
  )
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Company Input - Dropdown with FY year                                     */
/* ─────────────────────────────────────────────────────────────────────────── */

function CompanyInput({
  company,
  fy,
  companiesInSector,
  sector,
  onCompany,
  onFy,
}) {
  const [isOtherMode, setIsOtherMode] = useState(false)
  const otherInputRef = useRef(null)

  const knownNames = companiesInSector.map(c => c.name)

  useEffect(() => {
    setIsOtherMode(false)
  }, [sector])

  useEffect(() => {
    if (isOtherMode) {
      setTimeout(() => otherInputRef.current?.focus(), 30)
    }
  }, [isOtherMode])

  const selectValue = isOtherMode
    ? OTHER
    : knownNames.includes(company)
      ? company
      : ''

  function handleSelectChange(e) {
    const val = e.target.value
    if (val === OTHER) {
      setIsOtherMode(true)
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
    <div>
      {/* Dropdown + FY */}
      <div className="company-row">
        <select
          className="company-select"
          value={selectValue}
          onChange={handleSelectChange}
        >
          <option value="" disabled>Select company...</option>

          {companiesInSector.length > 0
            ? companiesInSector.map(c => (
                <option key={c.id} value={c.name}>{c.name}</option>
              ))
            : <option value="" disabled>— none in this sector —</option>
          }

          <option disabled>──────────────</option>
          <option value={OTHER}>✎ Enter custom name...</option>
        </select>

        <input
          className="fy-input"
          type="number"
          min={2015}
          max={2030}
          value={fy}
          onChange={e => onFy(e.target.value)}
          title="Fiscal Year"
        />
      </div>

      {/* Other input - shown when in "Other" mode */}
      {isOtherMode && (
        <div className="other-input-wrap">
          <div style={{ position: 'relative' }}>
            <input
              ref={otherInputRef}
              type="text"
              className="other-input"
              placeholder="Type company name..."
              value={company}
              onChange={handleOtherInput}
            />
            <button
              onClick={handleClearOther}
              title="Go back to dropdown"
              style={{
                position: 'absolute',
                right: 10,
                top: '50%',
                transform: 'translateY(-50%)',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--text-muted)',
                fontSize: 14,
                lineHeight: 1,
                padding: 2,
              }}
            >
              ✕
            </button>
          </div>
          <div className="other-input-hint">
            {company.trim()
              ? 'Will process and store for future use'
              : 'Start typing to search or add new company'
            }
          </div>
        </div>
      )}

      {/* Selected company badge */}
      {isKnownSelected && (
        <div className="selected-company-badge">✓ {company}</div>
      )}
    </div>
  )
}
