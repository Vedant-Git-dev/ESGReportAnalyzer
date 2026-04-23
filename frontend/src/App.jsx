// src/App.jsx
import React, { useState, useCallback, useEffect, useRef } from 'react'
import Sidebar    from './components/Sidebar'
import CompareTab from './components/CompareTab'
import UploadTab  from './components/UploadTab'
import { useAppState, useCompare } from './hooks/useApi'

const TABS = ['Comparison', 'Upload PDF']

export default function App() {
  const {
    health,
    metadata,
    companies,
    loading,
    refreshCompanies,
    getCompaniesBySector,
  } = useAppState()

  const [tab, setTab] = useState(0)

  // Sidebar form state
  const [form, setForm] = useState({
    company1: '',
    fy1:      2025,
    company2: '',
    fy2:      2024,
    sector:   'Information Technology',
    file1:    null,
    file2:    null,
    rtype1:   'BRSR',
    rtype2:   'BRSR',
  })

  const { run, cancel, state: compareState } = useCompare()
  const prevRunningRef = useRef(false)

  // Compare can create new companies via ingestion; refresh dropdown source once it finishes.
  useEffect(() => {
    if (prevRunningRef.current && !compareState.running) {
      refreshCompanies()
    }
    prevRunningRef.current = compareState.running
  }, [compareState.running, refreshCompanies])

  const handleCompare = useCallback(async () => {
    setTab(0)

    // If PDFs were attached, upload them first so the pipeline finds them in DB
    const uploadIfNeeded = async (file, company, fy, sector, rtype) => {
      if (!file) return
      const fd = new FormData()
      fd.append('file',        file)
      fd.append('company',     company)
      fd.append('fy',          String(fy))
      fd.append('sector',      sector)
      fd.append('report_type', rtype)
      try {
        await fetch('/api/upload', { method: 'POST', body: fd })
        // Refresh company list so newly ingested companies appear in dropdown
        refreshCompanies()
      } catch (e) {
        console.warn('Pre-upload failed:', e)
      }
    }

    await Promise.all([
      uploadIfNeeded(form.file1, form.company1, form.fy1, form.sector, form.rtype1),
      uploadIfNeeded(form.file2, form.company2, form.fy2, form.sector, form.rtype2),
    ])

    run({
      company1: form.company1,
      fy1:      form.fy1,
      company2: form.company2,
      fy2:      form.fy2,
      sector:   form.sector,
    })
  }, [form, run, refreshCompanies])

  if (loading) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', gap: 12, color: 'var(--sub)',
      }}>
        <span className="spinner" />
        Loading ESG Intelligence…
      </div>
    )
  }

  return (
    <div className="app-shell">
      <Sidebar
        health={health}
        metadata={metadata}
        form={form}
        setForm={setForm}
        onCompare={handleCompare}
        comparing={compareState.running}
        getCompaniesBySector={getCompaniesBySector}
      />

      <div className="main">
        <div className="tabs-bar">
          {TABS.map((t, i) => (
            <button
              key={t}
              className={`tab-btn ${tab === i ? 'active' : ''}`}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === 0 && (
          <div className="tab-content">
            <CompareTab compareState={compareState} form={form} />
          </div>
        )}

        {tab === 1 && (
          <UploadTab
            metadata={metadata}
            health={health}
            onUploadComplete={refreshCompanies}
          />
        )}
      </div>
    </div>
  )
}