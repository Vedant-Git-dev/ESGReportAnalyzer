// src/App.jsx
import React, { useState, useCallback, useEffect, useRef } from 'react'
import Sidebar from './components/Sidebar'
import CompareTab from './components/CompareTab'
import UploadTab from './components/UploadTab'
import { useAppState, useCompare } from './hooks/useApi'

const TABS = [
  { id: 'compare', label: 'Comparison', icon: '⚖' },
  { id: 'upload', label: 'Upload PDF', icon: '📄' },
]

export default function App() {
  const {
    health,
    metadata,
    companies,
    loading,
    refreshCompanies,
    getCompaniesBySector,
  } = useAppState()

  const [tab, setTab] = useState('compare')

  // Sidebar form state
  const [form, setForm] = useState({
    company1: '',
    fy1: 2025,
    company2: '',
    fy2: 2024,
    sector: 'Information Technology',
  })

  const { run, cancel, state: compareState } = useCompare()
  const prevRunningRef = useRef(false)

  useEffect(() => {
    if (prevRunningRef.current && !compareState.running) {
      refreshCompanies()
    }
    prevRunningRef.current = compareState.running
  }, [compareState.running, refreshCompanies])

  const handleCompare = useCallback(async () => {
    setTab('compare')

    run({
      company1: form.company1,
      fy1: form.fy1,
      company2: form.company2,
      fy2: form.fy2,
      sector: form.sector,
    })
  }, [form, run])

  if (loading) {
    return (
      <div className="loading-state" style={{ height: '100vh' }}>
        <div className="spinner" style={{ width: 28, height: 28 }} />
        <span className="loading-text">Initializing ESG Intelligence...</span>
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
        <div className="top-bar">
          <div className="top-bar-tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={`tab-btn ${tab === t.id ? 'active' : ''}`}
                onClick={() => setTab(t.id)}
              >
                <span className="tab-icon">{t.icon}</span>
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <div className="content-panel">
          {tab === 'compare' && (
            <CompareTab compareState={compareState} form={form} />
          )}

          {tab === 'upload' && (
            <UploadTab
              metadata={metadata}
              health={health}
              onUploadComplete={refreshCompanies}
            />
          )}
        </div>
      </div>
    </div>
  )
}
