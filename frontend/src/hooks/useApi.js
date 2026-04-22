// src/hooks/useApi.js
import { useState, useEffect, useCallback } from 'react'
import { api } from '../lib/api'

/**
 * Loads health, metadata (sectors, kpi_groups, kpi_names, report_types)
 * and company list at startup. All components read from this single source.
 */
export function useAppState() {
  const [health,    setHealth]    = useState({ db_online: false, llm_ready: false, status: 'loading' })
  const [metadata,  setMetadata]  = useState({ sectors: [], report_types: [], kpi_groups: {}, kpi_names: [] })
  const [companies, setCompanies] = useState([])
  const [loading,   setLoading]   = useState(true)

  useEffect(() => {
    Promise.all([
      api.health().catch(() => ({ db_online: false, llm_ready: false, status: 'error' })),
      api.metadata().catch(() => ({ sectors: [], report_types: [], kpi_groups: {}, kpi_names: [] })),
      api.companies().catch(() => []),
    ]).then(([h, m, c]) => {
      setHealth(h)
      setMetadata(m)
      setCompanies(c)
      setLoading(false)
    })
  }, [])

  const refreshCompanies = useCallback(() => {
    api.companies().then(setCompanies).catch(() => {})
  }, [])

  return { health, metadata, companies, loading, refreshCompanies }
}

/**
 * Runs the streaming compare pipeline.
 * Returns { run, cancel, state } where state has progress + result.
 */
export function useCompare() {
  const [state, setState] = useState({
    running:   false,
    progress1: [],   // log lines for company 1
    progress2: [],   // log lines for company 2
    result:    null,
    error:     null,
  })
  const cancelRef = { current: null }

  const run = useCallback((params) => {
    setState({ running: true, progress1: [], progress2: [], result: null, error: null })

    const close = api.compareStream(params, {
      onProgress: ({ company, message }) => {
        setState(prev => {
          if (company === params.company2) {
            return { ...prev, progress2: [...prev.progress2, message] }
          }
          return { ...prev, progress1: [...prev.progress1, message] }
        })
      },
      onResult: (data) => {
        setState(prev => ({ ...prev, result: data, running: false }))
      },
      onError: (msg) => {
        setState(prev => ({ ...prev, error: msg, running: false }))
      },
      onDone: () => {
        setState(prev => ({ ...prev, running: false }))
      },
    })

    cancelRef.current = close
  }, [])

  const cancel = useCallback(() => {
    cancelRef.current?.()
    setState(prev => ({ ...prev, running: false }))
  }, [])

  return { run, cancel, state }
}

/**
 * Uploads a PDF and returns the extraction result.
 */
export function useUpload() {
  const [state, setState] = useState({
    uploading: false,
    result:    null,
    error:     null,
  })

  const upload = useCallback(async (formData) => {
    setState({ uploading: true, result: null, error: null })
    try {
      const result = await api.uploadPdf(formData)
      setState({ uploading: false, result, error: null })
      return result
    } catch (err) {
      setState({ uploading: false, result: null, error: err.message })
      return null
    }
  }, [])

  const reset = useCallback(() => {
    setState({ uploading: false, result: null, error: null })
  }, [])

  return { upload, reset, state }
}