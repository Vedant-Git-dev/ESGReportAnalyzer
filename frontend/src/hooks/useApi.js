// src/hooks/useApi.js
import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../lib/api'

/**
 * Loads health, metadata and ALL companies at startup.
 * getCompaniesBySector(sector) filters the in-memory list — no extra
 * network call needed on every sector change.
 */
export function useAppState() {
  const [health,    setHealth]    = useState({ db_online: false, llm_ready: false, status: 'loading' })
  const [metadata,  setMetadata]  = useState({ sectors: [], report_types: [], kpi_groups: {}, kpi_names: [] })
  const [companies, setCompanies] = useState([])   // all companies, all sectors
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

  /**
   * Filter the already-loaded company list by sector (client-side, instant).
   * Returns all companies when sector is empty/null.
   */
  const getCompaniesBySector = useCallback((sector) => {
    if (!sector) return companies
    const wanted = sector.trim().toLowerCase()
    return companies.filter(
      c => c.sector?.trim().toLowerCase() === wanted
    )
  }, [companies])

  /**
   * Re-fetch companies from the server (e.g. after a new company is ingested
   * via the pipeline so it appears in future dropdowns immediately).
   */
  const refreshCompanies = useCallback(() => {
    api.companies().then(setCompanies).catch(() => {})
  }, [])

  return { health, metadata, companies, loading, refreshCompanies, getCompaniesBySector }
}

/**
 * Runs the streaming compare pipeline.
 * Returns { run, cancel, state } where state has progress + result.
 */
export function useCompare() {
  const [state, setState] = useState({
    running:   false,
    progress1: [],
    progress2: [],
    result:    null,
    error:     null,
  })
  // Store close function in a ref so cancel() always has the latest reference
  const closeRef = useRef(null)

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
      onNotFound: (data) => {
        setState(prev => ({ ...prev, error: data.message, running: false, notFound: data }))
      },
      onDone: () => {
        setState(prev => ({ ...prev, running: false }))
      },
    })

    closeRef.current = close
  }, [])

  const cancel = useCallback(() => {
    closeRef.current?.()
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