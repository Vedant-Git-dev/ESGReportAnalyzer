// src/lib/api.js
// Thin client over the FastAPI backend. All fetch calls live here.

const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  health:   () => request('/health'),
  metadata: () => request('/metadata'),

  /**
   * Fetch companies from DB.
   * @param {string|null} sector  when provided, filters to that sector only
   */
  companies: (sector = null) => {
    const path = sector
      ? `/companies?sector=${encodeURIComponent(sector)}`
      : '/companies'
    return request(path)
  },

  compare: (body) =>
    request('/compare', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    }),

  exportPdf: async (body) => {
    const res = await fetch(`${BASE}/export/pdf`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    })
    if (!res.ok) throw new Error(`Export failed: ${res.statusText}`)
    return res.blob()
  },

  uploadPdf: (formData) =>
    request('/upload', { method: 'POST', body: formData }),

  /**
   * Open an SSE connection to /api/compare/stream.
   * onProgress(event: {company, message}) called for each progress event.
   * onResult(data: CompareResponse)       called once when done.
   * onError(msg: string)                  called on error.
   * onNotFound(data: {company, message})  called when no report is found.
   * Returns a close() function.
   */
  compareStream(params, { onProgress, onResult, onError, onDone, onNotFound }) {
    const qs = new URLSearchParams(params).toString()
    const es = new EventSource(`${BASE}/compare/stream?${qs}`)

    es.addEventListener('progress', (e) => {
      try { onProgress?.(JSON.parse(e.data)) } catch {}
    })
    es.addEventListener('result', (e) => {
      try { onResult?.(JSON.parse(e.data)) } catch {}
    })
    es.addEventListener('error', (e) => {
      try { onError?.(JSON.parse(e.data)?.message || 'Stream error') } catch {}
    })
    es.addEventListener('not_found', (e) => {
      try { onNotFound?.(JSON.parse(e.data)) } catch {}
    })
    es.addEventListener('done', () => {
      es.close()
      onDone?.()
    })
    es.onerror = () => {
      onError?.('Connection lost')
      es.close()
    }

    return () => es.close()
  },
}