export type Mode = 'semantic' | 'hybrid'

export interface Hit {
  rank: number
  score: number
  source: string
  chunk: string
}

export interface AskRequest {
  question: string
  k: number
  mode: Mode
  threshold?: number
}

export interface AskResponse {
  answer: string
  hits: Hit[]
  top_score: number
  refused: boolean
  model: string
  mode: Mode
  k: number
  latency_ms: number
}

export interface SearchResponse {
  question: string
  hits: Hit[]
  top_score: number
  mode: Mode
}

export interface DocumentInfo {
  name: string
  characters: number
  chunks: number
}

export interface Health {
  status: string
  chunks: number
  documents: number
  embedding_model: string
  embedding_dim: number
  index_backend: string
  llm_provider: string
  llm_model: string
  threshold: number
}

export type StreamEvent =
  | { type: 'meta'; hits: Hit[]; top_score: number; refused: boolean; model: string; mode: Mode; k: number }
  | { type: 'token'; text: string }
  | { type: 'error'; message: string }
  | { type: 'done'; answer: string; model: string; refused: boolean; top_score: number; latency_ms: number }

const BASE = import.meta.env.VITE_API_URL ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const getHealth = () => request<Health>('/api/health')
export const getDocuments = () => request<DocumentInfo[]>('/api/documents')

export const search = (payload: AskRequest) =>
  request<SearchResponse>('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

export const ask = (payload: AskRequest) =>
  request<AskResponse>('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

/** Streams SSE events. Falls back to the non-streaming endpoint in App on failure. */
export async function streamAsk(
  payload: AskRequest,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/api/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
    signal,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  if (!res.body) throw new Error('Streaming is not available in this browser')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.startsWith('data:')) continue
      const raw = line.slice(5).trim()
      if (!raw) continue
      try {
        onEvent(JSON.parse(raw) as StreamEvent)
      } catch {
        /* ignore malformed keepalive frames */
      }
    }
  }
}
