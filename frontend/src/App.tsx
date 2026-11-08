import { useEffect, useRef, useState } from 'react'
import { ask, getDocuments, getHealth, streamAsk } from './api'
import type { DocumentInfo, Health, Mode } from './api'
import {
  Brand, DocumentList, EmptyState, HealthPill, MessageBubble, SettingsBar,
} from './components'
import type { ChatMessage } from './components'

const uid = () => Math.random().toString(36).slice(2, 10)

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState('')
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [k, setK] = useState(3)
  const [mode, setMode] = useState<Mode>('semantic')
  const [threshold, setThreshold] = useState(0.4)
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [busy, setBusy] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const autoSentRef = useRef(false)

  useEffect(() => {
    getHealth().then(setHealth).catch((e: Error) => setHealthError(e.message))
    getDocuments().then(setDocs).catch(() => setDocs([]))
    // Shareable deep link, e.g. /?q=What%20is%20precision%3F
    const q = new URLSearchParams(window.location.search).get('q')
    if (q && !autoSentRef.current) {
      autoSentRef.current = true
      void send(q)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  async function send(question?: string) {
    const q = (question ?? input).trim()
    if (!q || busy) return
    setInput('')
    setBusy(true)
    setSidebarOpen(false)

    const assistantId = uid()
    setMessages((prev) => [
      ...prev,
      { id: uid(), role: 'user', content: q },
      { id: assistantId, role: 'assistant', content: '', streaming: true },
    ])
    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)))

    const payload = { question: q, k, mode, threshold }
    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamAsk(payload, (event) => {
        switch (event.type) {
          case 'meta':
            patch((m) => ({
              ...m, hits: event.hits, topScore: event.top_score, refused: event.refused,
              model: event.model, mode: event.mode, k: event.k,
            }))
            break
          case 'token':
            patch((m) => ({ ...m, content: m.content + event.text }))
            break
          case 'error':
            patch((m) => ({ ...m, error: event.message }))
            break
          case 'done':
            patch((m) => ({
              ...m, streaming: false, latencyMs: event.latency_ms,
              model: event.model || m.model, refused: event.refused, topScore: event.top_score,
            }))
            break
        }
      }, controller.signal)
      patch((m) => ({ ...m, streaming: false }))
    } catch (err) {
      if (controller.signal.aborted) {
        patch((m) => ({ ...m, streaming: false, content: m.content || 'Stopped.' }))
      } else {
        try {
          const r = await ask(payload)
          patch((m) => ({
            ...m, content: r.answer, hits: r.hits, topScore: r.top_score,
            refused: r.refused, model: r.model, latencyMs: r.latency_ms,
            streaming: false, mode: r.mode, k: r.k,
          }))
        } catch (fallbackErr) {
          patch((m) => ({
            ...m, streaming: false,
            error: `Could not reach the RAG API. ${(fallbackErr as Error).message}${err instanceof Error ? ` (${err.message})` : ''}`,
          }))
        }
      }
    } finally {
      setBusy(false)
      abortRef.current = null
      textareaRef.current?.focus()
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <Brand />
        <DocumentList docs={docs} />
        <SettingsBar k={k} setK={setK} mode={mode} setMode={setMode}
          threshold={threshold} setThreshold={setThreshold} disabled={busy} />
        <div className="side-footer">
          <div className="small muted">Local embeddings · all-MiniLM-L6-v2</div>
          <div className="small muted">Grounded answers · low top score refuses</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="icon-btn menu-btn" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="Toggle settings">☰</button>
          <HealthPill health={health} error={healthError} />
          <div className="topbar-actions">
            <button className="ghost-btn" onClick={() => setMessages([])} disabled={busy || !messages.length}>
              New chat
            </button>
          </div>
        </header>

        <div className="chat-scroll">
          {messages.length === 0 ? (
            <EmptyState onPick={(q) => void send(q)} />
          ) : (
            <div className="thread">
              {messages.map((m) => <MessageBubble key={m.id} message={m} />)}
              <div ref={endRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          {healthError && (
            <div className="banner">
              Backend offline — start it with <code>uvicorn app.main:app --port 8000</code>
            </div>
          )}
          <div className="composer">
            <textarea
              ref={textareaRef}
              value={input}
              rows={1}
              placeholder="Ask anything from your course notes…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  void send()
                }
              }}
              disabled={busy}
            />
            {busy ? (
              <button className="send-btn stop" onClick={() => abortRef.current?.abort()} title="Stop">■</button>
            ) : (
              <button className="send-btn" onClick={() => void send()} disabled={!input.trim()} title="Send">↑</button>
            )}
          </div>
          <div className="composer-hint">
            Enter to send · Shift+Enter for a new line · k={k} · {mode} · threshold {threshold.toFixed(2)}
          </div>
        </div>
      </main>
    </div>
  )
}
