import { useState } from 'react'
import type { ReactNode } from 'react'
import type { DocumentInfo, Health, Hit, Mode } from './api'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  hits?: Hit[]
  topScore?: number
  refused?: boolean
  model?: string
  latencyMs?: number
  streaming?: boolean
  error?: string
  mode?: Mode
  k?: number
}

function prettyName(filename: string): string {
  return filename.replace(/^\d+_/, '').replace(/\.txt$/, '').replace(/_/g, ' ')
}

function scoreClass(score: number): string {
  if (score >= 0.6) return 'high'
  if (score >= 0.4) return 'medium'
  return 'low'
}

/** Dependency-free rendering for the LaTeX and citation markers in the corpus. */
function texToPlain(s: string): string {
  return s
    .replace(/\\text\{([^}]*)\}/g, '$1')
    .replace(/\\frac\{([^}]*)\}\{([^}]*)\}/g, '($1) / ($2)')
    .replace(/\\times/g, '×')
    .replace(/\\leq/g, '≤')
    .replace(/\\geq/g, '≥')
    .replace(/\\sum/g, 'Σ')
    .replace(/\\[a-zA-Z]+/g, '')
    .replace(/[{}$]/g, '')
    .trim()
}

function renderText(text: string): ReactNode[] {
  return text.split(/(【[^】]+】)/g).map((part, i) =>
    part.startsWith('【')
      ? <span className="cite" key={i}>{part.slice(1, -1)}</span>
      : <span key={i}>{part}</span>,
  )
}

function renderAnswer(content: string): ReactNode[] {
  return content.split(/(\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$)/g).map((part, i) => {
    const isMath = part.startsWith('\\[') || part.startsWith('$$')
    if (!isMath) return <span key={i}>{renderText(part)}</span>
    const inner = part.replace(/^\\\[|\\\]$|^\$\$|\$\$$/g, '')
    return <div className="formula" key={i}>{texToPlain(inner)}</div>
  })
}

export function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark">R</div>
      <div>
        <div className="brand-title">Course Notes RAG</div>
        <div className="brand-sub">Week 1–8 · grounded answers</div>
      </div>
    </div>
  )
}

export function HealthPill({ health, error }: { health: Health | null; error: string }) {
  if (error) return <div className="health offline"><span className="dot" />Backend offline</div>
  if (!health) return <div className="health"><span className="dot" />Connecting…</div>
  return (
    <div className="health" title={`${health.embedding_model} · ${health.index_backend} · ${health.llm_provider}`}>
      <span className="dot" />
      {health.chunks.toLocaleString()} chunks · {health.llm_provider}
    </div>
  )
}

export function DocumentList({ docs }: { docs: DocumentInfo[] }) {
  return (
    <div className="side-section">
      <div className="side-label">Corpus <span className="count">{docs.length} files</span></div>
      <div className="doc-list">
        {docs.length === 0 && <div className="muted small">Loading documents…</div>}
        {docs.map((d) => (
          <div className="doc-row" key={d.name} title={d.name}>
            <div className="doc-icon" />
            <div className="doc-meta">
              <div className="doc-name">{prettyName(d.name)}</div>
              <div className="doc-sub">{d.chunks} chunks · {(d.characters / 1000).toFixed(1)}k chars</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function SettingsBar({
  k, setK, mode, setMode, threshold, setThreshold, disabled,
}: {
  k: number
  setK: (v: number) => void
  mode: Mode
  setMode: (v: Mode) => void
  threshold: number
  setThreshold: (v: number) => void
  disabled: boolean
}) {
  return (
    <div className="side-section settings">
      <div className="side-label">Retrieval settings</div>

      <div className="setting">
        <div className="setting-head"><span>Top-k chunks</span><b>{k}</b></div>
        <input type="range" min={1} max={10} step={1} value={k}
          disabled={disabled} onChange={(e) => setK(Number(e.target.value))} />
        <div className="range-ends"><span>1</span><span>10</span></div>
      </div>

      <div className="setting">
        <div className="setting-head"><span>Search mode</span></div>
        <div className="segmented">
          {(['semantic', 'hybrid'] as Mode[]).map((m) => (
            <button key={m} className={mode === m ? 'active' : ''}
              disabled={disabled} onClick={() => setMode(m)}>{m}</button>
          ))}
        </div>
        <div className="hint">
          {mode === 'semantic' ? 'FAISS cosine similarity only.' : 'FAISS + TF-IDF exact-word matching.'}
        </div>
      </div>

      <div className="setting">
        <div className="setting-head"><span>Answer threshold</span><b>{threshold.toFixed(2)}</b></div>
        <input type="range" min={0} max={0.9} step={0.05} value={threshold}
          disabled={disabled} onChange={(e) => setThreshold(Number(e.target.value))} />
        <div className="hint">Below this score the system says “I do not know”.</div>
      </div>
    </div>
  )
}

export function SourceCard({ hit, combined }: { hit: Hit; combined: boolean }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="source-card">
      <button className="source-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="rank">{hit.rank}</span>
        <span className="source-name">{prettyName(hit.source)}</span>
        <span className={`score ${scoreClass(hit.score)}`}>{hit.score.toFixed(3)}</span>
        <span className={`chevron ${open ? 'up' : ''}`}>⌄</span>
      </button>
      <div className="score-track">
        <div className={`score-fill ${scoreClass(hit.score)}`}
          style={{ width: `${Math.min(hit.score, 1) * 100}%` }} />
      </div>
      {open && <div className="source-body">{hit.chunk}</div>}
      {combined && open && <div className="source-note">Hybrid score combines semantic + keyword.</div>}
    </div>
  )
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  const [copied, setCopied] = useState(false)
  if (message.role === 'user') {
    return (
      <div className="msg user">
        <div className="bubble">{message.content}</div>
      </div>
    )
  }
  const combined = message.mode === 'hybrid'
  const copy = () => {
    if (!navigator.clipboard) return
    navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }
  return (
    <div className="msg assistant">
      <div className="avatar">R</div>
      <div className="msg-body">
        <div className="assistant-head">
          <span className="assistant-name">RAG assistant</span>
          {message.refused && <span className="badge refused">Outside corpus</span>}
          {message.model && message.model !== 'guardrail' && (
            <span className="badge">{message.model}</span>
          )}
          {typeof message.topScore === 'number' && (
            <span className={`badge ${scoreClass(message.topScore)}`}>
              top {message.topScore.toFixed(3)}
            </span>
          )}
          {typeof message.latencyMs === 'number' && (
            <span className="badge quiet">{(message.latencyMs / 1000).toFixed(1)}s</span>
          )}
        </div>

        <div className={`answer ${message.refused ? 'refused-text' : ''}`}>
          {message.streaming ? (
            <>
              {message.content || <span className="typing">Thinking…</span>}
              {message.content && <span className="caret" />}
            </>
          ) : message.content ? renderAnswer(message.content) : ''}
        </div>

        {message.error && <div className="inline-error">{message.error}</div>}

        {!!message.hits?.length && (
          <div className="sources">
            <div className="sources-label">
              Retrieved context <span>{message.hits.length} chunks · {message.k ?? '–'} requested</span>
            </div>
            <div className="source-grid">
              {message.hits.map((h) => <SourceCard key={h.rank} hit={h} combined={combined} />)}
            </div>
          </div>
        )}

        {message.content && !message.streaming && (
          <button className="copy-btn" onClick={copy}>{copied ? 'Copied' : 'Copy answer'}</button>
        )}
      </div>
    </div>
  )
}

export function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  const suggestions = [
    'What is precision?',
    'What does top-k mean in retrieval?',
    'How do I handle missing values in pandas?',
    'Classification vs regression?',
    'What is LangChain?',
    'What accuracy did fine-tuning reach on SMS spam, and what is RAG?',
  ]
  return (
    <div className="empty">
      <div className="empty-mark">R</div>
      <h1>Ask your course notes</h1>
      <p>
        Answers come only from the 13 Week 1–8 guides. Every reply shows the exact
        chunks it used and how closely they matched.
      </p>
      <div className="suggestions">
        {suggestions.map((q) => (
          <button key={q} onClick={() => onPick(q)}>{q}</button>
        ))}
      </div>
      <div className="empty-tip">
        Try “What is LangChain?” to see the system refuse a question the corpus cannot answer.
      </div>
    </div>
  )
}
